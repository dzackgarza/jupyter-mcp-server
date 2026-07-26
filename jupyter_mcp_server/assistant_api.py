# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""FastAPI adapter exposing existing Jupyter tool classes as a REST API.

This is a thin HTTP/OpenAPI transport adapter over the existing
``jupyter_mcp_server.tools`` implementations.  It does not import
``jupyter_mcp_server.server``; it imports tool classes directly and
calls their ``execute`` methods.

The adapter is designed for GPT Actions: stateless HTTP contract (every
request names the notebook), one Uvicorn worker, deterministic
``nb_<base64>`` notebook IDs, and ``x-openai-isConsequential: false`` on
mutation endpoints so the GPT can use "always allow" behavior.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from jupyter_mcp_server.assistant_runtime import (
    AssistantRuntime,
    configure_jupyter,
)
from jupyter_mcp_server.config import get_config
from jupyter_mcp_server.notebook_id import decode_notebook_id, encode_notebook_id
from jupyter_mcp_server.tools import (
    ClearCellOutputTool,
    DeleteCellTool,
    EditCellSourceTool,
    ExecuteCellTool,
    ExecuteCodeTool,
    InsertCellTool,
    ListFilesTool,
    ListKernelsTool,
    MoveCellTool,
    OverwriteCellSourceTool,
    ReadCellTool,
    ReadNotebookTool,
    RestartNotebookTool,
    UnuseNotebookTool,
)
from jupyter_mcp_server.utils import (
    safe_extract_outputs,
    safe_notebook_operation,
    wait_for_kernel_idle,
)

# ---------------------------------------------------------------------------
# App + runtime
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Jupyter Assistant API",
    version="0.1.0",
    description=(
        "Thin HTTP transport over the existing Jupyter MCP Server tool "
        "implementations.  A notebook is identified by a deterministic "
        "nb_<base64> ID derived from its Jupyter-root-relative filepath."
    ),
    servers=[
        {"url": os.getenv("ASSISTANT_API_SERVER_URL", "https://jupyter-assistant.dzackgarza.com")},
    ],
)

runtime = AssistantRuntime()

_CONSEQUENTIAL_FALSE: dict[str, Any] = {"x-openai-isConsequential": False}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    ok: bool = False
    http_status: int
    error_type: str
    error_message: str
    traceback: str | None = None
    notebook_id: str | None = None
    notebook_path: str | None = None


def _describe(exc: BaseException) -> str:
    """Render an exception and its ``raise ... from`` chain as one line.

    ``str(exc)`` is empty for argless exceptions (``NotImplementedError()``)
    and drops the root cause for wrapped ones, so walk ``__cause__`` and fall
    back to the qualified type name when there is no message.
    """
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        module = type(cur).__module__
        name = type(cur).__name__
        qualified = name if module in ("builtins", "") else f"{module}.{name}"
        message = str(cur)
        parts.append(f"{qualified}: {message}" if message else qualified)
        cur = cur.__cause__ or cur.__context__
    return "\n  caused by: ".join(parts)


def _boundary_status(exc: BaseException) -> int | None:
    """Return the HTTP status a wrapped Jupyter-boundary failure carried.

    ``jupyter_server_client`` raises typed errors carrying ``.status_code``
    (404 for a missing path, 403, …).  The tool layer wraps those in
    ``ToolError``, which has no status, so without this the adapter would
    report every boundary failure as a generic 500 and the caller could not
    distinguish "you asked for a path that does not exist" from "the Jupyter
    server is broken".
    """
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        status = getattr(cur, "status_code", None)
        if isinstance(status, int):
            return status
        cur = cur.__cause__ or cur.__context__
    return None


def _error_json(request: Request, exc: BaseException, status_code: int) -> JSONResponse:
    """Build the single error envelope every failing endpoint returns.

    The envelope is served with **HTTP 200** and ``ok: false``, with the
    status the failure would otherwise have carried in ``http_status``.

    This is deliberate and specific to this API's consumer.  A GPT Action
    calls ``raise_for_status()`` on the response, so on any non-2xx the
    aiohttp client raises and the caller is shown the exception *type*
    (``ClientResponseError: <class 'aiohttp.client_exceptions...'>``) —
    the response body, and every diagnostic in it, is discarded before the
    model ever sees it.  Returning 200 is what makes the error readable to
    the only thing that reads it.  Clients must branch on ``ok``.
    """
    import traceback as tb_mod

    notebook_id = request.path_params.get("notebook_id")
    notebook_path = None
    if notebook_id:
        try:
            notebook_path = decode_notebook_id(notebook_id)
        except Exception:
            pass

    return JSONResponse(
        status_code=200,
        content=ErrorResponse(
            http_status=status_code,
            error_type=type(exc).__name__,
            error_message=_describe(exc),
            traceback="".join(
                tb_mod.format_exception(type(exc), exc, exc.__traceback__)
            ),
            notebook_id=notebook_id,
            notebook_path=notebook_path,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions and return structured error JSON.

    Without this, FastAPI's default handler returns a bare
    'Internal Server Error' string with zero diagnostic context, which
    leaves the GPT (and the user) unable to understand what went wrong.
    """
    return _error_json(request, exc, _boundary_status(exc) or 500)


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Give deliberate HTTPExceptions the same envelope as unhandled ones.

    FastAPI's default returns ``{"detail": ...}``, so a client would have to
    parse two different error shapes depending on which layer failed.
    """
    return _error_json(request, exc, exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Same envelope for 422s; the pydantic errors go in error_message."""
    return _error_json(request, exc, 422)


@app.on_event("startup")
async def _startup_configure_jupyter() -> None:
    """Configure the fixed Jupyter server when the app starts.

    This is needed when the app is launched via ``uvicorn`` module import
    rather than the ``main()`` entry point (e.g. during development or
    when run by a process manager that invokes uvicorn directly).
    """
    configure_jupyter()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class UseNotebookRequest(BaseModel):
    notebook_path: str = Field(..., description="Jupyter-root-relative path ending in .ipynb")
    mode: Literal["connect", "create"] = Field("connect", description="Open existing or create new")
    kernel_name: str = Field(
        "sagemath",
        description="Kernelspec name (e.g. sagemath, python3, pari_jupyter, gap, singular, lean4, octave, coconut, julia-1.10). Default: sagemath.",
    )


class InsertCellRequest(BaseModel):
    cell_index: int = Field(-1, ge=-1, description="0-based index; -1 means append")
    cell_type: Literal["code", "markdown"] = Field("code")
    cell_source: str = Field("")


class InsertExecuteRequest(BaseModel):
    cell_index: int = Field(-1, ge=-1, description="0-based index; -1 means append")
    cell_source: str = Field(...)
    handoff_after_seconds: int = Field(
        35,
        ge=1,
        le=40,
        description="Seconds to wait before returning status=running; execution continues",
    )


class OverwriteCellSourceRequest(BaseModel):
    cell_source: str = Field(...)


class EditCellSourceRequest(BaseModel):
    old_string: str = Field(...)
    new_string: str = Field(...)
    replace_all: bool = Field(False)


class DeleteCellRequest(BaseModel):
    cell_indices: list[int] = Field(..., description="0-based indices of cells to delete")
    include_source: bool = Field(True)


class MoveCellRequest(BaseModel):
    source_index: int = Field(..., ge=0)
    target_index: int = Field(..., ge=0)


class ExecuteCellRequest(BaseModel):
    handoff_after_seconds: int = Field(
        35,
        ge=1,
        le=40,
        description="Seconds to wait before returning status=running; execution continues",
    )


class ExecuteCodeRequest(BaseModel):
    code: str = Field(...)
    handoff_after_seconds: int = Field(
        35,
        ge=1,
        le=40,
        description="Seconds to wait before returning status=running; execution continues",
    )


class ReadNotebookQuery(BaseModel):
    response_format: Literal["brief", "detailed"] = Field("brief")
    start_index: int = Field(0, ge=0)
    limit: int = Field(20, ge=1, le=200)


class ListFilesQuery(BaseModel):
    path: str = Field("")
    max_depth: int = Field(1, ge=1, le=10)
    start_index: int = Field(0, ge=0)
    limit: int = Field(25, ge=1, le=200)
    pattern: str | None = Field(None, description="Glob pattern, e.g. *.ipynb")


# ---------------------------------------------------------------------------
# Response models — explicit so the OpenAPI schema has `properties`
# (GPT Action validator rejects object schemas without properties).
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    ok: bool
    status: str
    jupyter_url: str | None = None
    error: str | None = None


class UseNotebookResponse(BaseModel):
    ok: bool
    notebook_id: str
    notebook_path: str
    kernel_id: str | None = None
    kernel_name: str | None = None
    session_id: str | None = None
    kernel_reused: bool
    mode: str


class NotebookResultResponse(BaseModel):
    ok: bool
    notebook_id: str
    notebook_path: str
    result: Any = None


class CellResultResponse(BaseModel):
    ok: bool
    notebook_id: str
    notebook_path: str
    cell_index: int
    result: Any = None


class CellOutputsResponse(BaseModel):
    ok: bool
    notebook_id: str
    notebook_path: str
    cell_index: int | None = None
    status: Literal["running", "complete"]
    operation: str
    kernel_id: str | None = None
    poll_after_seconds: int | None = None
    instruction: str | None = None
    outputs: Any = None


class ExecuteOutputsResponse(BaseModel):
    ok: bool
    notebook_id: str
    notebook_path: str
    status: Literal["running", "complete"]
    operation: str
    kernel_id: str | None = None
    poll_after_seconds: int | None = None
    instruction: str | None = None
    outputs: Any = None


class ExecutionStatusResponse(BaseModel):
    ok: bool
    notebook_id: str
    notebook_path: str
    status: Literal["running", "complete"]
    operation: str
    cell_index: int | None = None
    kernel_id: str | None = None
    elapsed_seconds: float | None = None
    poll_after_seconds: int | None = None
    instruction: str | None = None
    outputs: Any = None


class DeleteCellResponse(BaseModel):
    ok: bool
    notebook_id: str
    notebook_path: str
    deleted_indices: list[int]
    result: Any = None


class MoveCellResponse(BaseModel):
    ok: bool
    notebook_id: str
    notebook_path: str
    source_index: int
    target_index: int
    result: Any = None


class GenericResultResponse(BaseModel):
    ok: bool
    result: Any = None


class RestartResponse(BaseModel):
    ok: bool
    notebook_id: str
    notebook_path: str
    result: Any = None


class UnuseNotebookResponse(BaseModel):
    ok: bool
    notebook_id: str
    notebook_path: str
    kernel_id: str | None = None
    result: Any = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(**extra: Any) -> dict[str, Any]:
    """Common ServerContext kwargs shared by all tool calls."""
    base = {
        "mode": runtime.context.mode,
        "server_client": runtime.context.server_client,
        "contents_manager": runtime.context.contents_manager,
        "kernel_manager": runtime.context.kernel_manager,
        "kernel_spec_manager": runtime.context.kernel_spec_manager,
        "notebook_manager": runtime.notebooks,
    }
    base.update(extra)
    return base


def _envelope(notebook_id: str, path: str, **extra: Any) -> dict[str, Any]:
    """Build the standard response envelope."""
    env: dict[str, Any] = {
        "ok": True,
        "notebook_id": notebook_id,
        "notebook_path": path,
    }
    env.update(extra)
    return env


def _jupyter_connection() -> tuple[str, dict[str, str]]:
    """Return the configured Jupyter URL and authentication headers."""
    config = get_config()
    base_url = config.runtime_url or "http://localhost:8888"
    token = config.runtime_token
    headers = {} if not token else {"Authorization": f"token {token}"}
    return base_url.rstrip("/"), headers


async def _ensure_notebook_file(notebook_path: str, *, create: bool) -> None:
    """Ensure a notebook exists before creating its Jupyter session."""
    import httpx

    base_url, headers = _jupyter_connection()
    contents_url = f"{base_url}/api/contents/{notebook_path}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(contents_url, headers=headers)
        if response.status_code == 200:
            return
        if response.status_code != 404:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Could not inspect notebook '{notebook_path}': {response.text[:500]}",
            )
        if not create:
            raise HTTPException(
                status_code=404,
                detail=f"Notebook '{notebook_path}' does not exist on the Jupyter server.",
            )
        created = await client.put(
            contents_url,
            headers=headers,
            json={
                "type": "notebook",
                "format": "json",
                "content": {
                    "cells": [],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                },
            },
        )
    if created.status_code not in (200, 201):
        raise HTTPException(
            status_code=created.status_code,
            detail=f"Failed to create notebook '{notebook_path}': {created.text[:500]}",
        )


async def _get_or_create_session_kernel(
    notebook_path: str,
    kernel_name: str,
) -> tuple[str, str, bool]:
    """Reuse the notebook's Jupyter session kernel or create one bound session."""
    import httpx

    base_url, headers = _jupyter_connection()
    async with httpx.AsyncClient(timeout=30) as client:
        specs_resp = await client.get(f"{base_url}/api/kernelspecs", headers=headers)
        if specs_resp.status_code == 200:
            available = sorted(specs_resp.json().get("kernelspecs", {}))
            if kernel_name not in available:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Kernelspec '{kernel_name}' is not installed on the "
                        f"Jupyter server at {base_url}. "
                        f"Available kernels: {', '.join(available) or '(none)'}."
                    ),
                )

        sessions_resp = await client.get(f"{base_url}/api/sessions", headers=headers)
        sessions_resp.raise_for_status()
        matching = [
            session
            for session in sessions_resp.json()
            if session.get("path") == notebook_path
        ]
        if len(matching) > 1:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Notebook '{notebook_path}' has {len(matching)} Jupyter sessions; "
                    "refusing to choose a kernel ambiguously."
                ),
            )
        if matching:
            session = matching[0]
            kernel = session["kernel"]
            active_name = kernel.get("name")
            if active_name != kernel_name:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Notebook '{notebook_path}' already has a '{active_name}' "
                        f"session kernel ({kernel.get('id')}); requested '{kernel_name}'. "
                        "Select the active kernelspec or explicitly restart the notebook "
                        "with a different kernel."
                    ),
                )
            return kernel["id"], session["id"], True

        resp = await client.post(
            f"{base_url}/api/sessions",
            json={
                "path": notebook_path,
                "name": notebook_path,
                "type": "notebook",
                "kernel": {"name": kernel_name},
            },
            headers=headers,
        )

    if resp.status_code != 201:
        detail = resp.text[:500]
        raise HTTPException(
            status_code=400,
            detail=(
                f"Failed to create a Jupyter session for '{notebook_path}' "
                f"with kernel '{kernel_name}': {resp.status_code} {detail}"
            ),
        )

    session = resp.json()
    return session["kernel"]["id"], session["id"], False


# ---------------------------------------------------------------------------
# Routes — server-level
# ---------------------------------------------------------------------------


@app.get("/health", operation_id="health", response_model=HealthResponse)
async def health() -> dict[str, Any]:
    """Report API and Jupyter server readiness."""
    try:
        # Touch ServerContext to force initialization; if Jupyter is
        # unreachable this raises.
        _ = runtime.context.mode
        return {"ok": True, "status": "healthy", "jupyter_url": get_config().runtime_url}
    except Exception as exc:
        return {"ok": False, "status": "unhealthy", "error": str(exc)}


@app.get("/v1/files", operation_id="list_files", response_model=GenericResultResponse)
async def list_files(
    path: str = "",
    max_depth: int = 1,
    start_index: int = 0,
    limit: int = 25,
    pattern: str | None = None,
) -> Any:
    """List files on the configured Jupyter server."""
    result = await safe_notebook_operation(
        lambda: ListFilesTool().execute(
            **_ctx(),
            path=path,
            max_depth=max_depth,
            start_index=start_index,
            limit=limit,
            pattern=pattern,
        )
    )
    return {"ok": True, "result": result}


@app.get("/v1/kernels", operation_id="list_kernels", response_model=GenericResultResponse)
async def list_kernels() -> Any:
    """List active kernels on the Jupyter server."""
    result = await safe_notebook_operation(
        lambda: ListKernelsTool().execute(
            mode=runtime.context.mode,
            server_client=runtime.context.server_client,
            kernel_manager=runtime.context.kernel_manager,
            kernel_spec_manager=runtime.context.kernel_spec_manager,
        )
    )
    return {"ok": True, "result": result}


@app.get("/v1/notebooks", operation_id="list_notebooks", response_model=GenericResultResponse)
async def list_notebooks(
    path: str = "",
    max_depth: int = 10,
    start_index: int = 0,
    limit: int = 50,
) -> Any:
    """List notebook files on the Jupyter server under the given path.

    Returns every ``*.ipynb`` file found.  Each entry includes the
    notebook path (relative to the Jupyter root), which can be passed
    directly to ``use_notebook`` as ``notebook_path``.
    """
    result = await safe_notebook_operation(
        lambda: ListFilesTool().execute(
            **_ctx(
                path=path,
                max_depth=max_depth,
                start_index=start_index,
                limit=limit,
                pattern="*.ipynb",
            )
        )
    )
    return {"ok": True, "result": result}


# ---------------------------------------------------------------------------
# Routes — notebook use
# ---------------------------------------------------------------------------


@app.post(
    "/v1/notebooks/use",
    operation_id="use_notebook",
    response_model=UseNotebookResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def use_notebook(request: UseNotebookRequest) -> dict[str, Any]:
    """Open or create a notebook and return its deterministic ID.

    A kernel is started with the given ``kernel_name`` (default
    ``sagemath``) and bound to the notebook.  Available kernelspecs can
    be listed via ``list_kernels`` or the Jupyter server's
    ``/api/kernelspecs`` endpoint.
    """
    try:
        notebook_id = encode_notebook_id(request.notebook_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with runtime.lock:
        await _ensure_notebook_file(
            request.notebook_path,
            create=(request.mode == "create"),
        )
        kernel_id, session_id, kernel_reused = await _get_or_create_session_kernel(
            request.notebook_path,
            request.kernel_name,
        )
        try:
            path = await runtime.activate(
                notebook_id,
                create=False,
                kernel_id=kernel_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        actual_kernel_id = runtime.notebooks.get_kernel_id(notebook_id)
        return _envelope(
            notebook_id,
            path,
            kernel_id=actual_kernel_id,
            kernel_name=request.kernel_name,
            session_id=session_id,
            kernel_reused=kernel_reused,
            mode=request.mode,
        )


# ---------------------------------------------------------------------------
# Routes — notebook reads
# ---------------------------------------------------------------------------


@app.get(
    "/v1/notebooks/{notebook_id}",
    operation_id="read_notebook",
    response_model=NotebookResultResponse,
)
async def read_notebook(
    notebook_id: str,
    response_format: Literal["brief", "detailed"] = "brief",
    start_index: int = 0,
    limit: int = 20,
) -> dict[str, Any]:
    """Read notebook contents (paginated)."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: ReadNotebookTool().execute(
                **_ctx(
                    notebook_name=notebook_id,
                    response_format=response_format,
                    start_index=start_index,
                    limit=limit,
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, result=result)


@app.get(
    "/v1/notebooks/{notebook_id}/cells/{cell_index}",
    operation_id="read_cell",
    response_model=CellResultResponse,
)
async def read_cell(
    notebook_id: str,
    cell_index: int,
    include_outputs: bool = True,
) -> dict[str, Any]:
    """Read a single cell by index."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: ReadCellTool().execute(
                **_ctx(
                    cell_index=cell_index,
                    include_outputs=include_outputs,
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, cell_index=cell_index, result=result)


# ---------------------------------------------------------------------------
# Routes — cell mutations
# ---------------------------------------------------------------------------


@app.post(
    "/v1/notebooks/{notebook_id}/cells",
    operation_id="insert_cell",
    response_model=CellResultResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def insert_cell(
    notebook_id: str,
    request: InsertCellRequest,
) -> dict[str, Any]:
    """Insert a cell at the given index."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: InsertCellTool().execute(
                **_ctx(
                    cell_index=request.cell_index,
                    cell_type=request.cell_type,
                    cell_source=request.cell_source,
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, cell_index=request.cell_index, result=result)


@app.put(
    "/v1/notebooks/{notebook_id}/cells/{cell_index}",
    operation_id="overwrite_cell_source",
    response_model=CellResultResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def overwrite_cell_source(
    notebook_id: str,
    cell_index: int,
    request: OverwriteCellSourceRequest,
) -> dict[str, Any]:
    """Overwrite the source of a cell."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: OverwriteCellSourceTool().execute(
                **_ctx(
                    cell_index=cell_index,
                    cell_source=request.cell_source,
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, cell_index=cell_index, result=result)


@app.patch(
    "/v1/notebooks/{notebook_id}/cells/{cell_index}",
    operation_id="edit_cell_source",
    response_model=CellResultResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def edit_cell_source(
    notebook_id: str,
    cell_index: int,
    request: EditCellSourceRequest,
) -> dict[str, Any]:
    """Apply a find-and-replace edit to a cell's source."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: EditCellSourceTool().execute(
                **_ctx(
                    cell_index=cell_index,
                    old_string=request.old_string,
                    new_string=request.new_string,
                    replace_all=request.replace_all,
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, cell_index=cell_index, result=result)


@app.delete(
    "/v1/notebooks/{notebook_id}/cells",
    operation_id="delete_cell",
    response_model=DeleteCellResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def delete_cell(
    notebook_id: str,
    request: DeleteCellRequest,
) -> dict[str, Any]:
    """Delete one or more cells by index."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: DeleteCellTool().execute(
                **_ctx(
                    cell_indices=request.cell_indices,
                    include_source=request.include_source,
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, deleted_indices=request.cell_indices, result=result)


@app.post(
    "/v1/notebooks/{notebook_id}/cells/move",
    operation_id="move_cell",
    response_model=MoveCellResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def move_cell(
    notebook_id: str,
    request: MoveCellRequest,
) -> dict[str, Any]:
    """Move a cell from source_index to target_index."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: MoveCellTool().execute(
                **_ctx(
                    source_index=request.source_index,
                    target_index=request.target_index,
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(
        notebook_id,
        path,
        source_index=request.source_index,
        target_index=request.target_index,
        result=result,
    )


@app.post(
    "/v1/notebooks/{notebook_id}/cells/{cell_index}/clear-output",
    operation_id="clear_cell_output",
    response_model=CellResultResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def clear_cell_output(
    notebook_id: str,
    cell_index: int,
) -> dict[str, Any]:
    """Clear the output of a single cell."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: ClearCellOutputTool().execute(**_ctx(cell_index=cell_index)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, cell_index=cell_index, result=result)


# ---------------------------------------------------------------------------
# Routes — cell execution
# ---------------------------------------------------------------------------


@app.post(
    "/v1/notebooks/{notebook_id}/cells/{cell_index}/execute",
    operation_id="execute_cell",
    response_model=CellOutputsResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def execute_cell(
    notebook_id: str,
    cell_index: int,
    request: ExecuteCellRequest,
) -> dict[str, Any]:
    """Execute a cell by index and return its outputs."""
    try:
        path, state = await runtime.start_execution(
            notebook_id,
            lambda: ExecuteCellTool().execute(
                **_ctx(
                    cell_index=cell_index,
                    timeout_seconds=None,
                    stream=False,
                    progress_interval=0,
                    ensure_kernel_alive_fn=runtime.ensure_kernel_alive,
                )
            ),
            operation_name="execute_cell",
            cell_index=cell_index,
            handoff_after_seconds=request.handoff_after_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, **state)


@app.post(
    "/v1/notebooks/{notebook_id}/cells/insert-and-execute",
    operation_id="insert_execute_code_cell",
    response_model=CellOutputsResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def insert_execute_code_cell(
    notebook_id: str,
    request: InsertExecuteRequest,
) -> dict[str, Any]:
    """Insert a code cell and immediately execute it.

    Mirrors the existing MCP wrapper: InsertCellTool then ExecuteCellTool
    on the same index, with one retry on the execution step.
    """
    async def _insert_and_execute() -> Any:
        await safe_notebook_operation(
            lambda: InsertCellTool().execute(
                **_ctx(
                    cell_index=request.cell_index,
                    cell_type="code",
                    cell_source=request.cell_source,
                )
            )
        )
        return await safe_notebook_operation(
            lambda: ExecuteCellTool().execute(
                **_ctx(
                    cell_index=request.cell_index,
                    timeout_seconds=None,
                    stream=False,
                    progress_interval=0,
                    ensure_kernel_alive_fn=runtime.ensure_kernel_alive,
                )
            ),
            max_retries=1,
        )

    try:
        path, state = await runtime.start_execution(
            notebook_id,
            _insert_and_execute,
            operation_name="insert_execute_code_cell",
            cell_index=request.cell_index,
            handoff_after_seconds=request.handoff_after_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, **state)


@app.post(
    "/v1/notebooks/{notebook_id}/execute-code",
    operation_id="execute_code",
    response_model=ExecuteOutputsResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def execute_code(
    notebook_id: str,
    request: ExecuteCodeRequest,
) -> dict[str, Any]:
    """Execute temporary code in the notebook's kernel without inserting a cell."""
    try:
        path, state = await runtime.start_execution(
            notebook_id,
            lambda: ExecuteCodeTool().execute(
                **_ctx(
                    code=request.code,
                    timeout=None,
                    ensure_kernel_alive_fn=runtime.ensure_kernel_alive,
                    wait_for_kernel_idle_fn=wait_for_kernel_idle,
                    safe_extract_outputs_fn=safe_extract_outputs,
                )
            ),
            operation_name="execute_code",
            cell_index=None,
            handoff_after_seconds=request.handoff_after_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, **state)


@app.get(
    "/v1/notebooks/{notebook_id}/execution",
    operation_id="get_execution_status",
    response_model=ExecutionStatusResponse,
)
async def get_execution_status(notebook_id: str) -> dict[str, Any]:
    """Poll execution state without waiting for the notebook operation lock."""
    try:
        path, state = runtime.execution_status(notebook_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(notebook_id, path, **state)


# ---------------------------------------------------------------------------
# Routes — notebook lifecycle
# ---------------------------------------------------------------------------


@app.post(
    "/v1/notebooks/{notebook_id}/restart",
    operation_id="restart_notebook",
    response_model=RestartResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def restart_notebook(notebook_id: str) -> dict[str, Any]:
    """Restart the notebook's kernel."""
    if runtime.execution_is_running(notebook_id):
        kernel_id = runtime.notebooks.get_kernel_id(notebook_id)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot restart notebook '{notebook_id}' because an execution is still "
                f"running on kernel '{kernel_id}'. Poll "
                f"/v1/notebooks/{notebook_id}/execution until it is complete."
            ),
        )
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: RestartNotebookTool().execute(**_ctx(notebook_name=notebook_id)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, result=result)


@app.post(
    "/v1/notebooks/{notebook_id}/unuse",
    operation_id="unuse_notebook",
    response_model=UnuseNotebookResponse,
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def unuse_notebook(notebook_id: str) -> dict[str, Any]:
    """Disconnect the Assistant API client without shutting down the session kernel."""
    if runtime.execution_is_running(notebook_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot unuse notebook '{notebook_id}' while its execution is running. "
                f"Poll /v1/notebooks/{notebook_id}/execution until it is complete."
            ),
        )
    path = decode_notebook_id(notebook_id)
    kernel_id = runtime.notebooks.get_kernel_id(notebook_id)
    async with runtime.lock:
        result = await safe_notebook_operation(
            lambda: UnuseNotebookTool().execute(
                **_ctx(notebook_name=notebook_id)
            )
        )
    runtime.pending_executions.pop(notebook_id, None)
    return _envelope(
        notebook_id,
        path,
        kernel_id=kernel_id,
        result=result,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure the fixed Jupyter server and run one Uvicorn worker."""
    configure_jupyter()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.getenv("PORT", "4042")),
        workers=1,
    )
