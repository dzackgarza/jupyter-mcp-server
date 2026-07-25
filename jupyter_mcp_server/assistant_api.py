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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jupyter_mcp_server.assistant_runtime import (
    AssistantRuntime,
    configure_jupyter,
)
from jupyter_mcp_server.config import get_config
from jupyter_mcp_server.notebook_id import encode_notebook_id
from jupyter_mcp_server.tools import (
    ClearCellOutputTool,
    DeleteCellTool,
    EditCellSourceTool,
    ExecuteCellTool,
    ExecuteCodeTool,
    InsertCellTool,
    ListFilesTool,
    ListKernelsTool,
    ListNotebooksTool,
    MoveCellTool,
    OverwriteCellSourceTool,
    ReadCellTool,
    ReadNotebookTool,
    RestartNotebookTool,
)
from jupyter_mcp_server.utils import safe_notebook_operation

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
)

runtime = AssistantRuntime()

_CONSEQUENTIAL_FALSE: dict[str, Any] = {"x-openai-isConsequential": False}


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


class InsertCellRequest(BaseModel):
    cell_index: int = Field(-1, ge=-1, description="0-based index; -1 means append")
    cell_type: Literal["code", "markdown"] = Field("code")
    cell_source: str = Field("")


class InsertExecuteRequest(BaseModel):
    cell_index: int = Field(-1, ge=-1, description="0-based index; -1 means append")
    cell_source: str = Field(...)
    timeout: int = Field(35, ge=1, le=40, description="Max seconds to wait for execution")


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
    timeout: int = Field(35, ge=1, le=40, description="Max seconds to wait for execution")


class ExecuteCodeRequest(BaseModel):
    code: str = Field(...)
    timeout: int = Field(35, ge=1, le=40)


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


# ---------------------------------------------------------------------------
# Routes — server-level
# ---------------------------------------------------------------------------


@app.get("/health", operation_id="health")
async def health() -> dict[str, Any]:
    """Report API and Jupyter server readiness."""
    try:
        # Touch ServerContext to force initialization; if Jupyter is
        # unreachable this raises.
        _ = runtime.context.mode
        return {"ok": True, "status": "healthy", "jupyter_url": get_config().runtime_url}
    except Exception as exc:
        return {"ok": False, "status": "unhealthy", "error": str(exc)}


@app.get("/v1/files", operation_id="list_files")
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


@app.get("/v1/kernels", operation_id="list_kernels")
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


@app.get("/v1/notebooks/managed", operation_id="list_notebooks")
async def list_notebooks() -> Any:
    """List notebooks currently registered in the in-process NotebookManager.

    Diagnostic only — for notebook discovery on the Jupyter server, call
    ``list_files`` with ``pattern=*.ipynb``.
    """
    result = await safe_notebook_operation(lambda: ListNotebooksTool().execute(**_ctx()))
    return {"ok": True, "result": result}


# ---------------------------------------------------------------------------
# Routes — notebook use
# ---------------------------------------------------------------------------


@app.post(
    "/v1/notebooks/use",
    operation_id="use_notebook",
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def use_notebook(request: UseNotebookRequest) -> dict[str, Any]:
    """Open or create a notebook and return its deterministic ID."""
    try:
        notebook_id = encode_notebook_id(request.notebook_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with runtime.lock:
        try:
            path = await runtime.activate(notebook_id, create=(request.mode == "create"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        kernel_id = runtime.notebooks.get_kernel_id(notebook_id)
        return _envelope(
            notebook_id,
            path,
            kernel_id=kernel_id,
            mode=request.mode,
        )


# ---------------------------------------------------------------------------
# Routes — notebook reads
# ---------------------------------------------------------------------------


@app.get(
    "/v1/notebooks/{notebook_id}",
    operation_id="read_notebook",
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
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def execute_cell(
    notebook_id: str,
    cell_index: int,
    request: ExecuteCellRequest,
) -> dict[str, Any]:
    """Execute a cell by index and return its outputs."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: ExecuteCellTool().execute(
                **_ctx(
                    cell_index=cell_index,
                    timeout_seconds=request.timeout,
                    stream=False,
                    progress_interval=0,
                    ensure_kernel_alive_fn=runtime.ensure_kernel_alive,
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, cell_index=cell_index, outputs=result)


@app.post(
    "/v1/notebooks/{notebook_id}/cells/insert-and-execute",
    operation_id="insert_execute_code_cell",
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
    try:
        async with runtime.lock:
            path = await runtime.activate(notebook_id)

            await safe_notebook_operation(
                lambda: InsertCellTool().execute(
                    **_ctx(
                        cell_index=request.cell_index,
                        cell_type="code",
                        cell_source=request.cell_source,
                    )
                )
            )

            result = await safe_notebook_operation(
                lambda: ExecuteCellTool().execute(
                    **_ctx(
                        cell_index=request.cell_index,
                        timeout_seconds=request.timeout,
                        stream=False,
                        progress_interval=0,
                        ensure_kernel_alive_fn=runtime.ensure_kernel_alive,
                    )
                ),
                max_retries=1,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(
        notebook_id,
        path,
        cell_index=request.cell_index,
        outputs=result,
    )


@app.post(
    "/v1/notebooks/{notebook_id}/execute-code",
    operation_id="execute_code",
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def execute_code(
    notebook_id: str,
    request: ExecuteCodeRequest,
) -> dict[str, Any]:
    """Execute temporary code in the notebook's kernel without inserting a cell."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: ExecuteCodeTool().execute(
                **_ctx(
                    code=request.code,
                    timeout=request.timeout,
                    ensure_kernel_alive_fn=runtime.ensure_kernel_alive,
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, outputs=result)


# ---------------------------------------------------------------------------
# Routes — notebook lifecycle
# ---------------------------------------------------------------------------


@app.post(
    "/v1/notebooks/{notebook_id}/restart",
    operation_id="restart_notebook",
    openapi_extra=_CONSEQUENTIAL_FALSE,
)
async def restart_notebook(notebook_id: str) -> dict[str, Any]:
    """Restart the notebook's kernel."""
    try:
        path, result = await runtime.run(
            notebook_id,
            lambda: RestartNotebookTool().execute(**_ctx(notebook_name=notebook_id)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(notebook_id, path, result=result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure the fixed Jupyter server and run one Uvicorn worker."""
    configure_jupyter()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "4041")),
        workers=1,
    )
