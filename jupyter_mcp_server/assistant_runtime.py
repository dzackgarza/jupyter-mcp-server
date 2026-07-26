# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Assistant runtime: thin orchestrator over existing tool classes.

This module deliberately does **not** import ``jupyter_mcp_server.server``
because importing that module constructs the FastMCP server and registers
the MCP wrappers.  Instead it imports the existing tool classes directly
and drives them through their ``execute`` methods.

The HTTP contract is stateless — every request names the notebook — but
the Jupyter kernel remains stateful because maintaining variables between
cell executions is the purpose of a notebook kernel. Notebook operations
are serialized per notebook so a slow kernel cannot block unrelated work.
Run one Uvicorn worker.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from jupyter_mcp_server.config import get_config, set_config
from jupyter_mcp_server.log import logger
from jupyter_mcp_server.notebook_id import decode_notebook_id
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.server_context import ServerContext
from jupyter_mcp_server.tools import UseNotebookTool
from jupyter_mcp_server.utils import (
    create_kernel,
    ensure_kernel_alive,
    safe_notebook_operation,
)

__all__ = ["AssistantRuntime", "configure_jupyter"]

_request_id: ContextVar[str] = ContextVar("assistant_request_id", default="-")
diagnostic_logger = logging.getLogger("uvicorn.error")


def bind_request_id(request_id: str) -> Token[str]:
    """Bind one HTTP request ID to logs emitted below the route layer."""
    return _request_id.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    """Restore the request context after an HTTP response."""
    _request_id.reset(token)


def current_request_id() -> str:
    """Return the active HTTP request ID for structured diagnostics."""
    return _request_id.get()


@dataclass
class PendingExecution:
    """One notebook execution that may outlive its initiating HTTP request."""

    task: asyncio.Task[tuple[str, Any]]
    operation: str
    cell_index: int | None
    started_at: float


def configure_jupyter() -> None:
    """Configure the fixed Jupyter server from environment variables.

    Reduced form of the CLI's environment-resolution logic: there is
    exactly one Jupyter server, no dynamic registration, no per-user
    selection.
    """
    url = os.getenv("JUPYTER_URL", "http://localhost:8888")
    token = os.getenv("JUPYTER_TOKEN")

    set_config(
        provider="jupyter",
        runtime_url=url,
        runtime_token=token,
        document_url=url,
        document_token=token,
        start_new_runtime=False,
        execution_timeout=int(os.getenv("JUPYTER_MCP_EXECUTION_TIMEOUT", "35")),
        max_execution_timeout=int(os.getenv("JUPYTER_MCP_MAX_EXECUTION_TIMEOUT", "40")),
        open_notebook_in_ui=False,
    )
    ServerContext.reset()


class AssistantRuntime:
    """Process-wide runtime holding the notebook manager and notebook locks."""

    def __init__(self) -> None:
        self.notebooks = NotebookManager()
        self.context = ServerContext.get_instance()
        self._notebook_locks: dict[str, asyncio.Lock] = {}
        self.pending_executions: dict[str, PendingExecution] = {}

    def lock_for(self, notebook_id: str) -> asyncio.Lock:
        """Return the serialization lock owned by one notebook."""
        lock = self._notebook_locks.get(notebook_id)
        if lock is None:
            lock = asyncio.Lock()
            self._notebook_locks[notebook_id] = lock
        return lock

    @asynccontextmanager
    async def operation_lock(self, notebook_id: str):
        """Acquire one notebook lock with correlated wait/hold diagnostics."""
        lock = self.lock_for(notebook_id)
        kernel_id = self.notebooks.get_kernel_id(notebook_id)
        wait_started = time.monotonic()
        diagnostic_logger.info(
            "assistant_lock_wait request_id=%s notebook_id=%s kernel_id=%s locked=%s",
            current_request_id(),
            notebook_id,
            kernel_id,
            lock.locked(),
        )
        await lock.acquire()
        acquired_at = time.monotonic()
        diagnostic_logger.info(
            "assistant_lock_acquired request_id=%s notebook_id=%s kernel_id=%s "
            "wait_seconds=%.3f",
            current_request_id(),
            notebook_id,
            kernel_id,
            acquired_at - wait_started,
        )
        try:
            yield
        finally:
            lock.release()
            diagnostic_logger.info(
                "assistant_lock_released request_id=%s notebook_id=%s kernel_id=%s "
                "hold_seconds=%.3f",
                current_request_id(),
                notebook_id,
                self.notebooks.get_kernel_id(notebook_id),
                time.monotonic() - acquired_at,
            )

    # ------------------------------------------------------------------
    # Kernel health
    # ------------------------------------------------------------------

    def ensure_kernel_alive(self) -> Any:
        """Ensure the current notebook's kernel is alive, restarting if needed."""
        config = get_config()
        current = self.notebooks.get_current_notebook() or "default"

        def _create() -> Any:
            return create_kernel(config, logger)

        return ensure_kernel_alive(self.notebooks, current, _create)

    # ------------------------------------------------------------------
    # Notebook activation
    # ------------------------------------------------------------------

    async def activate(
        self,
        notebook_id: str,
        *,
        create: bool = False,
        kernel_id: str | None = None,
    ) -> str:
        """Make ``notebook_id`` the current notebook.

        If the notebook is already managed, just set it current.
        Otherwise invoke ``UseNotebookTool.execute`` in connect or create
        mode.  If ``kernel_id`` is provided, ``UseNotebookTool`` connects
        to that pre-started kernel instead of starting a new one.

        Raises
        ------
        ValueError
            If activation fails (the notebook does not exist in connect
            mode, or creation fails).
        """
        path = decode_notebook_id(notebook_id)

        # Create mode must reach UseNotebookTool even when the id is already
        # known: only the tool checks whether the file is still on disk, and
        # short-circuiting here reports a successful create for a notebook
        # that may have been deleted underneath us.
        if notebook_id in self.notebooks and not create:
            self.notebooks.set_current_notebook(notebook_id)
            return path

        config = get_config()

        kwargs: dict[str, Any] = {
            "mode": self.context.mode,
            "server_client": self.context.server_client,
            "contents_manager": self.context.contents_manager,
            "kernel_manager": self.context.kernel_manager,
            "kernel_spec_manager": self.context.kernel_spec_manager,
            "notebook_manager": self.notebooks,
            "notebook_name": notebook_id,
            "notebook_path": path,
            "use_mode": "create" if create else "connect",
            "runtime_url": (config.runtime_url if config.runtime_url != "local" else None),
            "runtime_token": config.runtime_token,
        }
        if kernel_id is not None:
            kwargs["kernel_id"] = kernel_id

        result = await safe_notebook_operation(lambda: UseNotebookTool().execute(**kwargs))

        # Existing tools generally report domain errors as strings.
        # Manager membership is the reliable success test.
        if notebook_id not in self.notebooks:
            raise ValueError(str(result))

        return path

    # ------------------------------------------------------------------
    # Run an operation against an activated notebook
    # ------------------------------------------------------------------

    async def run(
        self,
        notebook_id: str,
        operation: Callable[[], Awaitable[Any]],
        *,
        create: bool = False,
    ) -> tuple[str, Any]:
        """Activate ``notebook_id`` under its lock, then run ``operation``."""
        async with self.operation_lock(notebook_id):
            path = await self.activate(notebook_id, create=create)
            result = await safe_notebook_operation(operation)
            return path, result

    # ------------------------------------------------------------------
    # Long-running execution handoff
    # ------------------------------------------------------------------

    def _execution_payload(
        self,
        notebook_id: str,
        pending: PendingExecution,
    ) -> tuple[str, dict[str, Any]]:
        path = decode_notebook_id(notebook_id)
        base: dict[str, Any] = {
            "operation": pending.operation,
            "cell_index": pending.cell_index,
            "kernel_id": self.notebooks.get_kernel_id(notebook_id),
        }
        if not pending.task.done():
            return path, {
                **base,
                "status": "running",
                "elapsed_seconds": round(time.monotonic() - pending.started_at, 1),
                "poll_after_seconds": 5,
                "instruction": (
                    f"Execution is still running. Check "
                    f"/v1/notebooks/{notebook_id}/execution again in about 5 seconds."
                ),
            }

        completed_path, result = pending.task.result()
        return completed_path, {
            **base,
            "status": "complete",
            "outputs": result,
        }

    def execution_status(self, notebook_id: str) -> tuple[str, dict[str, Any]]:
        """Return tracked execution state without waiting for the runtime lock."""
        pending = self.pending_executions.get(notebook_id)
        if pending is None:
            raise ValueError(f"Notebook '{notebook_id}' has no tracked execution.")
        return self._execution_payload(notebook_id, pending)

    def execution_is_running(self, notebook_id: str) -> bool:
        pending = self.pending_executions.get(notebook_id)
        return pending is not None and not pending.task.done()

    async def probe_kernel(self, notebook_id: str) -> tuple[bool, str]:
        """Probe kernel-info on the control channel, bypassing shell execution."""
        kernel = self.notebooks.get_kernel(notebook_id)
        kernel_id = self.notebooks.get_kernel_id(notebook_id)
        manager = getattr(kernel, "_manager", None)
        client = getattr(manager, "client", None)
        control = getattr(client, "control_channel", None)
        session = getattr(client, "session", None)
        if control is None or session is None:
            raise ValueError(
                f"Notebook '{notebook_id}' has no control channel "
                f"(kernel_id={kernel_id!r})."
            )

        def _request_kernel_info() -> bool:
            request = session.msg("kernel_info_request")
            request_id = request["header"]["msg_id"]
            control.send(request)
            deadline = time.monotonic() + 4
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                reply = control.get_msg(timeout=remaining)
                if reply.get("parent_header", {}).get("msg_id") == request_id:
                    return reply.get("msg_type") == "kernel_info_reply"

        try:
            responsive = await asyncio.to_thread(_request_kernel_info)
        except Exception:
            responsive = False
        diagnostic_logger.info(
            "assistant_kernel_probe request_id=%s notebook_id=%s kernel_id=%s "
            "responsive=%s probe=control_kernel_info",
            current_request_id(),
            notebook_id,
            kernel_id,
            responsive,
        )
        if responsive:
            return True, "Kernel control channel answered kernel_info_request."
        return False, (
            "Kernel heartbeat/control channel did not answer kernel_info_request "
            "within 4 seconds."
        )

    async def inspect_execution(
        self,
        notebook_id: str,
    ) -> tuple[str, dict[str, Any]]:
        """Return task state enriched with kernel responsiveness evidence."""
        path, state = self.execution_status(notebook_id)
        if state["status"] != "running":
            return path, state

        responsive, detail = await self.probe_kernel(notebook_id)
        state["kernel_responsive"] = responsive
        if not responsive:
            state.update(
                status="unresponsive",
                error_code="kernel_unresponsive",
                error_message=detail,
                instruction=(
                    f"POST /v1/notebooks/{notebook_id}/restart to terminate the "
                    "unresponsive session and attach a fresh kernel."
                ),
            )
        return path, state

    def abandon_execution(self, notebook_id: str) -> None:
        """Detach a nonresponsive task and rotate only its notebook lock."""
        pending = self.pending_executions.pop(notebook_id, None)
        if pending is not None:
            pending.task.cancel()

            def _consume_result(task: asyncio.Task[tuple[str, Any]]) -> None:
                try:
                    task.exception()
                except asyncio.CancelledError:
                    logger.info(
                        "Detached execution for notebook %s was cancelled.",
                        notebook_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Detached execution for notebook %s ended with %s: %s",
                        notebook_id,
                        type(exc).__name__,
                        exc,
                    )

            pending.task.add_done_callback(_consume_result)
        self._notebook_locks[notebook_id] = asyncio.Lock()

    async def start_execution(
        self,
        notebook_id: str,
        operation: Callable[[], Awaitable[Any]],
        *,
        operation_name: str,
        cell_index: int | None,
        handoff_after_seconds: int,
    ) -> tuple[str, dict[str, Any]]:
        """Start one serialized execution and hand it off at the HTTP deadline."""
        pending = self.pending_executions.get(notebook_id)
        if pending is not None and not pending.task.done():
            return self._execution_payload(notebook_id, pending)

        task = asyncio.create_task(
            self.run(notebook_id, operation),
            name=f"assistant-{operation_name}-{notebook_id}",
        )
        pending = PendingExecution(
            task=task,
            operation=operation_name,
            cell_index=cell_index,
            started_at=time.monotonic(),
        )
        self.pending_executions[notebook_id] = pending
        diagnostic_logger.info(
            "assistant_execution_started request_id=%s notebook_id=%s kernel_id=%s "
            "operation=%s handoff_seconds=%s",
            current_request_id(),
            notebook_id,
            self.notebooks.get_kernel_id(notebook_id),
            operation_name,
            handoff_after_seconds,
        )

        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=handoff_after_seconds,
            )
        except asyncio.TimeoutError:
            pass
        return self._execution_payload(notebook_id, pending)
