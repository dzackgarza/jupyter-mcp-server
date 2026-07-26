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
cell executions is the purpose of a notebook kernel.  Because
``NotebookManager`` keeps a process-wide current-notebook pointer, all
notebook-specific operations run under one ``asyncio.Lock``.  Run one
Uvicorn worker.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
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
    """Process-wide runtime holding the notebook manager and a serialization lock."""

    def __init__(self) -> None:
        self.notebooks = NotebookManager()
        self.context = ServerContext.get_instance()
        self.lock = asyncio.Lock()
        self.pending_executions: dict[str, PendingExecution] = {}

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
        """Activate ``notebook_id`` under the lock, then run ``operation``."""
        async with self.lock:
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

        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=handoff_after_seconds,
            )
        except asyncio.TimeoutError:
            pass
        return self._execution_payload(notebook_id, pending)
