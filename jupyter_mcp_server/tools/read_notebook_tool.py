# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""List cells tool implementation."""

from pathlib import Path
from typing import Any, Literal

from jupyter_core.utils import ensure_async
from jupyter_server_client import JupyterServerClient

from jupyter_mcp_server.models import Notebook
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.tools._base import BaseTool, ServerMode, ToolError, format_tool_error
from jupyter_mcp_server.utils import get_notebook_model


class ReadNotebookTool(BaseTool):
    """Tool to read a notebook and return index, source content, type, execution count of each cell."""

    async def execute(
        self,
        mode: ServerMode,
        server_client: JupyterServerClient | None = None,
        contents_manager: Any | None = None,
        notebook_manager: NotebookManager | None = None,
        notebook_name: str = None,
        response_format: Literal["brief", "detailed"] = "brief",
        start_index: int = 0,
        limit: int = 20,
        **kwargs,
    ) -> str:
        """Execute the read_notebook tool.

        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            contents_manager: Direct API access for JUPYTER_SERVER mode
            notebook_manager: Notebook manager instance
            notebook_name: Notebook identifier to read
            response_format: Response format (brief or detailed)
            start_index: Starting index for pagination (0-based)
            limit: Maximum number of items to return (0 means no limit)
            **kwargs: Additional parameters

        Returns:
            Formatted table with cell information
        """
        if notebook_name not in notebook_manager:
            connected = list(notebook_manager.list_all_notebooks().keys())
            raise ToolError(
                f"[read_notebook] Notebook '{notebook_name}' is not connected.\n"
                f"  Connected notebooks: {connected}\n"
                f"  Suggestions:\n"
                f"    - Use list_notebooks to see available notebooks.\n"
                f"    - Use use_notebook to connect to a notebook."
            )

        if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
            # Local mode: try the live YDoc first (collaborative session), same
            # as every cell-mutation tool, so a read right after our own write
            # sees it instead of the on-disk copy the autosave hasn't flushed yet.
            notebook_path = notebook_manager.get_notebook_path(notebook_name)

            from jupyter_mcp_server.jupyter_extension.context import get_server_context

            context = get_server_context()
            serverapp = context.serverapp
            ydoc_path = notebook_path
            if serverapp and not Path(ydoc_path).is_absolute():
                ydoc_path = str(Path(serverapp.root_dir) / ydoc_path)

            nb_model = await get_notebook_model(serverapp, ydoc_path) if serverapp else None
            if nb_model:
                notebook = Notebook(**nb_model.as_dict())
            else:
                model = await ensure_async(
                    contents_manager.get(notebook_path, content=True, type="notebook")
                )
                if "content" not in model:
                    raise ToolError(f"[read_notebook] Could not read notebook content from {notebook_path}")
                notebook = Notebook(**model["content"])
        elif mode == ServerMode.MCP_SERVER and notebook_manager is not None:
            # Remote mode: use WebSocket connection to Y.js document
            async with notebook_manager.get_notebook_connection(notebook_name) as notebook_content:
                notebook = Notebook(**notebook_content.as_dict())
        else:
            raise ToolError(f"[read_notebook] Invalid mode or missing required clients: mode={mode}")

        if start_index >= len(notebook):
            raise ToolError(
                f"[read_notebook] Start index {start_index} is out of range.\n"
                f"  Notebook has {len(notebook)} cells (valid indices: 0 to {len(notebook) - 1}).\n"
                f"  Suggestions:\n"
                f"    - Use a smaller start_index."
            )

        info_list = [f"Notebook {notebook_name} has {len(notebook)} cells.\n"]
        info_list.append(
            notebook.format_output(
                response_format=response_format, start_index=start_index, limit=limit
            )
        )

        return "\n".join(info_list)
