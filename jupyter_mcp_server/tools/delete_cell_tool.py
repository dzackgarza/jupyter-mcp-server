# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Delete cell tool implementation."""

from pathlib import Path
from typing import Any

import nbformat
from jupyter_server_client import JupyterServerClient

from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.tools._base import BaseTool, ServerMode, ToolError, format_tool_error
from jupyter_mcp_server.utils import (
    clean_notebook_outputs,
    get_current_notebook_context,
    get_notebook_model,
)


class DeleteCellTool(BaseTool):
    """Tool to delete specific cells from a notebook."""

    def _get_cell_source(self, cell: Any) -> str:
        """Get the cell source from the cell"""
        cell_source = cell.get("source", "")
        if isinstance(cell_source, list):
            return "".join(cell_source)
        else:
            return str(cell_source)

    def _validate_indices(self, cell_indices: list[int], total_cells: int) -> None:
        """Validate that every index is a valid 0-based cell position.

        Args:
            cell_indices: Indices of cells to delete (0-based)
            total_cells: Total number of cells in the notebook

        Raises:
            ToolError: When any index is negative or >= total_cells.
        """
        for cell_index in cell_indices:
            if cell_index < 0 or cell_index >= total_cells:
                raise ToolError(
                    f"[delete_cell] Cell index {cell_index} is out of range.\n"
                    f"  Notebook has {total_cells} cells (valid indices: 0 to {total_cells - 1}).\n"
                    f"  Requested indices: {cell_indices}\n"
                    f"  Suggestions:\n"
                    f"    - Use read_notebook to see all cells and their indices.\n"
                    f"    - Cell indices are 0-based."
                )

    async def _delete_cell_ydoc(
        self, serverapp: Any, notebook_path: str, cell_indices: list[int]
    ) -> list:
        """Delete cell using YDoc (collaborative editing mode).

        Args:
            serverapp: Jupyter ServerApp instance
            notebook_path: Path to the notebook
            cell_indices: List of indices of cells to delete

        Returns:
            NotebookNode
        """
        try:
            nb = await get_notebook_model(serverapp, notebook_path)
        except Exception as e:
            raise ToolError(
                format_tool_error(
                    "delete_cell",
                    f"read notebook via YDoc at '{notebook_path}'",
                    e,
                    context={"notebook_path": notebook_path},
                    suggestions=[
                        "Check that the notebook is open and the collaborative editing session is active.",
                        "Try reconnecting with use_notebook(mode='connect').",
                    ],
                )
            ) from e

        if nb:
            self._validate_indices(cell_indices, len(nb))
            cells = nb.delete_many_cells(cell_indices)
            return cells
        else:
            # YDoc not available, use file operations
            return await self._delete_cell_file(notebook_path, cell_indices)

    async def _delete_cell_file(self, notebook_path: str, cell_indices: list[int]) -> list:
        """Delete cell using file operations (non-collaborative mode).

        Args:
            notebook_path: Absolute path to the notebook
            cell_indices: List of indices of cells to delete

        Returns:
            List of deleted cells
        """
        path = Path(notebook_path)
        if not path.exists():
            raise ToolError(
                f"[delete_cell] Notebook file not found at '{notebook_path}'.\n"
                f"  The file does not exist on disk.\n"
                f"  Suggestions:\n"
                f"    - Use list_files to find the correct notebook path.\n"
                f"    - Use use_notebook(mode='connect') to connect to an existing notebook.\n"
                f"    - The notebook may have been renamed, moved, or deleted."
            )

        try:
            # Read notebook file as version 4 for consistency
            with open(notebook_path, encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)
        except nbformat.reader.NotJSONError as e:
            raise ToolError(
                format_tool_error(
                    "delete_cell",
                    f"read notebook file '{notebook_path}'",
                    e,
                    suggestions=[
                        "The file exists but is not valid JSON.",
                        "Check that the file is a valid .ipynb notebook.",
                    ],
                )
            ) from e
        except Exception as e:
            raise ToolError(
                format_tool_error(
                    "delete_cell",
                    f"read notebook file '{notebook_path}'",
                    e,
                    context={"file_exists": str(path.exists()), "file_size": str(path.stat().st_size) if path.exists() else "N/A"},
                )
            ) from e

        clean_notebook_outputs(notebook)

        self._validate_indices(cell_indices, len(notebook.cells))

        deleted_cells = []
        for cell_index in cell_indices:
            cell = notebook.cells[cell_index]
            result = {
                "index": cell_index,
                "cell_type": cell.cell_type,
                "source": self._get_cell_source(cell),
            }
            deleted_cells.append(result)

        # Delete the cell
        for cell_index in sorted(cell_indices, reverse=True):
            notebook.cells.pop(cell_index)

        try:
            # Write back to file
            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)
        except Exception as e:
            raise ToolError(
                format_tool_error(
                    "delete_cell",
                    f"write updated notebook back to '{notebook_path}'",
                    e,
                    context={
                        "cells_deleted": str(cell_indices),
                        "remaining_cells": str(len(notebook.cells)),
                    },
                    suggestions=[
                        "The cells were removed in memory but the file could not be saved.",
                        "Check file permissions and disk space.",
                    ],
                )
            ) from e

        return deleted_cells

    async def _delete_cell_websocket(
        self, notebook_manager: NotebookManager, cell_indices: list[int]
    ) -> list:
        """Delete cell using WebSocket connection (MCP_SERVER mode).

        Args:
            notebook_manager: Notebook manager instance
            cell_indices: List of indices of cells to delete

        Returns:
            List of deleted cell information
        """
        try:
            async with notebook_manager.get_current_connection() as notebook:
                self._validate_indices(cell_indices, len(notebook))
                cells = notebook.delete_many_cells(cell_indices)
                return cells
        except ToolError:
            raise  # Already enriched
        except Exception as e:
            current_nb = notebook_manager.get_current_notebook() or "unknown"
            current_path = notebook_manager.get_current_notebook_path() or "unknown"
            raise ToolError(
                format_tool_error(
                    "delete_cell",
                    f"delete cells {cell_indices} via WebSocket connection",
                    e,
                    context={
                        "notebook_name": current_nb,
                        "notebook_path": current_path,
                    },
                    suggestions=[
                        "The WebSocket connection to the notebook may have been lost.",
                        "Try reconnecting with use_notebook(mode='connect').",
                        "Check that the Jupyter server is still running.",
                    ],
                )
            ) from e

    async def execute(
        self,
        mode: ServerMode,
        server_client: JupyterServerClient | None = None,
        kernel_client: Any | None = None,
        contents_manager: Any | None = None,
        kernel_manager: Any | None = None,
        kernel_spec_manager: Any | None = None,
        notebook_manager: NotebookManager | None = None,
        # Tool-specific parameters
        cell_indices: list[int] = None,
        include_source: bool = True,
        **kwargs,
    ) -> str:
        """Execute the delete_cell tool.

        This tool supports three modes of operation:

        1. JUPYTER_SERVER mode with YDoc (collaborative):
           - Checks if notebook is open in a collaborative session
           - Uses YDoc for real-time collaborative editing
           - Changes are immediately visible to all connected users

        2. JUPYTER_SERVER mode without YDoc (file-based):
           - Falls back to direct file operations using nbformat
           - Suitable when notebook is not actively being edited

        3. MCP_SERVER mode (WebSocket):
           - Uses WebSocket connection to remote Jupyter server
           - Accesses YDoc through NbModelClient

        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            server_client: HTTP client for MCP_SERVER mode
            contents_manager: Direct API access for JUPYTER_SERVER mode
            notebook_manager: Notebook manager instance
            cell_indices: Indices of cells to delete (0-based)
            include_source: Whether to include source of deleted cells
            **kwargs: Additional parameters

        Returns:
            Success message with deleted cell info

        Raises:
            ToolError: If no notebook is active, indices are invalid, or deletion fails.
        """
        if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
            # JUPYTER_SERVER mode: Try YDoc first, fall back to file operations
            from jupyter_mcp_server.jupyter_extension.context import get_server_context

            context = get_server_context()
            serverapp = context.serverapp

            try:
                notebook_path, _ = get_current_notebook_context(notebook_manager)
            except Exception as e:
                raise ToolError(
                    f"[delete_cell] No notebook is currently active.\n"
                    f"  Error getting notebook context: {e}\n"
                    f"  Suggestions:\n"
                    f"    - Use use_notebook(notebook_name='...', notebook_path='...', mode='connect') to activate a notebook first.\n"
                    f"    - Use list_notebooks to see available notebooks."
                ) from e

            if not notebook_path:
                raise ToolError(
                    f"[delete_cell] No notebook is currently active.\n"
                    f"  Suggestions:\n"
                    f"    - Use use_notebook(notebook_name='...', notebook_path='...', mode='connect') to activate a notebook first.\n"
                    f"    - Use list_notebooks to see available notebooks."
                )

            # Resolve to absolute path
            if serverapp and not Path(notebook_path).is_absolute():
                root_dir = serverapp.root_dir
                notebook_path = str(Path(root_dir) / notebook_path)

            if serverapp:
                # Try YDoc approach first
                cells = await self._delete_cell_ydoc(serverapp, notebook_path, cell_indices)
            else:
                # Fall back to file operations
                cells = await self._delete_cell_file(notebook_path, cell_indices)

        elif mode == ServerMode.MCP_SERVER and notebook_manager is not None:
            # MCP_SERVER mode: Use WebSocket connection
            cells = await self._delete_cell_websocket(notebook_manager, cell_indices)
        else:
            raise ToolError(
                f"[delete_cell] Cannot delete cells: no notebook is active.\n"
                f"  mode={mode}, notebook_manager={'provided' if notebook_manager else 'None'}\n"
                f"  Suggestions:\n"
                f"    - Use use_notebook(notebook_name='...', notebook_path='...', mode='connect') to activate a notebook first.\n"
                f"    - Use list_notebooks to see available notebooks."
            )

        info_list = []
        for cell_index, cell_info in zip(cell_indices, cells, strict=False):
            info_list.append(f"Cell {cell_index} ({cell_info['cell_type']}) deleted successfully.")
            if include_source:
                info_list.append(f"deleted cell source:\n{cell_info['source']}")
                info_list.append("\n---\n")

        return "\n".join(info_list)
