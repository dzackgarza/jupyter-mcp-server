# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Use notebook tool implementation."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal

from jupyter_core.utils import ensure_async
from jupyter_kernel_client import KernelClient
from jupyter_server_client import JupyterServerClient, NotFoundError

from jupyter_mcp_server.models import Notebook
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.tools._base import BaseTool, ServerMode, ToolError, format_tool_error

logger = logging.getLogger(__name__)

# Maximum seconds to wait for a kernel WebSocket handshake before returning a
# structured error.  Must be well under Cloudflare's 100 s proxy timeout so
# the application can emit a meaningful response instead of a bare 524.
_KERNEL_START_TIMEOUT_SECONDS = 30


class UseNotebookTool(BaseTool):
    """Tool to use (connect to or create) a notebook file."""

    async def _start_kernel_local(self, kernel_manager: Any, path: str | None = None):
        # Start a new kernel using local API
        kernel_id = await kernel_manager.start_kernel()
        logger.info(f"Started kernel '{kernel_id}', waiting for it to be ready...")

        # CRITICAL: Wait for the kernel to actually start and be ready
        # The start_kernel() call returns immediately, but kernel takes time to start
        import asyncio

        max_wait_time = 30  # seconds
        wait_interval = 0.5  # seconds
        elapsed = 0
        kernel_ready = False

        while elapsed < max_wait_time:
            try:
                # Get kernel model to check its state
                kernel_model = kernel_manager.get_kernel(kernel_id)
                if kernel_model is not None:
                    # Kernel exists, check if it's ready
                    # In Jupyter, we can try to get connection info which indicates readiness
                    try:
                        kernel_manager.get_connection_info(kernel_id)
                        kernel_ready = True
                        logger.info(f"Kernel '{kernel_id}' is ready (took {elapsed:.1f}s)")
                        break
                    except:
                        # Connection info not available yet, kernel still starting
                        pass
            except Exception as e:
                logger.debug(f"Waiting for kernel to start: {e}")

            await asyncio.sleep(wait_interval)
            elapsed += wait_interval

        if not kernel_ready:
            logger.warning(
                f"Kernel '{kernel_id}' may not be fully ready after {max_wait_time}s wait"
            )

        return {"id": kernel_id}

    def _notebook_file_exists(
        self,
        mode: ServerMode,
        server_client: JupyterServerClient | None,
        contents_manager: Any | None,
        notebook_path: str,
    ) -> bool:
        """Return whether ``notebook_path`` is present on the Jupyter server.

        Listing the parent directory is used rather than fetching the file so
        this stays cheap and matches how ``_check_path_*`` already decides
        existence.
        """
        path = Path(notebook_path)
        parent = path.parent.as_posix() if path.parent.as_posix() != "." else ""

        if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
            return bool(contents_manager.exists(notebook_path))

        listing = server_client.contents.list_directory(parent)
        return path.name in [entry.name for entry in listing]

    async def _check_path_http(
        self, server_client: JupyterServerClient, notebook_path: str, mode: str
    ) -> None:
        """Verify notebook path exists (HTTP mode). Raises ToolError on failure."""
        path = Path(notebook_path)
        try:
            parent_path = path.parent.as_posix() if path.parent.as_posix() != "." else ""

            if parent_path:
                dir_contents = server_client.contents.list_directory(parent_path)
            else:
                dir_contents = server_client.contents.list_directory("")

            if mode == "connect":
                available_names = [file.name for file in dir_contents]
                file_exists = path.name in available_names
                if not file_exists:
                    # Build a helpful list of .ipynb files the agent can choose from
                    notebooks_in_dir = [n for n in available_names if n.endswith(".ipynb")]
                    ctx = {
                        "requested_path": notebook_path,
                        "parent_directory": parent_path or "root",
                    }
                    if notebooks_in_dir:
                        ctx["notebooks_in_directory"] = ", ".join(notebooks_in_dir)
                    else:
                        ctx["notebooks_in_directory"] = "(none)"
                    raise ToolError(
                        f"[use_notebook] Notebook '{notebook_path}' not found on the Jupyter server.\n"
                        + "\n".join(f"  {k}: {v}" for k, v in ctx.items())
                        + "\n  Suggestions:\n"
                        f"    - Check the filename for typos (names are case-sensitive).\n"
                        f"    - Use list_files(path='{parent_path or ''}') to browse available files.\n"
                        f"    - Use use_notebook(mode='create') to create a new notebook at this path.",
                        status_code=404,
                    )

        except ToolError:
            raise  # Already enriched
        except NotFoundError as e:
            parent_dir = (
                path.parent.as_posix() if path.parent.as_posix() != "." else "root directory"
            )
            raise ToolError(
                format_tool_error(
                    "use_notebook",
                    f"verify parent directory '{parent_dir}' exists",
                    e,
                    context={"notebook_path": notebook_path, "parent_directory": parent_dir},
                    suggestions=[
                        f"The directory '{parent_dir}' does not exist on the server.",
                        "Use list_files(path='') to see the server root directory.",
                        "Create the directory first, or use a different path.",
                    ],
                )
            ) from e
        except Exception as e:
            raise ToolError(
                format_tool_error(
                    "use_notebook",
                    f"check path '{notebook_path}' on Jupyter server",
                    e,
                    suggestions=[
                        "Check that the Jupyter server is running and accessible.",
                        "Verify the server token is correct.",
                    ],
                )
            ) from e

    async def _check_path_local(
        self, contents_manager: Any, notebook_path: str, mode: str
    ) -> None:
        """Verify notebook path exists (local mode). Raises ToolError on failure."""
        path = Path(notebook_path)
        try:
            parent_path = str(path.parent) if str(path.parent) != "." else ""

            # Get directory contents using local API
            model = await ensure_async(
                contents_manager.get(parent_path, content=True, type="directory")
            )

            if mode == "connect":
                available_names = [item["name"] for item in model.get("content", [])]
                file_exists = path.name in available_names
                if not file_exists:
                    notebooks_in_dir = [n for n in available_names if n.endswith(".ipynb")]
                    ctx = {
                        "requested_path": notebook_path,
                        "parent_directory": parent_path or "root",
                    }
                    if notebooks_in_dir:
                        ctx["notebooks_in_directory"] = ", ".join(notebooks_in_dir)
                    else:
                        ctx["notebooks_in_directory"] = "(none)"
                    raise ToolError(
                        f"[use_notebook] Notebook '{notebook_path}' not found on the Jupyter server.\n"
                        + "\n".join(f"  {k}: {v}" for k, v in ctx.items())
                        + "\n  Suggestions:\n"
                        f"    - Check the filename for typos (names are case-sensitive).\n"
                        f"    - Use list_files(path='{parent_path or ''}') to browse available files.\n"
                        f"    - Use use_notebook(mode='create') to create a new notebook at this path.",
                        status_code=404,
                    )

        except ToolError:
            raise  # Already enriched
        except Exception as e:
            parent_dir = str(path.parent) if str(path.parent) != "." else "root directory"
            raise ToolError(
                format_tool_error(
                    "use_notebook",
                    f"check path '{notebook_path}' in local contents manager",
                    e,
                    context={"parent_directory": parent_dir},
                    suggestions=[
                        f"The directory '{parent_dir}' may not exist.",
                        "Use list_files(path='') to see the server root directory.",
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
        session_manager: Any | None = None,
        notebook_manager: NotebookManager | None = None,
        # Tool-specific parameters
        notebook_name: str = None,
        notebook_path: str = None,
        use_mode: Literal["connect", "create"] = "connect",
        kernel_id: str | None = None,
        runtime_url: str | None = None,
        runtime_token: str | None = None,
        **kwargs,
    ) -> str:
        """Execute the use_notebook tool.

        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            server_client: HTTP client for MCP_SERVER mode
            contents_manager: Direct API access for JUPYTER_SERVER mode
            kernel_manager: Direct kernel manager for JUPYTER_SERVER mode
            session_manager: Session manager for creating kernel-notebook associations
            notebook_manager: Notebook manager instance
            notebook_name: Unique identifier for the notebook
            notebook_path: Path to the notebook file (optional, if not provided switches to existing notebook)
            use_mode: "connect" or "create"
            kernel_id: Optional specific kernel ID
            runtime_url: Runtime URL for HTTP mode
            runtime_token: Runtime token for HTTP mode
            **kwargs: Additional parameters

        Returns:
            Success message with notebook information
        """
        # Check server connectivity (HTTP mode only)
        if mode == ServerMode.MCP_SERVER and server_client is not None:
            try:
                server_client.get_status()
            except Exception as e:
                raise ToolError(
                    format_tool_error(
                        "use_notebook",
                        "connect to Jupyter server",
                        e,
                        context={"server_url": getattr(server_client, "base_url", "unknown")},
                        suggestions=[
                            "Check that the Jupyter server is running.",
                            "Verify the server URL and token are correct.",
                            "Use connect_to_jupyter to re-establish the connection.",
                        ],
                    )
                ) from e

        # Check the path exists (raises ToolError on failure)
        if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
            await self._check_path_local(contents_manager, notebook_path, use_mode)
        elif mode == ServerMode.MCP_SERVER and server_client is not None:
            await self._check_path_http(server_client, notebook_path, use_mode)
        else:
            raise ToolError(
                f"[use_notebook] Invalid server mode or missing required clients.\n"
                f"  mode={mode}, server_client={'provided' if server_client else 'None'}, "
                f"contents_manager={'provided' if contents_manager else 'None'}\n"
                f"  Suggestions:\n"
                f"    - Use connect_to_jupyter to connect to a Jupyter server first.\n"
                f"    - Check the server configuration."
            )

        info_list = []

        # A manager entry only means "we opened this once"; the file can have
        # been deleted since (by the user in JupyterLab, or by a test's
        # cleanup).  Decide "already created" from the file, not from memory:
        # otherwise create mode returns success having created nothing, and
        # every later read fails against a file that is not there.
        if (
            use_mode == "create"
            and notebook_name in notebook_manager
            and notebook_manager.get_notebook_path(notebook_name) == notebook_path
            and not self._notebook_file_exists(
                mode, server_client, contents_manager, notebook_path
            )
        ):
            info_list.append(
                f"[INFO] Recreating notebook '{notebook_name}': its file is no "
                f"longer present at '{notebook_path}'."
            )
            notebook_manager.remove_notebook(notebook_name)

        # Check if notebook already in notebook_manager (Cober all cases)
        if notebook_name in notebook_manager:
            if use_mode == "create":
                if notebook_manager.get_notebook_path(notebook_name) == notebook_path:
                    return f"Notebook '{notebook_name}'(path: {notebook_path}) is already created. DO NOT CREATE AGAIN."
                else:
                    return f"Notebook '{notebook_name}' is already used. Use different notebook_name to create a new notebook on {notebook_path}."
            else:
                if notebook_manager.get_notebook_path(notebook_name) == notebook_path:
                    if notebook_name == notebook_manager.get_current_notebook():
                        return f"Notebook '{notebook_name}' is already activated now. DO NOT REACTIVATE AGAIN."
                    else:
                        # the only correct case.
                        info_list.append(
                            f"[INFO] Reactivating notebook '{notebook_name}' and deactivating '{notebook_manager.get_current_notebook()}'."
                        )
                        notebook_manager.set_current_notebook(notebook_name)
                else:
                    return f"The path '{notebook_path}' is not the correct path for notebook '{notebook_name}'. Do you mean connect to '{notebook_manager.get_notebook_path(notebook_name)}'?"
        # add new notebook to notebook_manager
        else:
            # Create notebook if needed
            # This runs before the kernel and the Jupyter session below, which are
            # both pointed at notebook_path and so need the file to exist already.
            if use_mode == "create":
                content = {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [
                                "New Notebook Created by Jupyter MCP Server",
                            ],
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 4,
                }
                if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
                    # Use local API to create notebook
                    await ensure_async(
                        contents_manager.new(
                            model={"type": "notebook", "content": content, "format": "json"},
                            path=notebook_path,
                        )
                    )
                elif mode == ServerMode.MCP_SERVER and server_client is not None:
                    server_client.contents.create_notebook(notebook_path, content=content)

            # # Create/connect to kernel based on mode
            if mode == ServerMode.MCP_SERVER and server_client is not None:
                if kernel_id is not None:
                    kernels = server_client.kernels.list_kernels()
                    kernel_exists = any(kernel.id == kernel_id for kernel in kernels)
                    if not kernel_exists:
                        raise ToolError(
                        f"[use_notebook] Kernel '{kernel_id}' not found on the Jupyter server.\n"
                        f"  Suggestions:\n"
                        f"    - Use list_kernels to see available kernel IDs.\n"
                        f"    - Omit kernel_id to start a new kernel automatically."
                    )
                kernel = KernelClient(
                    server_url=runtime_url,
                    token=runtime_token,
                    kernel_id=kernel_id,
                    client_kwargs={"reconnect_interval": 1.0},
                )
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(kernel.start, path=notebook_path),
                        timeout=_KERNEL_START_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError as exc:
                    raise ToolError(
                        f"[use_notebook] Kernel start timed out after "
                        f"{_KERNEL_START_TIMEOUT_SECONDS}s for '{notebook_path}'.\n"
                        f"  runtime_url: {runtime_url}\n"
                        f"  kernel_id: {kernel_id or '(new — SageMath startup too slow)'}\n"
                        f"  The kernel WebSocket handshake did not complete in time.\n"
                        f"  Suggestions:\n"
                        f"    - Call list_kernels and pass an already-running kernel's "
                        f"id as kernel_id to skip startup entirely.\n"
                        f"    - Retry; the server may be transiently overloaded.\n"
                        f"    - Check Jupyter server health if retries keep failing.",
                        status_code=504,
                    ) from exc

                info_list.append(f"[INFO] Connected to kernel '{kernel.id}'.")
            elif mode == ServerMode.JUPYTER_SERVER and kernel_manager is not None:
                # JUPYTER_SERVER mode: Use local kernel manager API directly
                if kernel_id:
                    # Connect to existing kernel - verify it exists
                    if kernel_id not in kernel_manager:
                        raise ToolError(
                        f"[use_notebook] Kernel '{kernel_id}' not found in local kernel manager.\n"
                        f"  Suggestions:\n"
                        f"    - Use list_kernels to see available kernel IDs.\n"
                        f"    - Omit kernel_id to start a new kernel automatically."
                    )
                    kernel = {"id": kernel_id}
                else:
                    kernel = await self._start_kernel_local(kernel_manager, path=notebook_path)
                    kernel_id = kernel["id"]

                info_list.append(f"[INFO] Connected to kernel '{kernel_id}'.")
                # Create a Jupyter session to associate the kernel with the notebook
                # This is CRITICAL for JupyterLab to recognize the kernel-notebook connection
                if session_manager is not None:
                    try:
                        # create_session is an async method, so we await it directly
                        session_dict = await session_manager.create_session(
                            path=notebook_path,
                            kernel_id=kernel_id,
                            type="notebook",
                            name=notebook_path,
                        )
                        logger.info(
                            f"Created Jupyter session '{session_dict.get('id')}' for notebook '{notebook_path}' with kernel '{kernel_id}'"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create Jupyter session: {e}. Notebook may not be properly connected in JupyterLab UI."
                        )
                else:
                    logger.warning(
                        "No session_manager available. Notebook may not be properly connected in JupyterLab UI."
                    )

            # Add notebook to notebook_manager
            if mode == ServerMode.MCP_SERVER and runtime_url:
                notebook_manager.add_notebook(
                    notebook_name,
                    kernel,
                    server_url=runtime_url,
                    token=runtime_token,
                    path=notebook_path,
                )
            elif mode == ServerMode.JUPYTER_SERVER and kernel_manager is not None:
                notebook_manager.add_notebook(
                    notebook_name, kernel, server_url="local", token=None, path=notebook_path
                )
            else:
                raise ToolError(
                    f"[use_notebook] Cannot register notebook: invalid configuration.\n"
                    f"  mode={mode}, runtime_url={runtime_url}, kernel_manager={'provided' if kernel_manager else 'None'}\n"
                    f"  Suggestions:\n"
                    f"    - In MCP_SERVER mode, a runtime_url is required.\n"
                    f"    - In JUPYTER_SERVER mode, a kernel_manager must be available.\n"
                    f"    - Use connect_to_jupyter to configure the connection."
                )

            notebook_manager.set_current_notebook(notebook_name)
            info_list.append(f"[INFO] Successfully activate notebook '{notebook_name}'.")

        # Return the quick overview of currently activated notebook
        try:
            if mode == ServerMode.JUPYTER_SERVER and contents_manager is not None:
                # Read notebook to get cell count and first 20 cells
                model = await ensure_async(
                    contents_manager.get(notebook_path, content=True, type="notebook")
                )
                if "content" in model:
                    notebook = Notebook(**model["content"])
                else:
                    notebook = Notebook()

            elif mode == ServerMode.MCP_SERVER and notebook_manager is not None:
                # Use notebook manager to get cell info
                async with notebook_manager.get_current_connection() as notebook_content:
                    notebook = Notebook(**notebook_content.as_dict())

            info_list.append(f"\nNotebook has {len(notebook)} cells.")
            info_list.append(f"Showing first {min(20, len(notebook))} cells:\n")
            info_list.append(
                notebook.format_output(response_format="brief", start_index=0, limit=20)
            )
        except Exception as e:
            logger.debug(f"Failed to get notebook summary: {e}")

        # Check if we should open in JupyterLab UI (when JupyterLab mode is
        # enabled and the user opted in, since opening activates the tab)
        try:
            from jupyter_mcp_server.config import get_config
            from jupyter_mcp_server.jupyter_extension.context import get_server_context

            context = get_server_context()

            if context.is_jupyterlab_mode() and get_config().open_notebook_in_ui:
                logger.info(
                    f"JupyterLab mode enabled, attempting to open notebook '{notebook_path}' in JupyterLab UI"
                )

                # Determine base_url and token based on mode
                base_url = None
                token = None

                if mode == ServerMode.JUPYTER_SERVER and context.serverapp is not None:
                    # JUPYTER_SERVER mode: Use ServerApp connection details
                    base_url = context.serverapp.connection_url
                    token = context.serverapp.token
                elif mode == ServerMode.MCP_SERVER and runtime_url:
                    # MCP_SERVER mode: Use runtime_url and runtime_token
                    base_url = runtime_url
                    token = runtime_token

                if base_url and token:
                    try:
                        from jupyter_mcp_tools.client import MCPToolsClient

                        async with MCPToolsClient(base_url=base_url, token=token) as client:
                            execution_result = await client.execute_tool(
                                tool_id="docmanager_open",  # docmanager:open converted to underscore format
                                parameters={"path": notebook_path},
                            )

                            if execution_result.get("success"):
                                logger.info(
                                    f"Successfully opened notebook '{notebook_path}' in JupyterLab UI"
                                )
                            else:
                                logger.warning(
                                    f"Failed to open notebook in JupyterLab UI: {execution_result}"
                                )

                    except ImportError:
                        logger.warning(
                            "jupyter_mcp_tools not available, skipping JupyterLab UI opening"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to open notebook in JupyterLab UI: {e}")
                else:
                    logger.warning(
                        "No valid base_url or token available for opening notebook in JupyterLab UI"
                    )
        except Exception as e:
            logger.debug(f"Could not check JupyterLab mode: {e}")

        return "\n".join(info_list)
