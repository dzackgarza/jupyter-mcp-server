# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Base classes, enums, and error infrastructure for MCP tools."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from jupyter_kernel_client import KernelClient
from jupyter_server_client import JupyterServerClient


class ServerMode(str, Enum):
    """Enum to indicate which server mode the tool is running in."""

    MCP_SERVER = "mcp_server"
    JUPYTER_SERVER = "jupyter_server"


class ToolError(Exception):
    """Raised by tool implementations to signal a failure to the MCP client.

    FastMCP catches any exception from a tool function and returns it as an
    MCP error response (isError=True).  ToolError messages are written for
    the consuming agent: they explain what was attempted, why it failed, and
    what the agent should try instead.

    Use ``format_tool_error`` to build the message.

    ``status_code`` records the HTTP meaning of the failure when the tool
    knows it — 404 for "that notebook does not exist", for instance.  The
    HTTP adapter reads it so a caller error is not reported as a server
    fault; without it every tool failure looks like a 500.
    """

    def __init__(self, *args: Any, status_code: int | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code


def format_tool_error(
    tool_name: str,
    action: str,
    error: Exception,
    *,
    context: dict[str, str] | None = None,
    suggestions: list[str] | None = None,
) -> str:
    """Format a rich error message for an agent consuming MCP tool output.

    The message is structured so an LLM can parse the failure mode, root
    cause, and recovery path without guessing.

    Args:
        tool_name: Name of the tool that failed (e.g. ``"list_files"``).
        action: Human-readable description of what the tool was trying to do
            (e.g. ``"list directory contents at 'notebooks/'"``).
        error: The underlying exception.
        context: Optional key-value pairs of diagnostic context
            (e.g. ``{"server_root": "/home/user/notebooks"}``).
        suggestions: Optional list of recovery actions the agent should try.

    Returns:
        A multi-line, agent-readable error string.
    """
    # Build the exception identity line
    exc_type = type(error).__name__
    exc_module = type(error).__module__ or ""
    if exc_module and exc_module != "builtins":
        qualified_type = f"{exc_module}.{exc_type}"
    else:
        qualified_type = exc_type

    parts: list[str] = [f"[{tool_name}] {action} failed."]
    parts.append(f"  Error: {qualified_type}: {error}")

    # Pull HTTP context from exception attributes when available
    # (jupyter_server_client.JupyterServerError carries .status_code / .url)
    status_code = getattr(error, "status_code", None)
    url = getattr(error, "url", None)
    if status_code is not None:
        parts.append(f"  HTTP status: {status_code}")
    if url is not None:
        parts.append(f"  URL: {url}")

    # Additional caller-supplied context
    if context:
        for key, value in context.items():
            parts.append(f"  {key}: {value}")

    # Recovery suggestions
    if suggestions:
        parts.append("  Suggestions:")
        for s in suggestions:
            parts.append(f"    - {s}")

    return "\n".join(parts)


class BaseTool(ABC):
    """Abstract base class for all MCP tools.

    Each tool must implement the execute method which handles both
    MCP_SERVER mode (using HTTP clients) and JUPYTER_SERVER mode
    (using direct API access to serverapp managers).
    """

    def __init__(self):
        """Initialize the tool."""
        pass

    @abstractmethod
    async def execute(
        self,
        mode: ServerMode,
        server_client: JupyterServerClient | None = None,
        kernel_client: KernelClient | None = None,
        contents_manager: Any | None = None,
        kernel_manager: Any | None = None,
        kernel_spec_manager: Any | None = None,
        **kwargs,
    ) -> Any:
        """Execute the tool logic.

        Args:
            mode: ServerMode indicating MCP_SERVER or JUPYTER_SERVER
            server_client: JupyterServerClient for HTTP access (MCP_SERVER mode)
            kernel_client: KernelClient for kernel HTTP access (MCP_SERVER mode)
            contents_manager: Direct access to contents manager (JUPYTER_SERVER mode)
            kernel_manager: Direct access to kernel manager (JUPYTER_SERVER mode)
            kernel_spec_manager: Direct access to kernel spec manager (JUPYTER_SERVER mode)
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result (type varies by tool)
        """
        pass
