# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Regression proof for RTC notebook execution ownership."""

import asyncio
import gc
import uuid

import pytest
import requests
from jupyter_kernel_client import KernelClient

from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.tools._base import ServerMode
from jupyter_mcp_server.tools.execute_cell_tool import ExecuteCellTool

from .conftest import JUPYTER_TOKEN


@pytest.mark.asyncio
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
async def test_timed_out_rtc_execution_keeps_pycrdt_on_its_owning_thread(jupyter_server):
    """A real timed-out kernel execution must not destroy YDoc state in its worker."""
    notebook_path = f"thread-affinity-{uuid.uuid4().hex}.ipynb"
    contents_url = f"{jupyter_server}/api/contents/{notebook_path}"
    headers = {"Authorization": f"token {JUPYTER_TOKEN}"}
    response = requests.put(
        contents_url,
        headers=headers,
        json={
            "type": "notebook",
            "format": "json",
            "content": {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "print('thread-safe output')",
                        "metadata": {},
                        "outputs": [],
                        "execution_count": None,
                        "id": "output-cell",
                    },
                    {
                        "cell_type": "code",
                        "source": "import time; time.sleep(3); print('finished')",
                        "metadata": {},
                        "outputs": [],
                        "execution_count": None,
                        "id": "blocking-cell",
                    }
                ],
            },
        },
        timeout=10,
    )
    response.raise_for_status()

    kernel = KernelClient(server_url=jupyter_server, token=JUPYTER_TOKEN)
    kernel.start()
    notebooks = NotebookManager()
    notebooks.add_notebook(
        "thread-affinity",
        kernel,
        server_url=jupyter_server,
        token=JUPYTER_TOKEN,
        path=notebook_path,
    )

    try:
        outputs = await ExecuteCellTool().execute(
            mode=ServerMode.MCP_SERVER,
            notebook_manager=notebooks,
            cell_index=0,
            timeout_seconds=10,
            ensure_kernel_alive_fn=lambda: kernel,
        )
        assert "thread-safe output" in "\n".join(str(output) for output in outputs)

        await ExecuteCellTool().execute(
            mode=ServerMode.MCP_SERVER,
            notebook_manager=notebooks,
            cell_index=1,
            timeout_seconds=1,
            ensure_kernel_alive_fn=lambda: kernel,
        )
        gc.collect()
    finally:
        kernel.stop()
        await asyncio.sleep(1)
        requests.delete(contents_url, headers=headers, timeout=10).raise_for_status()
