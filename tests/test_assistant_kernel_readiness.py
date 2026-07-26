# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Real-Jupyter proof that session helpers return only usable kernels."""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest
from jupyter_kernel_client import KernelClient

from jupyter_mcp_server.assistant_api import _get_or_create_session_kernel
from jupyter_mcp_server.config import reset_config, set_config

from .conftest import JUPYTER_TOKEN


@pytest.mark.asyncio
async def test_created_session_kernel_executes_before_helper_returns(
    jupyter_server: str,
) -> None:
    """Readiness is kernel responsiveness, not stale REST execution state."""
    path = f"test-kernel-readiness-{uuid.uuid4().hex}.ipynb"
    headers = {"Authorization": f"token {JUPYTER_TOKEN}"}
    session_id: str | None = None
    set_config(runtime_url=jupyter_server, runtime_token=JUPYTER_TOKEN)

    async with httpx.AsyncClient(timeout=30) as client:
        created = await client.put(
            f"{jupyter_server}/api/contents/{path}",
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
        created.raise_for_status()
        try:
            kernel_id, session_id, _ = await _get_or_create_session_kernel(
                path,
                "python3",
            )
            kernel = KernelClient(
                server_url=jupyter_server,
                token=JUPYTER_TOKEN,
                kernel_id=kernel_id,
            )
            try:
                await asyncio.to_thread(kernel.start)
                reply = await asyncio.to_thread(
                    kernel.execute,
                    "print(2 + 2)",
                    timeout=10,
                )
                assert reply["status"] == "ok"
                assert reply["outputs"] == [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": "4\n",
                    }
                ]
            finally:
                await asyncio.to_thread(kernel.stop, shutdown_kernel=False)
        finally:
            if session_id is not None:
                await client.delete(
                    f"{jupyter_server}/api/sessions/{session_id}",
                    headers=headers,
                )
            await client.delete(
                f"{jupyter_server}/api/contents/{path}",
                headers=headers,
            )
            reset_config()
