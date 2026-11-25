# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Real-Jupyter proof that session helpers return only usable kernels."""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest
from jupyter_kernel_client import KernelClient

from jupyter_mcp_server.assistant_api import _get_or_create_session_kernel
from jupyter_mcp_server.config import reset_config, set_config

from .conftest import JUPYTER_TOKEN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kernel_name", "code", "expected"),
    [
        ("python3", "print(2 + 2)", "4\n"),
        pytest.param(
            "sagemath",
            "from sage.all import QQ; print(QQ(1) / QQ(3))",
            "1/3\n",
            marks=pytest.mark.skipif(
                os.environ.get("JUPYTER_MCP_TEST_SAGE") != "1",
                reason="activate the Sage development kernelspec first",
            ),
        ),
    ],
    ids=["python3", "sagemath"],
)
async def test_helper_returns_kernel_that_executes_immediately(
    jupyter_server: str,
    kernel_name: str,
    code: str,
    expected: str,
) -> None:
    """Readiness is kernel responsiveness, not stale REST execution state."""
    path = f"test-kernel-readiness-{uuid.uuid4().hex}.ipynb"
    headers = {"Authorization": f"token {JUPYTER_TOKEN}"}
    session_id: str | None = None

    async with httpx.AsyncClient(timeout=30) as client:
        if kernel_name == "sagemath":
            specs = await client.get(
                f"{jupyter_server}/api/kernelspecs", headers=headers
            )
            specs.raise_for_status()
            argv = specs.json()["kernelspecs"][kernel_name]["spec"]["argv"]
            assert os.path.isabs(argv[0]), argv
            assert argv[1:] == [
                "-m",
                "sage.repl.ipython_kernel",
                "-f",
                "{connection_file}",
            ]
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
        set_config(runtime_url=jupyter_server, runtime_token=JUPYTER_TOKEN)
        try:
            kernel_id, session_id, _ = await _get_or_create_session_kernel(
                path,
                kernel_name,
            )
            session = await client.get(
                f"{jupyter_server}/api/sessions/{session_id}",
                headers=headers,
            )
            session.raise_for_status()
            assert session.json()["kernel"]["name"] == kernel_name
            kernel = KernelClient(
                server_url=jupyter_server,
                token=JUPYTER_TOKEN,
                kernel_id=kernel_id,
            )
            try:
                await asyncio.to_thread(kernel.start)
                reply = await asyncio.to_thread(
                    kernel.execute,
                    code,
                    timeout=30 if kernel_name == "sagemath" else 10,
                )
                assert reply["status"] == "ok"
                assert reply["outputs"] == [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": expected,
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
