# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Real-Jupyter proof that session helpers return only usable kernels."""

from __future__ import annotations

import uuid

import httpx
import pytest
from fastapi import HTTPException

from jupyter_mcp_server.assistant_api import _get_or_create_session_kernel
from jupyter_mcp_server.config import reset_config, set_config

from .conftest import JUPYTER_TOKEN


@pytest.mark.asyncio
async def test_created_session_kernel_is_idle_before_helper_returns(
    jupyter_server: str,
) -> None:
    """The helper must not expose Jupyter's transient ``starting`` kernel."""
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
            try:
                kernel_id, session_id, _ = await _get_or_create_session_kernel(
                    path,
                    "python3",
                )
            except HTTPException as exc:
                assert exc.status_code == 504
                assert "did not become usable" in exc.detail
                sessions = await client.get(
                    f"{jupyter_server}/api/sessions",
                    headers=headers,
                )
                sessions.raise_for_status()
                assert all(session["path"] != path for session in sessions.json())
            else:
                kernel = await client.get(
                    f"{jupyter_server}/api/kernels/{kernel_id}",
                    headers=headers,
                )
                kernel.raise_for_status()
                assert kernel.json()["execution_state"] in {"idle", "busy"}, kernel.json()
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
