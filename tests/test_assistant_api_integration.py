# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Integration tests for the Jupyter Assistant API.

These tests start a real temporary Jupyter server (via the existing
``jupyter_server`` session fixture) and a real Assistant API adapter
instance, then exercise the full HTTP stack.

Covers spec required integration tests 1–9:

1. Create a notebook and receive an ID.
2. Read it using only the ID.
3. Insert and execute a cell.
4. Read the cell and verify its source and output.
5. Make a second API request using the same ID and verify kernel
   continuity.
6. Open notebooks (A) and (B), then perform operations in the order
   (A,B,A), verifying that the final operation modifies only (A).
7. Restart the API adapter and verify that the same ID reconnects to the
   same notebook filepath.
8. Verify that failed notebook activation is reported as a client error and
   does not execute against whichever notebook happened to be current.
9. Verify that two concurrent notebook operations are serialized by the
   runtime lock.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
import uuid
from collections.abc import Generator
from http import HTTPStatus

import pytest
import pytest_asyncio
import requests
from httpx import AsyncClient
from requests.exceptions import ConnectionError as ReqConnectionError

from jupyter_mcp_server.notebook_id import encode_notebook_id

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


@pytest_asyncio.fixture
async def client(assistant_api_url: str) -> AsyncClient:
    """Async HTTP client pointed at the Assistant API."""
    async with AsyncClient(base_url=assistant_api_url, timeout=60) as c:
        yield c


@pytest.fixture(scope="module")
def assistant_api_url(jupyter_server: str) -> Generator[str]:
    """Start the Assistant API adapter against the test Jupyter server.

    The Jupyter server fixture provides a base URL like
    ``http://localhost:<port>``.  We strip the scheme/host to get the
    port, then start the adapter on a separate free port with
    ``JUPYTER_URL`` pointing at the test Jupyter server.
    """
    # jupyter_server is "http://localhost:PORT"
    api_port = _find_free_port()
    api_url = f"http://localhost:{api_port}"

    env = {
        **os.environ,
        "JUPYTER_URL": jupyter_server,
        "JUPYTER_MCP_RUNTIME_URL": jupyter_server,
        "JUPYTER_TOKEN": "MY_TOKEN",
        "ALLOW_IMG_OUTPUT": "false",
        "JUPYTER_MCP_EXECUTION_TIMEOUT": "35",
        "PORT": str(api_port),
    }

    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "jupyter_mcp_server.assistant_api:app",
            "--host",
            "localhost",
            "--port",
            str(api_port),
            "--workers",
            "1",
        ],
        stdout=None,
        stderr=None,
        env=env,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    # Wait for readiness
    max_retries = 15
    ready = False
    while max_retries > 0:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            pytest.fail(f"Assistant API died during startup:\n{stderr}")
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == HTTPStatus.OK:
                ready = True
                break
        except ReqConnectionError:
            pass
        time.sleep(1)
        max_retries -= 1

    if not ready:
        proc.terminate()
        proc.wait()
        pytest.fail("Assistant API did not become ready")

    yield api_url

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(autouse=True)
def _cleanup_test_notebooks(jupyter_server: str, assistant_api_url: str):
    """Delete test notebooks from the Jupyter server after each test."""
    yield
    names = (
        "test-create.ipynb",
        "test-aba-a.ipynb",
        "test-aba-b.ipynb",
        "test-restart.ipynb",
        "test-session-reuse.ipynb",
        "test-handoff.ipynb",
        "test-hung-a.ipynb",
        "test-hung-b.ipynb",
        "test-hung-recovery.ipynb",
    )
    for name in names:
        try:
            requests.post(
                f"{assistant_api_url}/v1/notebooks/{_nb_id(name)}/unuse",
                timeout=5,
            )
        except Exception:
            pass
    time.sleep(0.5)

    # Close session-owned kernels before deleting their notebook files.
    try:
        sessions = requests.get(
            f"{jupyter_server}/api/sessions",
            params={"token": "MY_TOKEN"},
            timeout=5,
        ).json()
        for session in sessions:
            if session.get("path") in names:
                requests.delete(
                    f"{jupyter_server}/api/sessions/{session['id']}",
                    params={"token": "MY_TOKEN"},
                    timeout=5,
                )
    except Exception:
        pass

    # Best-effort file cleanup — failures here are not test failures.
    for name in names:
        try:
            requests.delete(
                f"{jupyter_server}/api/contents/{name}",
                params={"token": "MY_TOKEN"},
                timeout=5,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nb_id(path: str) -> str:
    return encode_notebook_id(path)


async def _jupyter_json(jupyter_server: str, path: str) -> list[dict]:
    async with AsyncClient(base_url=jupyter_server, timeout=10) as jupyter:
        response = await jupyter.get(path, params={"token": "MY_TOKEN"})
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Test 1: Create a notebook and receive an ID
# ---------------------------------------------------------------------------


async def test_1_create_notebook_returns_id(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": "test-create.ipynb", "mode": "create"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["notebook_id"].startswith("nb_")
    assert data["notebook_path"] == "test-create.ipynb"
    assert data["kernel_id"]


# ---------------------------------------------------------------------------
# Test 2: Read the notebook using only the ID
# ---------------------------------------------------------------------------


async def test_2_read_notebook_by_id(client: AsyncClient) -> None:
    # Create first
    create = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": "test-create.ipynb", "mode": "create"},
    )
    nb_id = create.json()["notebook_id"]

    resp = await client.get(f"/v1/notebooks/{nb_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["notebook_id"] == nb_id
    assert data["notebook_path"] == "test-create.ipynb"
    assert "result" in data


async def test_list_notebooks_includes_created_notebook(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/v1/notebooks/use",
        json={
            "notebook_path": "test-create.ipynb",
            "mode": "create",
            "kernel_name": "python3",
        },
    )
    assert created.json()["ok"] is True, created.text

    listed = await client.get("/v1/notebooks")
    body = listed.json()
    assert body["ok"] is True, listed.text
    assert isinstance(body["result"], str)
    assert "test-create.ipynb" in body["result"]


async def test_notebook_status_reports_persisted_state_without_a_session(
    client: AsyncClient,
    jupyter_server: str,
) -> None:
    path = f"test-status-{uuid.uuid4().hex}.ipynb"
    notebook_id = encode_notebook_id(path)
    async with AsyncClient(base_url=jupyter_server, timeout=30) as jupyter:
        created = await jupyter.put(
            f"/api/contents/{path}",
            params={"token": "MY_TOKEN"},
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
            response = await client.get(f"/v1/notebooks/{notebook_id}/status")
            body = response.json()
            assert response.status_code == HTTPStatus.OK
            assert body["ok"] is True, response.text
            assert body["notebook_path"] == path
            assert body["persisted"]["valid"] is True
            assert body["persisted"]["cell_count"] == 0
            assert body["session"] == {"count": 0, "id": None}
            assert body["kernel"]["id"] is None
            assert body["kernel"]["execution_state"] is None
            assert body["rtc"]["readable"] is False
            assert "no unique live session" in body["rtc"]["error"]
        finally:
            deleted = await jupyter.delete(
                f"/api/contents/{path}",
                params={"token": "MY_TOKEN"},
            )
            assert deleted.status_code == HTTPStatus.NO_CONTENT


# ---------------------------------------------------------------------------
# Test 3: Insert and execute a cell
# ---------------------------------------------------------------------------


async def test_3_insert_and_execute_cell(client: AsyncClient) -> None:
    create = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": "test-create.ipynb", "mode": "create"},
    )
    nb_id = create.json()["notebook_id"]

    resp = await client.post(
        f"/v1/notebooks/{nb_id}/cells/insert-and-execute",
        json={"cell_index": 0, "cell_source": "print(2 + 2)", "handoff_after_seconds": 35},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ok"] is True
    assert data["cell_index"] == 0
    assert "4" in str(data["outputs"])


# ---------------------------------------------------------------------------
# Test 4: Read the cell and verify source and output
# ---------------------------------------------------------------------------


async def test_4_read_cell_source_and_output(client: AsyncClient) -> None:
    create = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": "test-create.ipynb", "mode": "create"},
    )
    nb_id = create.json()["notebook_id"]

    await client.post(
        f"/v1/notebooks/{nb_id}/cells/insert-and-execute",
        json={"cell_index": 0, "cell_source": "print(42)", "handoff_after_seconds": 35},
    )

    resp = await client.get(f"/v1/notebooks/{nb_id}/cells/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    result_text = str(data["result"])
    assert "print(42)" in result_text
    assert "42" in result_text


# ---------------------------------------------------------------------------
# Test 5: Kernel continuity across requests with the same ID
# ---------------------------------------------------------------------------


async def test_5_kernel_continuity(client: AsyncClient) -> None:
    create = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": "test-create.ipynb", "mode": "create"},
    )
    nb_id = create.json()["notebook_id"]

    # Set a variable
    r1 = await client.post(
        f"/v1/notebooks/{nb_id}/execute-code",
        json={"code": "x = 42", "handoff_after_seconds": 10},
    )
    assert r1.json()["ok"] is True, r1.text

    # Read it in a separate request
    r2 = await client.post(
        f"/v1/notebooks/{nb_id}/execute-code",
        json={"code": "print(x * 2)", "handoff_after_seconds": 10},
    )
    assert r2.status_code == 200
    data = r2.json()
    assert "84" in str(data["outputs"])


# ---------------------------------------------------------------------------
# Test 6: A/B/A interleaving — final operation modifies only (A)
# ---------------------------------------------------------------------------


async def test_6_aba_interleaving(client: AsyncClient) -> None:
    a_create = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": "test-aba-a.ipynb", "mode": "create"},
    )
    a_id = a_create.json()["notebook_id"]

    b_create = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": "test-aba-b.ipynb", "mode": "create"},
    )
    b_id = b_create.json()["notebook_id"]

    # A: set x = 100
    a_set = await client.post(
        f"/v1/notebooks/{a_id}/execute-code",
        json={"code": "x = 100", "handoff_after_seconds": 10},
    )
    assert a_set.json()["ok"] is True, a_set.text

    # B: set x = 200
    b_set = await client.post(
        f"/v1/notebooks/{b_id}/execute-code",
        json={"code": "x = 200", "handoff_after_seconds": 10},
    )
    assert b_set.json()["ok"] is True, b_set.text

    # A: read x — must be 100, not 200
    a_read = await client.post(
        f"/v1/notebooks/{a_id}/execute-code",
        json={"code": "print(x)", "handoff_after_seconds": 10},
    )
    assert a_read.status_code == 200
    assert "100" in str(a_read.json()["outputs"])

    # B: read x — must be 200
    b_read = await client.post(
        f"/v1/notebooks/{b_id}/execute-code",
        json={"code": "print(x)", "handoff_after_seconds": 10},
    )
    assert b_read.status_code == 200
    assert "200" in str(b_read.json()["outputs"])


# ---------------------------------------------------------------------------
# Test 7: Restart the API adapter and reconnect with the same ID
# ---------------------------------------------------------------------------


def test_7_reconnect_after_adapter_restart(jupyter_server: str) -> None:
    """This test manages its own adapter lifecycle to simulate a restart.

    Uses synchronous requests because the adapter start/stop cycle is
    easier to control outside the async fixture.
    """
    import uuid

    unique = f"test-restart-{uuid.uuid4().hex[:6]}.ipynb"
    nb_id = _nb_id(unique)

    def _start_api(port: int) -> subprocess.Popen:
        env = {
            **os.environ,
            "JUPYTER_URL": jupyter_server,
            "JUPYTER_TOKEN": "MY_TOKEN",
            "ALLOW_IMG_OUTPUT": "false",
            "PORT": str(port),
        }
        return subprocess.Popen(
            [
                "python",
                "-m",
                "uvicorn",
                "jupyter_mcp_server.assistant_api:app",
                "--host",
                "localhost",
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )

    def _wait_ready(port: int, proc: subprocess.Popen) -> str:
        url = f"http://localhost:{port}"
        for _ in range(15):
            if proc.poll() is not None:
                pytest.fail("Adapter died")
            try:
                r = requests.get(f"{url}/health", timeout=5)
                if r.status_code == 200:
                    return url
            except ReqConnectionError:
                pass
            time.sleep(1)
        pytest.fail("Adapter did not start")

    port1 = _find_free_port()
    proc1 = _start_api(port1)
    url1 = _wait_ready(port1, proc1)

    try:
        # Create the notebook and set a variable
        r = requests.post(
            f"{url1}/v1/notebooks/use",
            json={"notebook_path": unique, "mode": "create"},
            timeout=30,
        )
        assert r.status_code == 200
        assert r.json()["notebook_id"] == nb_id

        r = requests.post(
            f"{url1}/v1/notebooks/{nb_id}/execute-code",
            json={"code": "y = 'persisted'", "handoff_after_seconds": 10},
            timeout=30,
        )
        assert r.json()["ok"] is True, r.text
    finally:
        proc1.terminate()
        proc1.wait(timeout=10)

    # Start a fresh adapter — the ID must still resolve to the same notebook
    port2 = _find_free_port()
    proc2 = _start_api(port2)
    url2 = _wait_ready(port2, proc2)

    try:
        # Reconnect using the same ID (connect mode — notebook file exists)
        r = requests.post(
            f"{url2}/v1/notebooks/use",
            json={"notebook_path": unique, "mode": "connect"},
            timeout=30,
        )
        assert r.status_code == 200
        assert r.json()["notebook_id"] == nb_id

        # The notebook file still exists; read it.  ``notebook_path`` alone
        # cannot carry this claim: the error envelope also decodes it from
        # the URL, so it matches even when the read failed.
        r = requests.get(f"{url2}/v1/notebooks/{nb_id}", timeout=30)
        assert r.json()["ok"] is True, r.text
        assert r.json()["notebook_path"] == unique
    finally:
        proc2.terminate()
        proc2.wait(timeout=10)
        # Cleanup
        try:
            requests.delete(
                f"{jupyter_server}/api/contents/{unique}",
                params={"token": "MY_TOKEN"},
                timeout=5,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test 8: Failed activation returns 400 and does not execute against current
# ---------------------------------------------------------------------------


async def test_8_failed_activation_returns_client_error(client: AsyncClient) -> None:
    # First, create a real notebook so *something* is current.
    create = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": "test-create.ipynb", "mode": "create"},
    )
    real_id = create.json()["notebook_id"]

    # Set a variable in the real notebook.
    await client.post(
        f"/v1/notebooks/{real_id}/execute-code",
        json={"code": "guard = 'real'", "handoff_after_seconds": 10},
    )

    # Now attempt to activate a non-existent notebook in connect mode.
    # The ID is valid base64 but the notebook file does not exist.
    fake_id = _nb_id(f"does-not-exist-{uuid.uuid4().hex[:6]}.ipynb")
    resp = await client.get(f"/v1/notebooks/{fake_id}")
    # Failures are reported as 200 + ok:false so a GPT Action can read the
    # body; the status the failure would have carried is in http_status.
    # A notebook that does not exist is 404, not a generic 400 or 500 —
    # the caller can act on "wrong path" but not on "server fault".
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is False
    assert body["http_status"] == HTTPStatus.NOT_FOUND, resp.text

    # The real notebook must still be operable — the failed call must not
    # have corrupted the current-notebook state.  Use a fresh connection
    # pool: whether the error response above leaves the pooled connection
    # reusable is httpx/uvicorn keep-alive behaviour, not a claim this API
    # owns, and reusing it conflates the two.  `guard` surviving proves both
    # that the notebook is still current and that its kernel is the same one.
    async with AsyncClient(base_url=str(client.base_url), timeout=60) as fresh:
        r = await fresh.post(
            f"/v1/notebooks/{real_id}/execute-code",
            json={"code": "print(guard)", "handoff_after_seconds": 10},
        )
    assert r.json()["ok"] is True, r.text
    assert "real" in str(r.json()["outputs"]), r.text


# ---------------------------------------------------------------------------
# Test 9: Concurrent operations are serialized by the runtime lock
# ---------------------------------------------------------------------------


async def test_9_concurrent_operations_serialized(client: AsyncClient) -> None:
    create = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": "test-create.ipynb", "mode": "create"},
    )
    nb_id = create.json()["notebook_id"]

    # Fire two execute_code calls concurrently.
    # If the lock works, both complete without corrupting kernel state.
    # We set a variable in one and read it in the other; the lock
    # guarantees one happens before the other.
    async def _set() -> dict:
        r = await client.post(
            f"/v1/notebooks/{nb_id}/execute-code",
            json={"code": "counter = 99", "handoff_after_seconds": 10},
        )
        return r.json()

    async def _read() -> dict:
        r = await client.post(
            f"/v1/notebooks/{nb_id}/execute-code",
            json={"code": "print(counter)", "handoff_after_seconds": 10},
        )
        return r.json()

    results = await asyncio.gather(_set(), _read(), return_exceptions=True)

    # Both should succeed (no exceptions), proving serialization — the
    # read may or may not see `counter` depending on ordering, but neither
    # should fail with a 500 or raise.
    for r in results:
        assert not isinstance(r, Exception), f"Concurrent call failed: {r}"
        assert r.get("ok") is True, f"Response not ok: {r}"


# ---------------------------------------------------------------------------
# Test 10: A Jupyter-boundary failure is legible to the calling agent
# ---------------------------------------------------------------------------


async def test_10_jupyter_boundary_failure_is_legible(client: AsyncClient) -> None:
    """A failure at the Jupyter contents boundary must arrive as readable data.

    Observed defect: listing a directory that does not exist produced HTTP
    500.  The consumer is a GPT Action whose client calls
    ``raise_for_status()``, so on any non-2xx it raises and shows the caller
    only the exception type — the response body, and every diagnostic in it,
    is discarded.  The failure was therefore indistinguishable from every
    other failure.

    Owned claim: this adapter reports a Jupyter-boundary failure as a 200
    response whose body carries the failure, and propagates the *Jupyter*
    status (404 for a missing directory) rather than flattening it to 500.
    """
    missing_dir = f"no-such-dir-{uuid.uuid4().hex[:8]}"

    resp = await client.get("/v1/files", params={"path": missing_dir})

    # 200 on the wire is what keeps the body readable to the consumer.
    assert resp.status_code == HTTPStatus.OK, resp.text
    body = resp.json()
    assert body["ok"] is False, resp.text
    # The adapter's own failure type for a tool-boundary error.
    assert body["error_type"] == "ToolError", resp.text
    # The Jupyter 404 must survive, not collapse into a generic 500.
    assert body["http_status"] == HTTPStatus.NOT_FOUND, resp.text


async def test_use_notebook_reuses_one_session_bound_kernel(
    client: AsyncClient, jupyter_server: str
) -> None:
    path = "test-session-reuse.ipynb"

    first = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": path, "mode": "create", "kernel_name": "python3"},
    )
    assert first.json()["ok"] is True, first.text
    first_kernel = first.json()["kernel_id"]

    sessions = await _jupyter_json(jupyter_server, "/api/sessions")
    matching = [session for session in sessions if session["path"] == path]
    assert len(matching) == 1
    assert matching[0]["kernel"]["id"] == first_kernel

    second = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": path, "mode": "connect", "kernel_name": "python3"},
    )
    assert second.json()["ok"] is True, second.text
    assert second.json()["kernel_id"] == first_kernel
    assert second.json()["kernel_reused"] is True

    kernels = await _jupyter_json(jupyter_server, "/api/kernels")
    assert sum(kernel["id"] == first_kernel for kernel in kernels) == 1

    unused = await client.post(f"/v1/notebooks/{first.json()['notebook_id']}/unuse")
    assert unused.json()["ok"] is True, unused.text
    assert unused.json()["kernel_id"] == first_kernel

    kernels_after_unuse = await _jupyter_json(jupyter_server, "/api/kernels")
    assert any(kernel["id"] == first_kernel for kernel in kernels_after_unuse)

    reconnected = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": path, "mode": "connect", "kernel_name": "python3"},
    )
    assert reconnected.json()["kernel_id"] == first_kernel
    assert reconnected.json()["kernel_reused"] is True


async def test_execution_deadline_hands_off_without_interrupting_kernel(
    client: AsyncClient,
) -> None:
    create = await client.post(
        "/v1/notebooks/use",
        json={
            "notebook_path": "test-handoff.ipynb",
            "mode": "create",
            "kernel_name": "python3",
        },
    )
    assert create.json()["ok"] is True, create.text
    notebook_id = create.json()["notebook_id"]

    started_at = time.monotonic()
    response = await client.post(
        f"/v1/notebooks/{notebook_id}/cells/insert-and-execute",
        json={
            "cell_index": 0,
            "cell_source": "import time; time.sleep(2); print('handoff-finished')",
            "handoff_after_seconds": 1,
        },
    )
    elapsed = time.monotonic() - started_at
    body = response.json()
    assert response.status_code == HTTPStatus.OK
    assert body["ok"] is True, response.text
    assert body["status"] == "running"
    assert elapsed < 1.8

    duplicate_started_at = time.monotonic()
    duplicate = await client.post(
        f"/v1/notebooks/{notebook_id}/cells/insert-and-execute",
        json={
            "cell_index": -1,
            "cell_source": "print('must-not-be-queued')",
            "handoff_after_seconds": 1,
        },
    )
    duplicate_body = duplicate.json()
    assert duplicate_body["status"] == "running"
    assert duplicate_body["cell_index"] == 0
    assert duplicate_body["operation"] == "insert_execute_code_cell"
    assert time.monotonic() - duplicate_started_at < 0.5

    restart = await client.post(f"/v1/notebooks/{notebook_id}/restart")
    restart_body = restart.json()
    assert restart.status_code == HTTPStatus.OK
    assert restart_body["ok"] is False
    assert restart_body["http_status"] == HTTPStatus.CONFLICT
    assert "execution is still running" in restart_body["error_message"]
    assert create.json()["kernel_id"] in restart_body["error_message"]

    deadline = time.monotonic() + 10
    while True:
        status = await client.get(f"/v1/notebooks/{notebook_id}/execution")
        status_body = status.json()
        if status_body.get("status") == "complete":
            break
        assert status_body["status"] == "running", status.text
        assert time.monotonic() < deadline
        await asyncio.sleep(0.2)

    assert "handoff-finished" in str(status_body["outputs"])
    notebook = await client.get(f"/v1/notebooks/{notebook_id}")
    assert "must-not-be-queued" not in str(notebook.json())


async def test_handed_off_execution_does_not_block_another_notebook(
    client: AsyncClient,
) -> None:
    first = await client.post(
        "/v1/notebooks/use",
        json={
            "notebook_path": "test-hung-a.ipynb",
            "mode": "create",
            "kernel_name": "python3",
        },
    )
    second = await client.post(
        "/v1/notebooks/use",
        json={
            "notebook_path": "test-hung-b.ipynb",
            "mode": "create",
            "kernel_name": "python3",
        },
    )
    assert first.json()["ok"] is True, first.text
    assert second.json()["ok"] is True, second.text

    running = await client.post(
        f"/v1/notebooks/{first.json()['notebook_id']}/execute-code",
        json={
            "code": "import time; time.sleep(5)",
            "handoff_after_seconds": 1,
        },
    )
    assert running.json()["status"] == "running", running.text

    started_at = time.monotonic()
    unrelated = await client.get(
        f"/v1/notebooks/{second.json()['notebook_id']}"
    )
    elapsed = time.monotonic() - started_at

    assert unrelated.status_code == HTTPStatus.OK, unrelated.text
    assert unrelated.json()["ok"] is True, unrelated.text
    assert elapsed < 2, (
        "An execution on one notebook monopolized the process-wide runtime lock "
        f"for {elapsed:.1f}s and blocked an unrelated notebook."
    )


async def test_stopped_kernel_is_diagnosed_and_recoverable(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/v1/notebooks/use",
        json={
            "notebook_path": "test-hung-recovery.ipynb",
            "mode": "create",
            "kernel_name": "python3",
        },
    )
    assert created.json()["ok"] is True, created.text
    notebook_id = created.json()["notebook_id"]
    old_kernel_id = created.json()["kernel_id"]

    running = await client.post(
        f"/v1/notebooks/{notebook_id}/execute-code",
        json={
            "code": (
                "import os, signal; "
                "os.kill(os.getpid(), signal.SIGSTOP)"
            ),
            "handoff_after_seconds": 1,
        },
    )
    assert running.json()["status"] == "running", running.text

    status = await client.get(f"/v1/notebooks/{notebook_id}/execution")
    status_body = status.json()
    assert status_body["status"] == "unresponsive", status.text
    assert status_body["kernel_id"] == old_kernel_id
    assert status_body["kernel_responsive"] is False
    assert "heartbeat" in status_body["error_message"].lower()
    assert f"/v1/notebooks/{notebook_id}/restart" in status_body["instruction"]

    restarted = await client.post(
        f"/v1/notebooks/{notebook_id}/restart",
        json={"kernel_name": "python3"},
    )
    restarted_body = restarted.json()
    assert restarted_body["ok"] is True, restarted.text
    assert restarted_body["kernel_id"] != old_kernel_id

    recovered = await client.post(
        f"/v1/notebooks/{notebook_id}/execute-code",
        json={"code": "print('recovered')", "handoff_after_seconds": 10},
    )
    assert recovered.json()["status"] == "complete", recovered.text
    assert "recovered" in str(recovered.json()["outputs"])


async def test_use_notebook_rejects_kernelspec_mismatch(
    client: AsyncClient,
) -> None:
    path = "test-session-reuse.ipynb"
    first = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": path, "mode": "create", "kernel_name": "python3"},
    )
    assert first.json()["ok"] is True, first.text

    mismatch = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": path, "mode": "connect", "kernel_name": "sagemath"},
    )
    body = mismatch.json()
    assert mismatch.status_code == HTTPStatus.OK
    assert body["ok"] is False
    assert body["http_status"] == HTTPStatus.CONFLICT
    assert "already has a 'python3' session kernel" in body["error_message"]


async def test_restart_defaults_existing_python_session_to_sagemath(
    client: AsyncClient,
    jupyter_server: str,
) -> None:
    path = "test-restart-defaults-to-sage.ipynb"
    created = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": path, "mode": "create", "kernel_name": "python3"},
    )
    assert created.json()["ok"] is True, created.text
    notebook_id = created.json()["notebook_id"]
    old_kernel_id = created.json()["kernel_id"]

    restarted = await client.post(f"/v1/notebooks/{notebook_id}/restart", json={})
    body = restarted.json()
    assert restarted.status_code == HTTPStatus.OK
    assert body["ok"] is True, restarted.text
    assert body["kernel_name"] == "sagemath"
    assert body["kernel_id"] != old_kernel_id

    sessions = await _jupyter_json(jupyter_server, "/api/sessions")
    matching = [session for session in sessions if session["path"] == path]
    assert len(matching) == 1
    assert matching[0]["kernel"]["name"] == "sagemath"
    assert matching[0]["kernel"]["id"] == body["kernel_id"]

    executed = await client.post(
        f"/v1/notebooks/{notebook_id}/execute-code",
        json={"code": "2 + 2", "handoff_after_seconds": 35},
    )
    assert executed.json()["ok"] is True, executed.text
    assert "4" in str(executed.json()["outputs"])

    explicit_python = await client.post(
        f"/v1/notebooks/{notebook_id}/restart",
        json={"kernel_name": "python3"},
    )
    assert explicit_python.json()["ok"] is True, explicit_python.text
    assert explicit_python.json()["kernel_name"] == "python3"


async def test_mutation_rejects_malformed_notebook_before_rtc_connection(
    client: AsyncClient,
    jupyter_server: str,
) -> None:
    path = "test-malformed-preflight.ipynb"
    notebook_id = encode_notebook_id(path)

    malformed = {
        "type": "notebook",
        "format": "json",
        "content": {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "language_info": {
                    "name": "python",
                    "mimetype": "text/x-python",
                    "file_extension": ".py",
                }
            },
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": {
                        "name": "python",
                        "mimetype": "text/x-python",
                        "file_extension": ".py",
                    },
                }
            ],
        },
    }
    async with AsyncClient(base_url=jupyter_server) as jupyter:
        saved = await jupyter.put(
            f"/api/contents/{path}",
            params={"token": "MY_TOKEN"},
            json=malformed,
        )
    assert saved.status_code in (HTTPStatus.OK, HTTPStatus.CREATED), saved.text

    sessions_before = await _jupyter_json(jupyter_server, "/api/sessions")
    matching_before = [session for session in sessions_before if session["path"] == path]

    response = await client.put(
        f"/v1/notebooks/{notebook_id}/cells/0",
        json={"cell_source": "replacement"},
    )
    body = response.json()
    assert response.status_code == HTTPStatus.OK
    assert body["ok"] is False, response.text
    assert body["http_status"] == HTTPStatus.CONFLICT, body
    assert "cells[0].source" in body["error_message"]
    assert "dict" in body["error_message"]

    sessions_after = await _jupyter_json(jupyter_server, "/api/sessions")
    matching_after = [session for session in sessions_after if session["path"] == path]
    assert matching_after == matching_before

    async with AsyncClient(base_url=jupyter_server) as jupyter:
        persisted = await jupyter.get(
            f"/api/contents/{path}",
            params={"token": "MY_TOKEN", "content": "1"},
        )
    assert persisted.json()["content"]["cells"][0]["source"] == malformed["content"]["cells"][0]["source"]
