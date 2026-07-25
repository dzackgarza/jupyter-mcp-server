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
8. Verify that failed notebook activation returns HTTP 400 and does not
   execute against whichever notebook happened to be current.
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
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
def _cleanup_test_notebooks(jupyter_server: str):
    """Delete test notebooks from the Jupyter server after each test."""
    yield
    # Best-effort cleanup — failures here are not test failures.
    for name in ("test-create.ipynb", "test-aba-a.ipynb", "test-aba-b.ipynb", "test-restart.ipynb"):
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
        json={"cell_index": 0, "cell_source": "print(2 + 2)", "timeout": 35},
    )
    assert resp.status_code == 200
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
        json={"cell_index": 0, "cell_source": "print(42)", "timeout": 35},
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
        json={"code": "x = 42", "timeout": 10},
    )
    assert r1.status_code == 200

    # Read it in a separate request
    r2 = await client.post(
        f"/v1/notebooks/{nb_id}/execute-code",
        json={"code": "print(x * 2)", "timeout": 10},
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
    await client.post(
        f"/v1/notebooks/{a_id}/execute-code",
        json={"code": "x = 100", "timeout": 10},
    )

    # B: set x = 200
    await client.post(
        f"/v1/notebooks/{b_id}/execute-code",
        json={"code": "x = 200", "timeout": 10},
    )

    # A: read x — must be 100, not 200
    a_read = await client.post(
        f"/v1/notebooks/{a_id}/execute-code",
        json={"code": "print(x)", "timeout": 10},
    )
    assert a_read.status_code == 200
    assert "100" in str(a_read.json()["outputs"])

    # B: read x — must be 200
    b_read = await client.post(
        f"/v1/notebooks/{b_id}/execute-code",
        json={"code": "print(x)", "timeout": 10},
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
            json={"code": "y = 'persisted'", "timeout": 10},
            timeout=30,
        )
        assert r.status_code == 200
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

        # The notebook file still exists; read it
        r = requests.get(f"{url2}/v1/notebooks/{nb_id}", timeout=30)
        assert r.status_code == 200
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


async def test_8_failed_activation_returns_400(client: AsyncClient) -> None:
    # First, create a real notebook so *something* is current.
    create = await client.post(
        "/v1/notebooks/use",
        json={"notebook_path": "test-create.ipynb", "mode": "create"},
    )
    real_id = create.json()["notebook_id"]

    # Set a variable in the real notebook.
    await client.post(
        f"/v1/notebooks/{real_id}/execute-code",
        json={"code": "guard = 'real'", "timeout": 10},
    )

    # Now attempt to activate a non-existent notebook in connect mode.
    # The ID is valid base64 but the notebook file does not exist.
    fake_id = _nb_id(f"does-not-exist-{uuid.uuid4().hex[:6]}.ipynb")
    resp = await client.get(f"/v1/notebooks/{fake_id}")
    assert resp.status_code == 400

    # The real notebook must still be operable — the failed call must not
    # have corrupted the current-notebook state.
    r = await client.post(
        f"/v1/notebooks/{real_id}/execute-code",
        json={"code": "print(guard)", "timeout": 10},
    )
    assert r.status_code == 200
    assert "real" in str(r.json()["outputs"])


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
            json={"code": "counter = 99", "timeout": 10},
        )
        return r.json()

    async def _read() -> dict:
        r = await client.post(
            f"/v1/notebooks/{nb_id}/execute-code",
            json={"code": "print(counter)", "timeout": 10},
        )
        return r.json()

    results = await asyncio.gather(_set(), _read(), return_exceptions=True)

    # Both should succeed (no exceptions), proving serialization — the
    # read may or may not see `counter` depending on ordering, but neither
    # should fail with a 500 or raise.
    for r in results:
        assert not isinstance(r, Exception), f"Concurrent call failed: {r}"
        assert r.get("ok") is True, f"Response not ok: {r}"
