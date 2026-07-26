# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Real-process proof for Assistant API health and request correlation."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from http import HTTPStatus

import requests


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_health_probes_jupyter_and_logs_request_correlation() -> None:
    api_port = _free_port()
    unreachable_jupyter_port = _free_port()
    request_id = "health-correlation-proof"
    env = {
        **os.environ,
        "JUPYTER_URL": f"http://127.0.0.1:{unreachable_jupyter_port}",
        "JUPYTER_TOKEN": "",
        "PORT": str(api_port),
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "jupyter_mcp_server.assistant_api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
            "--workers",
            "1",
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    response = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                response = requests.get(
                    f"http://127.0.0.1:{api_port}/health",
                    headers={"X-Request-ID": request_id},
                    timeout=5,
                )
                break
            except requests.ConnectionError:
                time.sleep(0.2)
        assert response is not None, "Assistant API did not start."
        assert response.status_code == HTTPStatus.OK
        assert response.headers["X-Request-ID"] == request_id
        body = response.json()
        assert body["ok"] is False
        assert body["status"] == "unhealthy"
        assert f"127.0.0.1:{unreachable_jupyter_port}" in body["error"]
    finally:
        process.terminate()
        stdout, _ = process.communicate(timeout=10)

    assert "assistant_request_start" in stdout
    assert "assistant_request_end" in stdout
    assert f"request_id={request_id}" in stdout
    assert "path=/health" in stdout
