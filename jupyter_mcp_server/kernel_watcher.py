"""Periodic detection and desktop notification for unresponsive Jupyter kernels."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from jupyter_kernel_client import KernelClient

from jupyter_mcp_server.assistant_runtime import configure_jupyter
from jupyter_mcp_server.config import get_config

KernelEvent = Literal["unresponsive", "recovered"]
logger = logging.getLogger("jupyter_kernel_watch")


@dataclass(frozen=True)
class KernelWatchState:
    """Persistent state for one kernel across timer invocations."""

    busy_since: float | None = None
    alerted_at: float | None = None


def classify_kernel(
    previous: KernelWatchState,
    *,
    execution_state: str,
    now: float,
    threshold_seconds: float,
    probe_responsive: bool | None,
    repeat_seconds: float,
) -> tuple[KernelWatchState, KernelEvent | None]:
    """Advance one kernel's state and return an alert-worthy transition."""
    if execution_state != "busy":
        event: KernelEvent | None = (
            "recovered" if previous.alerted_at is not None else None
        )
        return KernelWatchState(), event

    busy_since = previous.busy_since if previous.busy_since is not None else now
    current = KernelWatchState(
        busy_since=busy_since,
        alerted_at=previous.alerted_at,
    )
    if now - busy_since < threshold_seconds or probe_responsive is not False:
        return current, None

    if (
        previous.alerted_at is None
        or now - previous.alerted_at >= repeat_seconds
    ):
        return KernelWatchState(busy_since=busy_since, alerted_at=now), "unresponsive"
    return current, None


def _state_path() -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime_dir:
        raise RuntimeError("XDG_RUNTIME_DIR is required for kernel watcher state.")
    return Path(runtime_dir) / "jupyter-kernel-watch.json"


def _load_state(path: Path) -> dict[str, KernelWatchState]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Watcher state in {path} must be a JSON object.")
    return {kernel_id: KernelWatchState(**value) for kernel_id, value in raw.items()}


def _save_state(path: Path, state: dict[str, KernelWatchState]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {kernel_id: asdict(value) for kernel_id, value in state.items()},
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def _get_json(path: str) -> list[dict[str, object]]:
    config = get_config()
    url = f"{config.runtime_url.rstrip('/')}{path}"
    if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError(f"Jupyter URL must use HTTP(S), got {url!r}.")
    headers = {}
    if config.runtime_token:
        headers["Authorization"] = f"token {config.runtime_token}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError(f"Jupyter returned a non-list response for {path}.")
    return payload


def _probe_control_channel(kernel_id: str, timeout_seconds: float = 4.0) -> bool:
    """Ask an existing kernel for kernel_info on its control channel."""
    config = get_config()
    kernel = KernelClient(
        server_url=config.runtime_url,
        token=config.runtime_token,
        kernel_id=kernel_id,
    )
    try:
        # The client currently prints WebSocket diagnostics, which can include auth data.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            kernel.start(timeout=timeout_seconds)
            client = kernel._manager.client
            request = client.session.msg("kernel_info_request")
            request_id = request["header"]["msg_id"]
            client.control_channel.send(request)
            deadline = time.monotonic() + timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                reply = client.control_channel.get_msg(timeout=remaining)
                if reply.get("parent_header", {}).get("msg_id") == request_id:
                    return reply.get("msg_type") == "kernel_info_reply"
    except Exception:
        return False
    finally:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            kernel.stop(shutdown_kernel=False)


def _notify(summary: str, body: str, *, urgency: str) -> None:
    completed = subprocess.run(  # noqa: S603
        [
            "/usr/bin/notify-send",
            "--app-name=Jupyter Assistant",
            f"--urgency={urgency}",
            "--print-id",
            summary,
            body,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    notification_id = completed.stdout.strip()
    logger.info(
        "kernel_watch_notification notification_id=%r summary=%r",
        notification_id,
        summary,
    )


def _session_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for session in _get_json("/api/sessions"):
        kernel = session.get("kernel")
        notebook = session.get("notebook")
        if not isinstance(kernel, dict) or not isinstance(notebook, dict):
            continue
        kernel_id = kernel.get("id")
        path = notebook.get("path")
        if isinstance(kernel_id, str) and isinstance(path, str):
            names[kernel_id] = path
    return names


def run_watch(
    *,
    state_path: Path,
    threshold_seconds: float,
    repeat_seconds: float,
) -> None:
    """Inspect all kernels once and persist state for the next timer invocation."""
    configure_jupyter()
    now = time.time()
    previous = _load_state(state_path)
    current: dict[str, KernelWatchState] = {}
    sessions = _session_names()
    kernels = _get_json("/api/kernels")

    for kernel in kernels:
        kernel_id = kernel.get("id")
        execution_state = kernel.get("execution_state")
        if not isinstance(kernel_id, str) or not isinstance(execution_state, str):
            raise ValueError("Jupyter kernel model lacks string id or execution_state.")

        old = previous.get(kernel_id, KernelWatchState())
        probe_responsive: bool | None = None
        busy_since = old.busy_since if old.busy_since is not None else now
        if execution_state == "busy" and now - busy_since >= threshold_seconds:
            probe_responsive = _probe_control_channel(kernel_id)

        new, event = classify_kernel(
            old,
            execution_state=execution_state,
            now=now,
            threshold_seconds=threshold_seconds,
            probe_responsive=probe_responsive,
            repeat_seconds=repeat_seconds,
        )
        current[kernel_id] = new
        notebook = sessions.get(kernel_id, "unknown notebook")
        elapsed_minutes = (
            0 if new.busy_since is None else round((now - new.busy_since) / 60)
        )
        logger.info(
            "kernel_watch_observation kernel_id=%s notebook=%r state=%s "
            "busy_minutes=%s control_responsive=%s event=%s",
            kernel_id,
            notebook,
            execution_state,
            elapsed_minutes,
            probe_responsive,
            event,
        )
        if event == "unresponsive":
            _notify(
                "Jupyter kernel may be hung",
                f"{notebook}\nKernel {kernel_id}\nBusy for about "
                f"{elapsed_minutes} minutes and did not answer its control channel.\n"
                "No automatic restart was performed.",
                urgency="critical",
            )
        elif event == "recovered":
            _notify(
                "Jupyter kernel recovered",
                f"{notebook}\nKernel {kernel_id} is {execution_state} again.",
                urgency="normal",
            )

    for kernel_id, old in previous.items():
        if kernel_id not in current and old.alerted_at is not None:
            _notify(
                "Jupyter kernel no longer running",
                f"Kernel {kernel_id} disappeared after an unresponsive alert.",
                urgency="normal",
            )

    _save_state(state_path, current)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-seconds", type=float, default=1200.0)
    parser.add_argument("--repeat-seconds", type=float, default=3600.0)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--test-notification", action="store_true")
    args = parser.parse_args()

    if args.threshold_seconds <= 0 or args.repeat_seconds <= 0:
        parser.error("threshold and repeat seconds must be positive")
    if args.test_notification:
        _notify(
            "Jupyter kernel watcher enabled",
            "Desktop notifications from the kernel watcher are working.",
            urgency="normal",
        )
        return

    run_watch(
        state_path=args.state_file or _state_path(),
        threshold_seconds=args.threshold_seconds,
        repeat_seconds=args.repeat_seconds,
    )


if __name__ == "__main__":
    main()
