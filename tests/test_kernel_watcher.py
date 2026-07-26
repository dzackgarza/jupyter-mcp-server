"""Behavioral proof for the periodic kernel watcher."""

from pathlib import Path

from jupyter_mcp_server.kernel_watcher import KernelWatchState, classify_kernel


def test_busy_kernel_requires_threshold_and_failed_probe_before_alerting():
    state = KernelWatchState()

    state, event = classify_kernel(
        state,
        execution_state="busy",
        now=100.0,
        threshold_seconds=1200.0,
        probe_responsive=None,
        repeat_seconds=3600.0,
    )
    assert event is None
    assert state.busy_since == 100.0

    state, event = classify_kernel(
        state,
        execution_state="busy",
        now=1299.0,
        threshold_seconds=1200.0,
        probe_responsive=True,
        repeat_seconds=3600.0,
    )
    assert event is None

    state, event = classify_kernel(
        state,
        execution_state="busy",
        now=1301.0,
        threshold_seconds=1200.0,
        probe_responsive=False,
        repeat_seconds=3600.0,
    )
    assert event == "unresponsive"
    assert state.alerted_at == 1301.0


def test_alert_is_rate_limited_and_idle_transition_reports_recovery():
    state = KernelWatchState(busy_since=100.0, alerted_at=1301.0)

    state, event = classify_kernel(
        state,
        execution_state="busy",
        now=1400.0,
        threshold_seconds=1200.0,
        probe_responsive=False,
        repeat_seconds=3600.0,
    )
    assert event is None

    state, event = classify_kernel(
        state,
        execution_state="idle",
        now=1500.0,
        threshold_seconds=1200.0,
        probe_responsive=None,
        repeat_seconds=3600.0,
    )
    assert event == "recovered"
    assert state == KernelWatchState()


def test_systemd_timer_uses_repository_owned_watcher():
    root = Path(__file__).parents[1]
    service = (root / "dev/systemd/jupyter-kernel-watch.service").read_text()
    timer = (root / "dev/systemd/jupyter-kernel-watch.timer").read_text()

    assert "jupyter-kernel-watch" in service
    assert "--threshold-seconds 1200" in service
    assert "OnUnitActiveSec=5min" in timer
    assert "Persistent=true" in timer
