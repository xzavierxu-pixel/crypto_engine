from __future__ import annotations

import pandas as pd

from src.model.drift import Stage1DriftMonitor, Stage2DirectionDriftMonitor, compute_ks_distance


def test_compute_ks_distance_detects_distribution_shift() -> None:
    reference = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    observed = pd.Series([0.8, 0.85, 0.9, 0.95, 0.99])

    ks = compute_ks_distance(reference, observed)

    assert ks > 0.5


def test_stage1_drift_monitor_flags_large_shift_after_minimum_history() -> None:
    monitor = Stage1DriftMonitor(
        pd.Series([0.1, 0.2, 0.3] * 100),
        threshold=0.1,
        window_size=60,
        min_history=50,
        alert_consecutive=1,
    )

    state = {}
    for _ in range(60):
        state = monitor.update(0.95)

    assert state["window_size"] == 60
    assert state["ks_distance"] > 0.1
    assert state["threshold_breached"] is True
    assert state["consecutive_alerts"] >= 1
    assert state["alert"] is True


def test_stage1_drift_monitor_requires_consecutive_breaches_before_alerting() -> None:
    monitor = Stage1DriftMonitor(
        pd.Series([0.1, 0.2, 0.3] * 100),
        threshold=0.1,
        window_size=5,
        min_history=3,
        alert_consecutive=2,
    )

    state = monitor.update(0.95)
    state = monitor.update(0.95)
    assert state["threshold_breached"] is False
    assert state["alert"] is False

    state = monitor.update(0.95)
    assert state["threshold_breached"] is True
    assert state["consecutive_alerts"] == 1
    assert state["alert"] is False

    state = monitor.update(0.95)
    assert state["consecutive_alerts"] == 2
    assert state["alert"] is True


def test_stage2_direction_drift_monitor_tracks_p_up_minus_p_down_shift() -> None:
    monitor = Stage2DirectionDriftMonitor(
        pd.Series([0.1, 0.15, 0.2] * 100),
        threshold=0.1,
        window_size=20,
        min_history=10,
        alert_consecutive=1,
    )

    state = {}
    for _ in range(20):
        state = monitor.update(p_up=0.95, p_down=0.05)

    assert state["window_size"] == 20
    assert state["threshold_breached"] is True
    assert state["alert"] is True
