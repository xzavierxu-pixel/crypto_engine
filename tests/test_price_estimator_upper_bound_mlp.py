from __future__ import annotations

import numpy as np

from price_estimator.upper_bound_mlp.train_upper_bound_mlp import (
    logit_np,
    minimal_feasible_logit_shift,
    sigmoid_np,
    upper_bound_metrics,
)


def test_minimal_feasible_logit_shift_enforces_training_constraint() -> None:
    y = np.array([0.20, 0.50, 0.90], dtype=np.float32)
    epsilon = 0.01
    target_z = logit_np(np.clip(y + epsilon, 1e-6, 1.0 - 1e-6))
    z = target_z - np.array([0.00, 0.25, 0.10], dtype=np.float32)

    shift = minimal_feasible_logit_shift(target_z, z, enabled=True)
    metrics = upper_bound_metrics(y, z + shift, epsilon=epsilon, tolerance=1e-6)

    assert shift >= 0.25
    assert metrics["violation_rate"] == 0.0
    assert metrics["coverage"] == 1.0
    assert metrics["min_gap"] + 1e-6 >= epsilon


def test_upper_bound_metrics_reports_violations_in_probability_space() -> None:
    y = np.array([0.20, 0.50], dtype=np.float32)
    epsilon = 0.01
    p = np.array([0.22, 0.505], dtype=np.float32)
    z = logit_np(p)

    metrics = upper_bound_metrics(y, z, epsilon=epsilon, tolerance=1e-6)

    assert metrics["sample_count"] == 2.0
    assert metrics["violation_rate"] == 0.5
    assert metrics["coverage"] == 0.5
    assert np.isclose(metrics["max_violation"], 0.005, atol=1e-6)
    assert np.allclose(sigmoid_np(z), p, atol=1e-6)
