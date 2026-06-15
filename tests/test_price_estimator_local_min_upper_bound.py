from __future__ import annotations

import numpy as np

from price_estimator.upper_bound_mlp.train_local_min_upper_bound import (
    local_min_upper_bound_metrics,
    local_min_upper_bound_predict,
    select_margin_threshold,
)


def test_local_min_upper_bound_rejects_side_infeasible_without_clipping() -> None:
    y = np.array([0.20, 0.40], dtype=np.float32)
    mu = np.array([0.20, 0.44], dtype=np.float32)
    sigma = np.array([0.10, 0.10], dtype=np.float32)
    p_side = np.array([0.35, 0.50], dtype=np.float32)

    pred = local_min_upper_bound_predict(mu, sigma, p_side, q=1.0, sigma_floor=0.01, margin_threshold=0.2)

    assert pred.accepted.tolist() == [True, False]
    assert np.isclose(pred.p_raw[1], 0.54)
    assert np.isnan(pred.p_pred[1])


def test_local_min_upper_bound_metrics_only_evaluates_accepted_samples() -> None:
    y = np.array([0.20, 0.40, 0.50], dtype=np.float32)
    mu = np.array([0.20, 0.39, 0.52], dtype=np.float32)
    sigma = np.array([0.02, 0.02, 0.02], dtype=np.float32)
    p_side = np.array([0.80, 0.80, 0.80], dtype=np.float32)
    pred = local_min_upper_bound_predict(mu, sigma, p_side, q=1.0, sigma_floor=0.01, margin_threshold=0.021)

    metrics = local_min_upper_bound_metrics(y, p_side, pred, epsilon=0.01, tolerance=1e-6)

    assert metrics["accepted_count"] == 3.0
    assert metrics["accepted_coverage"] == 1.0
    assert metrics["side_violation_rate"] == 0.0
    assert metrics["covered_mean_gap"] > 0.0


def test_select_margin_threshold_enforces_coverage_constraint() -> None:
    y = np.array([0.22, 0.30, 0.40, 0.50], dtype=np.float32)
    mu = np.array([0.21, 0.31, 0.36, 0.46], dtype=np.float32)
    sigma = np.array([0.01, 0.01, 0.05, 0.05], dtype=np.float32)
    p_side = np.array([0.55, 0.55, 0.55, 0.55], dtype=np.float32)

    threshold, metrics, pred = select_margin_threshold(
        y,
        p_side,
        mu,
        sigma,
        q=1.0,
        sigma_floor=0.01,
        epsilon=0.01,
        tolerance=1e-6,
        thresholds=[0.01, 0.051],
        min_accepted_coverage=0.70,
    )

    assert threshold == 0.051
    assert metrics["accepted_coverage"] >= 0.70
    assert pred.accepted.sum() == 4
