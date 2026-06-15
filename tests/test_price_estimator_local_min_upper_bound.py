from __future__ import annotations

import numpy as np
import pandas as pd

from price_estimator.upper_bound_mlp.train_local_min_upper_bound import (
    fit_local_non_normalized_margins,
    local_min_upper_bound_metrics,
    local_min_upper_bound_predict,
    selective_grouped_diagnostics,
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


def test_acceptance_filter_rejects_high_p_side_before_ranking() -> None:
    y = np.array([0.20, 0.20, 0.20], dtype=np.float32)
    mu = np.array([0.20, 0.20, 0.20], dtype=np.float32)
    margin = np.array([0.02, 0.02, 0.02], dtype=np.float32)
    p_side = np.array([0.70, 0.80, 0.90], dtype=np.float32)
    eligible = p_side < 0.80

    pred = local_min_upper_bound_predict(mu, margin, p_side, q=1.0, sigma_floor=0.01, margin_threshold=0.03, eligible=eligible)
    metrics = local_min_upper_bound_metrics(y, p_side, pred, epsilon=0.01, tolerance=1e-6)

    assert pred.accepted.tolist() == [True, False, False]
    assert metrics["removed_sample_count"] == 2.0
    assert metrics["eligible_sample_count"] == 1.0


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


def test_grouped_diagnostics_preserves_accepted_subset_values_without_index_alignment_na() -> None:
    df = pd.DataFrame(
        {
            "target_raw": [0.1, 0.2, 0.3, 0.4],
            "p_side": [0.45, 0.55, 0.65, 0.75],
            "p_bin": ["0.40_0.50", "0.55_0.60", "0.60_0.65", "0.70_1.00"],
            "selected_side": ["UP", "DOWN", "UP", "DOWN"],
        }
    )
    y = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    pred = local_min_upper_bound_predict(
        mu=np.array([0.2, 0.2, 0.3, 0.4], dtype=np.float32),
        sigma=np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32),
        p_side=df["p_side"].to_numpy(dtype=np.float32),
        q=1.0,
        sigma_floor=0.01,
        margin_threshold=0.01,
    )
    config = {
        "diagnostics": {
            "group_columns": ["p_bin", "selected_side"],
            "p_side_bin_edges": [0.0, 0.5, 0.6, 0.7, 1.0],
            "price_bin_column": "target_raw",
            "price_bin_edges": [0.0, 0.2, 0.4, 1.0],
        }
    }

    diagnostics = selective_grouped_diagnostics(df, y, df["p_side"].to_numpy(), pred, 0.01, 1e-6, config)

    assert diagnostics["na_rates"]["p_bin_na_rate_accepted"] == 0.0
    assert diagnostics["na_rates"]["selected_side_na_rate_accepted"] == 0.0
    assert diagnostics["na_rates"]["price_bin_na_rate_accepted"] == 0.0


def test_local_non_normalized_margin_uses_group_then_global_fallback() -> None:
    calibration = pd.DataFrame(
        {
            "target_raw": [0.2, 0.3, 0.4, 0.5],
            "p_side": [0.55, 0.56, 0.75, 0.76],
            "selected_side": ["UP", "UP", "DOWN", "DOWN"],
        }
    )
    config = {
        "calibration": {"coverage_quantile": 0.9},
        "non_normalized_conformal": {"coverage_quantile": 0.9, "min_group_count": 2},
        "diagnostics": {
            "p_side_bin_edges": [0.0, 0.6, 1.0],
            "price_bin_column": "target_raw",
            "price_bin_edges": [0.0, 0.35, 1.0],
        },
    }

    model = fit_local_non_normalized_margins(
        calibration,
        y_cal=np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        mu_cal=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        config=config,
    )

    assert model["selected_side_p_side_bin"][("UP", "0.00_0.60")] > 0.0
    assert model["global_margin"] > 0.0
