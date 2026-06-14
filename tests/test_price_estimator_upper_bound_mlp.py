from __future__ import annotations

import numpy as np

from price_estimator.upper_bound_mlp.train_upper_bound_mlp import (
    constraint_weights_from_y,
    grouped_diagnostics,
    is_better,
    logit_np,
    sigmoid_np,
    tightness_weights_from_y,
    upper_bound_metrics,
)


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


def test_grouped_diagnostics_reports_p_bin_and_price_bin() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "p_bin": ["0.50_0.55", "0.50_0.55", "0.70_1.00"],
            "target_raw": [0.15, 0.25, 0.75],
        }
    )
    y = df["target_raw"].to_numpy(dtype=np.float32)
    p = np.array([0.17, 0.255, 0.80], dtype=np.float32)
    z = logit_np(p)
    config = {
        "diagnostics": {
            "group_columns": ["p_bin"],
            "p_side_bin_edges": [0.0, 0.5, 0.7, 1.0],
            "price_bin_column": "target_raw",
            "price_bin_edges": [0.0, 0.2, 0.4, 1.0],
        }
    }
    df["p_side"] = [0.45, 0.55, 0.75]

    diagnostics = grouped_diagnostics(df, y, z, epsilon=0.01, tolerance=1e-6, config=config)

    assert "by_p_bin" in diagnostics
    assert "by_p_side_bin" in diagnostics
    assert "by_price_bin" in diagnostics
    assert diagnostics["by_p_bin"][0]["sample_count"] == 2
    assert {row["value"] for row in diagnostics["by_p_side_bin"]} == {"0.00_0.50", "0.50_0.70", "0.70_1.00"}
    assert {row["value"] for row in diagnostics["by_price_bin"]} == {"0.00_0.20", "0.20_0.40", "0.40_1.00"}


def test_grouped_diagnostics_fills_missing_p_bin_from_p_side() -> None:
    import pandas as pd

    df = pd.DataFrame({"p_bin": [None, "0.50_0.55"], "p_side": [0.45, 0.52], "target_raw": [0.2, 0.2]})
    y = df["target_raw"].to_numpy(dtype=np.float32)
    z = logit_np(np.array([0.3, 0.3], dtype=np.float32))
    config = {"diagnostics": {"group_columns": ["p_bin"], "p_side_bin_edges": [0.0, 0.5, 1.0]}}

    diagnostics = grouped_diagnostics(df, y, z, epsilon=0.01, tolerance=1e-6, config=config)

    assert {row["value"] for row in diagnostics["by_p_bin"]} == {"0.00_0.50", "0.50_0.55"}


def test_constraint_weights_from_y_uses_price_bins_and_normalizes() -> None:
    y = np.array([0.05, 0.25, 0.85], dtype=np.float32)
    config = {
        "constraint_weighting": {
            "enabled": True,
            "normalize_mean": True,
            "price_bin_edges": [0.0, 0.2, 0.6, 1.0],
            "price_bin_weights": [0.5, 1.0, 4.0],
        }
    }

    weights = constraint_weights_from_y(y, config)

    assert np.isclose(weights.mean(), 1.0)
    assert weights[0] < weights[1] < weights[2]


def test_tightness_weights_from_y_can_prioritize_low_price_gap() -> None:
    y = np.array([0.05, 0.25, 0.85], dtype=np.float32)
    config = {
        "tightness_weighting": {
            "enabled": True,
            "normalize_mean": True,
            "price_bin_edges": [0.0, 0.2, 0.6, 1.0],
            "price_bin_weights": [4.0, 1.0, 0.5],
        }
    }

    weights = tightness_weights_from_y(y, config)

    assert np.isclose(weights.mean(), 1.0)
    assert weights[0] > weights[1] > weights[2]


def test_is_better_requires_max_violation_after_coverage() -> None:
    incumbent = {"coverage": 0.91, "mean_gap": 0.20, "max_violation": 0.35}
    candidate = {"coverage": 0.91, "mean_gap": 0.24, "max_violation": 0.25}

    assert is_better(candidate, incumbent, min_validation_coverage=0.90, max_validation_violation=0.30)
