from __future__ import annotations

import numpy as np

from price_estimator.upper_bound_mlp.train_upper_bound_mlp import (
    grouped_diagnostics,
    logit_np,
    sigmoid_np,
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
            "price_bin_column": "target_raw",
            "price_bin_edges": [0.0, 0.2, 0.4, 1.0],
        }
    }

    diagnostics = grouped_diagnostics(df, y, z, epsilon=0.01, tolerance=1e-6, config=config)

    assert "by_p_bin" in diagnostics
    assert "by_price_bin" in diagnostics
    assert diagnostics["by_p_bin"][0]["sample_count"] == 2
    assert {row["value"] for row in diagnostics["by_price_bin"]} == {"0.00_0.20", "0.20_0.40", "0.40_1.00"}
