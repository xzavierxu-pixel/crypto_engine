from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from price_estimator.safe_lowest_price_gap.train_safe_lowest_price_gap import (
    bucket_conf_ok,
    fit_bucket_model,
    infer_prices,
    metric_summary,
    normalized_asym_loss,
    safe_targets,
    split_fit_calibration,
)


def test_safe_targets_keeps_infeasible_samples_and_builds_s_eff() -> None:
    df = pd.DataFrame({"target_raw": [0.40, 0.70], "p_side": [0.60, 0.705]})
    config = {"target": {"raw_column": "target_raw", "buffer": 0.01}, "loss": {"s_floor": 0.02}}

    y_safe, s, s_eff = safe_targets(df, config)

    assert np.allclose(y_safe, [0.41, 0.71])
    assert s[0] > 0
    assert s[1] < 0
    assert np.isclose(s_eff[1], 0.02)


def test_normalized_asym_loss_penalizes_small_room_over_prediction_more() -> None:
    z = torch.logit(torch.tensor([[0.42], [0.42]], dtype=torch.float32))
    y = torch.tensor([[0.40], [0.40]], dtype=torch.float32)
    s_eff_small = torch.tensor([[0.02], [0.20]], dtype=torch.float32)

    loss_small_room = normalized_asym_loss(z[:1], y[:1], s_eff_small[:1], alpha=4.0, c=0.01, kappa=1.0)
    loss_large_room = normalized_asym_loss(z[1:], y[1:], s_eff_small[1:], alpha=4.0, c=0.01, kappa=1.0)

    assert loss_small_room.item() > loss_large_room.item()


def test_infer_prices_caps_to_p_side_and_marks_actions() -> None:
    f = np.array([0.40, 0.80, 0.30])
    p_side = np.array([0.70, 0.75, 0.60])
    conf_ok = np.array([True, True, False])

    pred = infer_prices(f, p_side, conf_ok, delta=0.01, tick_size=0.01, tick_tol=1e-9)

    assert pred.action.tolist() == ["active", "clamp_over_pside", "abstain_low_conf"]
    assert np.all(pred.p_pred <= p_side + 1e-12)
    assert pred.p_pred[2] == p_side[2]


def test_metric_summary_uses_feasible_coverage_not_overall_ceiling() -> None:
    y = np.array([0.40, 0.80])
    p_side = np.array([0.70, 0.75])
    pred = infer_prices(np.array([0.40, 0.90]), p_side, np.array([True, True]), 0.01, 0.01, 1e-9)

    metrics = metric_summary(y, p_side, pred, tolerance=1e-6)

    assert metrics["max_possible_coverage"] == 0.5
    assert metrics["coverage_feasible"] == 1.0
    assert metrics["coverage_overall"] == 0.5


def test_bucket_model_marks_high_miss_bucket_low_confidence() -> None:
    calibration = pd.DataFrame(
        {
            "p_side_bucket": ["low", "low", "high", "high"],
            "market_time_bucket": ["asia", "asia", "asia", "asia"],
            "selected_side": ["UP", "UP", "UP", "UP"],
        }
    )
    config = {
        "target": {"tick_size": 0.01, "tick_rounding_tolerance": 1e-9},
        "bucket_abstain": {
            "enabled": True,
            "keys": ["p_side_bucket", "market_time_bucket", "selected_side"],
            "min_bucket_count": 1,
        },
    }
    y = np.array([0.40, 0.41, 0.70, 0.71])
    p_side = np.array([0.80, 0.80, 0.80, 0.80])
    f = np.array([0.40, 0.41, 0.60, 0.60])

    model = fit_bucket_model(calibration, y, p_side, f, delta=0.0, config=config, tolerance=1e-6)
    ok = bucket_conf_ok(calibration, model, threshold=0.5)

    assert ok.tolist() == [True, True, False, False]


def test_split_fit_calibration_uses_train_tail_days() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC"),
            "target_raw": np.linspace(0.1, 0.9, 10),
            "p_side": np.linspace(0.2, 1.0, 10),
        }
    )
    config = {"split": {"timestamp_column": "timestamp", "calibration_tail_days": 3}}

    fit, cal = split_fit_calibration(df, config)

    assert len(fit) == 6
    assert len(cal) == 4
    assert cal["timestamp"].min() >= df["timestamp"].max() - pd.Timedelta(days=3)
