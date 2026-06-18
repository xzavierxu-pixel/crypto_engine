from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from execution_engine.artifacts import SafeLowestPriceGapNumpyModel
from price_estimator.safe_lowest_price_gap.train_safe_lowest_price_gap import (
    CandidateResult,
    PredictionResult,
    bucket_conf_ok,
    candidate_key,
    fit_delta_model,
    fit_bucket_model,
    infer_prices,
    metric_summary,
    normalized_asym_loss,
    normalized_residual_diagnostics,
    per_pside_bin_delta_norm,
    pside_bin_metrics,
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


def test_normalized_asym_loss_penalizes_small_room_under_prediction_more() -> None:
    z = torch.logit(torch.tensor([[0.38], [0.38]], dtype=torch.float32))
    y = torch.tensor([[0.40], [0.40]], dtype=torch.float32)
    s_eff = torch.tensor([[0.02], [0.20]], dtype=torch.float32)

    loss_small_room = normalized_asym_loss(z[:1], y[:1], s_eff[:1], alpha=1.0, c=0.01, kappa=1.0)
    loss_large_room = normalized_asym_loss(z[1:], y[1:], s_eff[1:], alpha=1.0, c=0.01, kappa=1.0)

    assert loss_small_room.item() > loss_large_room.item()


def test_infer_prices_caps_to_p_side_and_marks_actions() -> None:
    f = np.array([0.40, 0.80, 0.30])
    p_side = np.array([0.70, 0.75, 0.60])
    conf_ok = np.array([True, True, False])

    pred = infer_prices(f, p_side, conf_ok, delta_norm=0.50, tick_size=0.01, tick_tol=1e-9, s_floor=0.02)

    assert pred.action.tolist() == ["active", "clamp_over_pside", "abstain_low_conf"]
    assert np.all(pred.p_pred <= p_side + 1e-12)
    assert pred.p_pred[2] == p_side[2]
    assert np.isclose(pred.p_pred[0], 0.55)


def test_metric_summary_uses_feasible_coverage_not_overall_ceiling() -> None:
    y = np.array([0.40, 0.80])
    p_side = np.array([0.70, 0.75])
    pred = infer_prices(
        np.array([0.40, 0.90]),
        p_side,
        np.array([True, True]),
        delta_norm=0.10,
        tick_size=0.01,
        tick_tol=1e-9,
        s_floor=0.02,
    )

    metrics = metric_summary(y, p_side, pred, tolerance=1e-6)

    assert metrics["max_possible_coverage"] == 0.5
    assert metrics["coverage_feasible"] == 1.0
    assert metrics["coverage_overall"] == 0.5
    assert metrics["active_covered_gap_norm_mean"] == metrics["covered_gap_norm_mean"]
    assert metrics["non_active_share"] == 0.5


def test_candidate_key_uses_configured_active_metric_before_full_gap() -> None:
    def candidate(active_gap: float, full_gap: float) -> CandidateResult:
        return CandidateResult(
            alpha=1.0,
            delta_norm=0.5,
            delta_quantile=0.7,
            bucket_miss_threshold=0.1,
            bucket_model={},
            metrics={
                "coverage_feasible": 0.72,
                "side_violation_rate": 0.0,
                "non_active_share": 0.2,
                "active_covered_gap_norm_mean": active_gap,
                "covered_gap_norm_mean": full_gap,
                "covered_feasible_count": 100.0,
            },
            prediction=PredictionResult(
                p_pred=np.array([], dtype=float),
                action=np.array([], dtype=str),
                conf_ok=np.array([], dtype=bool),
            ),
        )

    better_active = candidate(active_gap=0.50, full_gap=0.80)
    better_full = candidate(active_gap=0.60, full_gap=0.40)

    assert candidate_key(
        better_active,
        min_coverage=0.70,
        max_non_active_share=0.45,
        optimize_metric="active_covered_gap_norm_mean",
        tie_breaker_metric="covered_gap_norm_mean",
    ) < candidate_key(
        better_full,
        min_coverage=0.70,
        max_non_active_share=0.45,
        optimize_metric="active_covered_gap_norm_mean",
        tie_breaker_metric="covered_gap_norm_mean",
    )


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
        "loss": {"s_floor": 0.02},
        "bucket_abstain": {
            "enabled": True,
            "keys": ["p_side_bucket", "market_time_bucket", "selected_side"],
            "min_bucket_count": 1,
        },
    }
    y = np.array([0.40, 0.41, 0.70, 0.71])
    p_side = np.array([0.80, 0.80, 0.80, 0.80])
    f = np.array([0.40, 0.41, 0.60, 0.60])

    delta_model = fit_delta_model(y, p_side, f, 0.5, 0.0, config)
    model = fit_bucket_model(calibration, y, p_side, f, delta_norm=0.0, delta_model=delta_model, config=config, tolerance=1e-6)
    ok = bucket_conf_ok(calibration, model, threshold=0.5)

    assert ok.tolist() == [True, True, False, False]


def test_pside_bin_delta_model_uses_local_delta_with_global_fallback() -> None:
    config = {
        "loss": {"s_floor": 0.02},
        "calibration": {"delta_mode": "pside_bin", "delta_min_bucket_count": 2},
        "diagnostics": {"p_side_bin_edges": [0.0, 0.5, 1.0]},
    }
    y = np.array([0.40, 0.42, 0.75])
    p_side = np.array([0.45, 0.46, 0.90])
    f = np.array([0.30, 0.32, 0.70])

    model = fit_delta_model(y, p_side, f, delta_quantile=0.5, delta_norm=0.25, config=config)

    assert model["mode"] == "pside_bin"
    assert np.isclose(model["table"][0]["delta_norm"], 0.6904761904761905)
    assert model["table"][1]["delta_norm"] == 0.25
    assert model["table"][1]["fallback_used"] == 1.0


def test_residual_diagnostics_and_pside_bins_are_structured() -> None:
    y = np.array([0.40, 0.45, 0.70])
    p_side = np.array([0.60, 0.80, 0.75])
    f = np.array([0.38, 0.46, 0.68])
    pred = infer_prices(
        f,
        p_side,
        np.array([True, True, True]),
        delta_norm=0.50,
        tick_size=0.01,
        tick_tol=1e-9,
        s_floor=0.02,
    )

    diag = normalized_residual_diagnostics(y, p_side, f, s_floor=0.02, quantiles=[0.50, 0.90])
    delta_rows = per_pside_bin_delta_norm(y, p_side, f, [0.0, 0.7, 1.0], 0.02, 0.90)
    metric_rows = pside_bin_metrics(y, p_side, pred, [0.0, 0.7, 1.0], tolerance=1e-6)

    assert diag["feasible_count"] == 3.0
    assert "spread_q90_q50" in diag["norm_residual_proxy_s"]
    assert delta_rows[0]["feasible_count"] == 1.0
    assert "local_delta_norm" in delta_rows[1]
    assert metric_rows[0]["sample_count"] == 1.0
    assert "active_covered_gap_norm_mean" in metric_rows[1]


def test_execution_numpy_safe_gap_uses_delta_norm_scaled_room() -> None:
    model = SafeLowestPriceGapNumpyModel.__new__(SafeLowestPriceGapNumpyModel)
    model.delta_norm = 0.50
    model.delta_model = {"mode": "global", "fallback_delta_norm": 0.50, "table": []}
    model.tick_size = 0.01
    model.tick_rounding_tolerance = 1e-9
    model.s_floor = 0.02
    f = np.array([0.40, 0.80])
    p_side = np.array([0.70, 0.75])
    conf_ok = np.array([True, True])

    p_pred, action = model._infer_prices(f, p_side, conf_ok)

    assert np.allclose(p_pred, [0.55, 0.75])
    assert action.tolist() == ["active", "clamp_over_pside"]


def test_execution_numpy_safe_gap_can_use_pside_bin_delta_model() -> None:
    model = SafeLowestPriceGapNumpyModel.__new__(SafeLowestPriceGapNumpyModel)
    model.delta_norm = 0.50
    model.delta_model = {
        "mode": "pside_bin",
        "fallback_delta_norm": 0.50,
        "edges": [0.0, 0.5, 1.0],
        "table": [
            {"bucket_index": 1.0, "delta_norm": 0.25},
            {"bucket_index": 2.0, "delta_norm": 0.75},
        ],
    }
    model.tick_size = 0.01
    model.tick_rounding_tolerance = 1e-9
    model.s_floor = 0.02
    f = np.array([0.20, 0.40])
    p_side = np.array([0.40, 0.80])
    conf_ok = np.array([True, True])

    p_pred, action = model._infer_prices(f, p_side, conf_ok)

    assert np.allclose(p_pred, [0.25, 0.70])
    assert action.tolist() == ["active", "active"]


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
