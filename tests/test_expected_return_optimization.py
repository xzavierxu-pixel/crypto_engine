from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parents[1] / "price_estimator" / "expected_return"
sys.path.insert(0, str(MODULE_DIR))

from build_expected_return_target import (  # noqa: E402
    apply_trades_coverage_start,
    classify_low_join,
    order_window_mask,
)
from run_fixed_bid_validation import (  # noqa: E402
    evaluate_fixed_bid_frame,
    evaluate_pside_multiplier_bid_frame,
    evaluate_pside_piecewise_bid_frame,
)
from run_empirical_pside_bin_cdf import (  # noqa: E402
    fit_empirical_pside_bin_cdf,
    gc_for_frame,
    pside_bin_index,
)
from run_h2_gc_calibration import adjust_gc, choose_best_policy, split_calibration_halves  # noqa: E402
from train_low_cdf_and_backtest import (  # noqa: E402
    backtest_metrics,
    backtest_with_bid,
    choose_expected_return_bids,
    choose_survival_expected_return_bids,
    build_tick_grid,
    event_indices,
    hazard_nll,
    select_min_ev,
    split_fit_calibration,
    survival_cdf,
)


def test_forced_wrong_fill_and_missing_correct_low() -> None:
    frame = pd.DataFrame(
        {
            "correct": [True, True, False, False],
            "chosen_low": [0.20, np.nan, np.nan, 0.30],
            "selected_side": ["UP", "DOWN", "UP", "DOWN"],
            "p_up": [0.8, 0.2, 0.8, 0.2],
            "target": [1, 0, 0, 1],
            "selected_t_up": [0.6] * 4,
            "selected_t_down": [0.4] * 4,
        }
    )
    result = backtest_with_bid(frame, np.asarray([0.25, 0.25, 0.10, 0.0]))

    assert result.filled.tolist() == [True, False, True, False]
    assert result.printed_filled.tolist() == [True, False, False, False]
    assert result.pnl.tolist() == [0.75, 0.0, -0.10, 0.0]
    metrics = backtest_metrics(frame, result)
    assert metrics["mean_accepted_pnl"] == 0.1625
    assert metrics["wrong_fill_forced"] == 1.0
    assert metrics["wrong_fill_printed"] == 0.0
    assert metrics["order_coverage"] == 0.75


def test_min_ev_abstains_without_changing_best_ev() -> None:
    p_side = np.asarray([0.70])
    f_pred = np.asarray([0.50])
    residuals = np.asarray([-0.10, 0.0, 0.10])
    bid, ev, _ = choose_expected_return_bids(p_side, f_pred, residuals, 0.01, 0.01, 1.0)

    assert bid.tolist() == [0.0]
    assert ev[0] < 1.0


def test_survival_bid_selection_strictly_filters_gc_candidates() -> None:
    q = np.array([0.8])
    grid = np.array([0.1, 0.2, 0.3])
    gc = np.array([[0.5, 0.6, 0.7]])
    bids, _, fill_prob = choose_survival_expected_return_bids(
        q, gc, grid, 0.01, float("-inf"), min_fill_probability=0.6
    )
    assert bids.tolist() == [0.3]
    assert fill_prob.tolist() == [0.7]


def test_seven_day_split_is_time_based_and_all_side() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC"),
            "threshold_accepted": [True, False] * 10,
        }
    )
    fit, calibration = split_fit_calibration(
        frame,
        {"split": {"timestamp_column": "timestamp", "calibration_tail_days": 7}},
    )

    assert len(fit) == 12
    assert len(calibration) == 8
    assert set(fit["threshold_accepted"]) == {True, False}


def test_missing_low_reason_is_independent_of_threshold() -> None:
    decision = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = pd.DataFrame(
        {
            "condition_id": ["none", "wrong_side", "before", "late", "matched"],
            "predicted_outcome": ["up"] * 5,
            "decision_time": [decision] * 5,
            "endDate": [decision + pd.Timedelta(minutes=5)] * 5,
            "threshold_accepted": [True, False, True, False, True],
        }
    )
    trades = pd.DataFrame(
        {
            "condition_id": ["wrong_side", "before", "late", "matched"],
            "outcome": ["down", "up", "up", "up"],
            "trade_time": [
                decision + pd.Timedelta(minutes=1),
                decision - pd.Timedelta(minutes=1),
                decision + pd.Timedelta(minutes=6),
                decision + pd.Timedelta(minutes=1),
            ],
        }
    )
    matched_low = pd.Series([np.nan, np.nan, np.nan, np.nan, 0.2])

    reasons = classify_low_join(rows, trades, matched_low)

    assert reasons.tolist() == [
        "no_condition_trades",
        "no_predicted_outcome_trades",
        "no_trades_after_decision",
        "no_trades_before_settlement",
        "matched",
    ]


def test_order_window_respects_inclusive_boundaries() -> None:
    decision = pd.Timestamp("2026-01-01T00:02:00Z")
    end = pd.Timestamp("2026-01-01T00:05:00Z")
    joined = pd.DataFrame(
        {
            "trade_time": [decision, decision + pd.Timedelta(seconds=1), end],
            "decision_time": [decision] * 3,
            "endDate": [end] * 3,
        }
    )

    inclusive = order_window_mask(joined, {"include_start": True, "include_end": True})
    exclusive = order_window_mask(joined, {"include_start": False, "include_end": False})

    assert inclusive.tolist() == [True, True, True]
    assert exclusive.tolist() == [False, True, False]


def test_trade_coverage_start_filters_before_target_building() -> None:
    frame = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2026-02-11T23:55:00Z", "2026-02-12T00:00:00Z", "2026-02-12T00:05:00Z"]
            ),
            "condition_id": ["old", "first", "next"],
        }
    )

    filtered, report = apply_trades_coverage_start(frame, "2026-02-12T00:00:00Z")

    assert filtered["condition_id"].tolist() == ["first", "next"]
    assert report["source_rows_before_coverage_filter"] == 3
    assert report["rows_excluded_before_trades_coverage"] == 1
    assert report["rows_after_trades_coverage_filter"] == 2


def test_survival_cdf_is_monotone_and_matches_hazard_product() -> None:
    logits = torch.zeros((2, 3))
    gc = survival_cdf(logits).numpy()

    assert np.allclose(gc[0], [0.5, 0.75, 0.875])
    assert np.all(np.diff(gc, axis=1) >= 0.0)


def test_hazard_event_index_and_masked_likelihood() -> None:
    grid = build_tick_grid(0.01, 0.03)
    index = event_indices(np.asarray([0.015, 0.04]), grid)
    logits = torch.zeros((2, 3))

    assert index.tolist() == [1, 3]
    # First row uses survival at tick 1 + event at tick 2; second is censored after all 3 ticks.
    assert np.isclose(float(hazard_nll(logits, torch.from_numpy(index))), 2.5 * np.log(2.0))


def test_survival_bid_selection_uses_sample_specific_cdf() -> None:
    grid = build_tick_grid(0.01, 0.03)
    q = np.asarray([0.7, 0.7])
    gc = np.asarray([[0.9, 0.95, 0.99], [0.01, 0.02, 0.03]])
    bid, ev, fill = choose_survival_expected_return_bids(q, gc, grid, 0.01, -1.0)

    assert bid[0] == 0.03
    assert bid[1] == 0.03
    assert ev[0] > ev[1]
    assert fill.tolist() == [0.99, 0.03]


def test_min_ev_selection_is_calibration_only_with_agreed_ties() -> None:
    frame = pd.DataFrame(
        {
            "correct": [True, False],
            "chosen_low": [0.01, np.nan],
            "selected_side": ["UP", "DOWN"],
            "p_up": [0.8, 0.2],
            "target": [1, 1],
            "selected_t_up": [0.6, 0.6],
            "selected_t_down": [0.4, 0.4],
        }
    )
    same = backtest_with_bid(frame, np.asarray([0.01, 0.01]))
    selected, rows = select_min_ev({0.0: same, 0.01: same}, frame, len(frame), 2)

    assert selected == 0.01
    assert len(rows) == 2


def test_fixed_absolute_bid_is_generated_before_accepted_filter() -> None:
    frame = pd.DataFrame(
        {
            "threshold_accepted": [True, False, True],
            "correct": [True, False, False],
            "chosen_low": [0.40, np.nan, np.nan],
            "selected_side": ["UP", "DOWN", "UP"],
            "p_up": [0.8, 0.2, 0.8],
            "target": [1, 1, 0],
            "selected_t_up": [0.6] * 3,
            "selected_t_down": [0.4] * 3,
        }
    )

    all_result, accepted_result, accepted, all_metrics, accepted_metrics = evaluate_fixed_bid_frame(frame, 0.50)

    assert all_result.bid.tolist() == [0.5, 0.5, 0.5]
    assert accepted_result.bid.tolist() == [0.5, 0.5]
    assert len(accepted) == 2
    assert all_metrics["order_coverage"] == 1.0
    assert accepted_metrics["order_coverage"] == 1.0
    assert accepted_metrics["sum_pnl"] == 0.0


def test_pside_multiplier_bid_is_ticked_before_accepted_filter() -> None:
    frame = pd.DataFrame(
        {
            "threshold_accepted": [True, False, True],
            "correct": [True, False, False],
            "chosen_low": [0.40, np.nan, np.nan],
            "selected_side": ["UP", "DOWN", "UP"],
            "p_up": [0.8, 0.2, 0.8],
            "p_side": [0.63, 0.72, 0.81],
            "target": [1, 1, 0],
            "selected_t_up": [0.6] * 3,
            "selected_t_down": [0.4] * 3,
        }
    )

    all_result, accepted_result, accepted, all_metrics, accepted_metrics = evaluate_pside_multiplier_bid_frame(
        frame, 0.90, 0.01
    )

    assert np.allclose(all_result.bid, [0.56, 0.64, 0.72])
    assert np.allclose(accepted_result.bid, [0.56, 0.72])
    assert len(accepted) == 2
    assert all_metrics["order_coverage"] == 1.0
    assert accepted_metrics["order_coverage"] == 1.0
    assert np.isclose(accepted_metrics["sum_pnl"], -0.28)


def test_pside_piecewise_bid_boundaries_and_abstention() -> None:
    frame = pd.DataFrame(
        {
            "threshold_accepted": [True] * 4,
            "correct": [True, True, True, False],
            "chosen_low": [0.01, 0.01, 0.01, np.nan],
            "selected_side": ["UP"] * 4,
            "p_up": [0.29, 0.30, 0.70, 0.71],
            "p_side": [0.29, 0.30, 0.70, 0.71],
            "target": [1, 1, 1, 0],
            "selected_t_up": [0.6] * 4,
            "selected_t_down": [0.4] * 4,
        }
    )

    all_result, accepted_result, _, _, accepted_metrics = evaluate_pside_piecewise_bid_frame(
        frame, 0.30, 0.70, 0.90, 0.01, 0.01
    )

    assert np.allclose(all_result.bid, [0.0, 0.27, 0.63, 0.70])
    assert np.array_equal(accepted_result.bid, all_result.bid)
    assert accepted_metrics["order_count"] == 3.0
    assert accepted_metrics["order_coverage"] == 0.75


def test_empirical_pside_bin_cdf_and_empty_bin_fallback() -> None:
    calibration = pd.DataFrame(
        {
            "p_side": [0.601, 0.619, 0.641],
            "chosen_low": [0.01, 0.03, 0.02],
        }
    )
    grid = np.asarray([0.01, 0.02, 0.03])

    gc_by_bin, summary = fit_empirical_pside_bin_cdf(calibration, grid, 0.02)
    frame = pd.DataFrame({"p_side": [0.60, 0.62, 0.64]})
    gc = gc_for_frame(frame, gc_by_bin, 0.02)

    assert pside_bin_index(np.asarray([0.60, 0.6199, 0.62]), 0.02).tolist() == [30, 30, 31]
    assert np.allclose(gc[0], [0.5, 0.5, 1.0])
    assert np.allclose(gc[1], gc[0])
    assert np.allclose(gc[2], [0.0, 1.0, 1.0])
    assert bool(summary.loc[summary["bin_index"] == 31, "fallback_used"].iloc[0])
    assert np.all(np.diff(gc, axis=1) >= 0.0)


def test_h2_gc_adjustments_are_monotone_and_conservative() -> None:
    h2 = np.asarray([[0.20, 0.50, 0.80], [0.10, 0.40, 0.90]])
    empirical = np.zeros((50, 3))
    empirical[30] = [0.10, 0.30, 0.70]
    upper = np.zeros((50, 3))
    upper[30] = [0.15, 0.45, 0.75]
    calibrator = {"empirical": empirical, "upper_0.9": upper}
    p_side = np.asarray([0.601, 0.619])

    blend = adjust_gc(h2, p_side, calibrator, 0.02, "blend", 0.5)
    capped = adjust_gc(h2, p_side, calibrator, 0.02, "upper_cap", 0.9)

    assert np.allclose(blend[0], [0.15, 0.40, 0.75])
    assert np.all(capped <= h2 + 1e-12)
    assert np.all(np.diff(blend, axis=1) >= 0.0)
    assert np.all(np.diff(capped, axis=1) >= 0.0)


def test_gc_policy_selection_uses_pnl_then_orders_and_parameter() -> None:
    rows = [
        {"parameter": 0.25, "min_ev": 0.0, "order_count": 100.0, "mean_accepted_pnl": 0.01},
        {"parameter": 0.50, "min_ev": 0.0, "order_count": 110.0, "mean_accepted_pnl": 0.01},
        {"parameter": 0.75, "min_ev": 0.0, "order_count": 90.0, "mean_accepted_pnl": 0.02},
    ]

    selected = choose_best_policy(rows, 100)

    assert selected["parameter"] == 0.50


def test_calibration_halves_are_chronological_and_disjoint() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"), "row": range(6)}
    )

    gc_fit, policy_select = split_calibration_halves(frame, "timestamp")

    assert gc_fit["row"].tolist() == [0, 1, 2]
    assert policy_select["row"].tolist() == [3, 4, 5]
    assert gc_fit["timestamp"].max() < policy_select["timestamp"].min()
