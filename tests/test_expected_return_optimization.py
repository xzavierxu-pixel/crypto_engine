from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parents[1] / "price_estimator" / "expected_return"
sys.path.insert(0, str(MODULE_DIR))

from build_expected_return_target import apply_trades_coverage_start, classify_low_join  # noqa: E402
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
