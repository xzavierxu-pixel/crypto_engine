#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))

from price_estimator_common import load_config, resolve_path  # noqa: E402

from expected_return_common import git_commit, write_json  # noqa: E402
from train_low_cdf_and_backtest import (  # noqa: E402
    BacktestResult,
    backtest_metrics,
    backtest_with_bid,
    floor_to_tick,
)


def subset_result(result: BacktestResult, mask: np.ndarray) -> BacktestResult:
    return BacktestResult(
        bid=result.bid[mask],
        expected_ev=result.expected_ev[mask],
        fill_prob=result.fill_prob[mask],
        pnl=result.pnl[mask],
        filled=result.filled[mask],
        printed_filled=result.printed_filled[mask],
    )


def evaluate_fixed_bid_frame(
    frame: pd.DataFrame,
    fixed_bid: float,
) -> tuple[BacktestResult, BacktestResult, pd.DataFrame, dict[str, float], dict[str, float]]:
    if not 0.0 < fixed_bid < 1.0:
        raise ValueError("fixed_bid must be strictly between 0 and 1")
    if "threshold_accepted" not in frame.columns:
        raise ValueError("validation frame is missing threshold_accepted")
    bid = np.full(len(frame), fixed_bid, dtype=float)
    all_result = backtest_with_bid(frame, bid)
    accepted_mask = frame["threshold_accepted"].astype(bool).to_numpy()
    accepted = frame.loc[accepted_mask].copy()
    accepted_result = subset_result(all_result, accepted_mask)
    all_metrics = backtest_metrics(frame, all_result, len(frame))
    accepted_metrics = backtest_metrics(accepted, accepted_result, len(frame))
    return all_result, accepted_result, accepted, all_metrics, accepted_metrics


def evaluate_pside_multiplier_bid_frame(
    frame: pd.DataFrame,
    multiplier: float,
    tick_size: float,
) -> tuple[BacktestResult, BacktestResult, pd.DataFrame, dict[str, float], dict[str, float]]:
    if not 0.0 < multiplier <= 1.0:
        raise ValueError("p_side_multiplier must be in (0, 1]")
    if tick_size <= 0.0:
        raise ValueError("tick_size must be positive")
    if "p_side" not in frame.columns:
        raise ValueError("validation frame is missing p_side")
    if "threshold_accepted" not in frame.columns:
        raise ValueError("validation frame is missing threshold_accepted")
    p_side = pd.to_numeric(frame["p_side"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(p_side).all() or np.any((p_side <= 0.0) | (p_side >= 1.0)):
        raise ValueError("p_side must contain finite probabilities strictly between 0 and 1")
    bid = floor_to_tick(multiplier * p_side, tick_size)
    all_result = backtest_with_bid(frame, bid)
    accepted_mask = frame["threshold_accepted"].astype(bool).to_numpy()
    accepted = frame.loc[accepted_mask].copy()
    accepted_result = subset_result(all_result, accepted_mask)
    all_metrics = backtest_metrics(frame, all_result, len(frame))
    accepted_metrics = backtest_metrics(accepted, accepted_result, len(frame))
    return all_result, accepted_result, accepted, all_metrics, accepted_metrics


def evaluate_pside_piecewise_bid_frame(
    frame: pd.DataFrame,
    no_order_below: float,
    multiplier_until: float,
    middle_multiplier: float,
    high_offset: float,
    tick_size: float,
) -> tuple[BacktestResult, BacktestResult, pd.DataFrame, dict[str, float], dict[str, float]]:
    if not 0.0 <= no_order_below < multiplier_until < 1.0:
        raise ValueError("piecewise boundaries must satisfy 0 <= no_order_below < multiplier_until < 1")
    if not 0.0 < middle_multiplier <= 1.0:
        raise ValueError("middle_multiplier must be in (0, 1]")
    if not 0.0 <= high_offset < 1.0:
        raise ValueError("high_offset must be in [0, 1)")
    if tick_size <= 0.0:
        raise ValueError("tick_size must be positive")
    if "p_side" not in frame.columns:
        raise ValueError("validation frame is missing p_side")
    if "threshold_accepted" not in frame.columns:
        raise ValueError("validation frame is missing threshold_accepted")
    p_side = pd.to_numeric(frame["p_side"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(p_side).all() or np.any((p_side <= 0.0) | (p_side >= 1.0)):
        raise ValueError("p_side must contain finite probabilities strictly between 0 and 1")
    raw_bid = np.where(
        p_side < no_order_below,
        0.0,
        np.where(p_side <= multiplier_until, middle_multiplier * p_side, p_side - high_offset),
    )
    bid = floor_to_tick(np.maximum(raw_bid, 0.0), tick_size)
    all_result = backtest_with_bid(frame, bid)
    accepted_mask = frame["threshold_accepted"].astype(bool).to_numpy()
    accepted = frame.loc[accepted_mask].copy()
    accepted_result = subset_result(all_result, accepted_mask)
    all_metrics = backtest_metrics(frame, all_result, len(frame))
    accepted_metrics = backtest_metrics(accepted, accepted_result, len(frame))
    return all_result, accepted_result, accepted, all_metrics, accepted_metrics


def write_predictions(frame: pd.DataFrame, result: BacktestResult, path: Path) -> None:
    columns = [
        "timestamp",
        "decision_time",
        "condition_id",
        "polymarket_slug",
        "selected_side",
        "p_up",
        "p_side",
        "selected_t_up",
        "selected_t_down",
        "threshold_accepted",
        "target",
        "correct",
        "chosen_low",
    ]
    out = frame[[column for column in columns if column in frame.columns]].copy()
    out["bid"] = result.bid
    out["submitted"] = result.bid > 0.0
    out["filled"] = result.filled
    out["printed_filled"] = result.printed_filled
    out["realized_pnl"] = result.pnl
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def window(frame: pd.DataFrame) -> dict[str, Any]:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamp.min()), "end": str(timestamp.max())}


def bid_policy_diagnostics(
    frame: pd.DataFrame,
    result: BacktestResult,
    bid_policy: str,
    bid_parameters: dict[str, float],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "row_count": int(len(frame)),
        "submitted_count": int((result.bid > 0.0).sum()),
        "no_order_count": int((result.bid <= 0.0).sum()),
    }
    if bid_policy == "p_side_piecewise":
        p_side = frame["p_side"].to_numpy(dtype=float)
        lower = bid_parameters["no_order_below"]
        upper = bid_parameters["multiplier_until"]
        diagnostics.update(
            {
                "below_no_order_boundary_count": int((p_side < lower).sum()),
                "middle_multiplier_count": int(((p_side >= lower) & (p_side <= upper)).sum()),
                "high_offset_count": int((p_side > upper).sum()),
                "p_side_min": float(p_side.min()),
                "p_side_max": float(p_side.max()),
            }
        )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    backtest_config = config["backtest"]
    bid_policy = str(backtest_config.get("bid_policy", "fixed_absolute"))
    if bid_policy == "fixed_absolute":
        fixed_bid = float(backtest_config["fixed_bid"])
        bid_parameters: dict[str, float] = {"fixed_bid": fixed_bid}
        primary_metric = f"validation accepted-sample forced mean_accepted_pnl at absolute bid {fixed_bid:.2f}"
    elif bid_policy == "p_side_multiplier":
        multiplier = float(backtest_config["p_side_multiplier"])
        tick_size = float(backtest_config["tick_size"])
        bid_parameters = {"p_side_multiplier": multiplier, "tick_size": tick_size}
        primary_metric = (
            "validation accepted-sample forced mean_accepted_pnl at "
            f"floor_to_tick({multiplier:.6g} * p_side)"
        )
    elif bid_policy == "p_side_piecewise":
        no_order_below = float(backtest_config["no_order_below"])
        multiplier_until = float(backtest_config["multiplier_until"])
        middle_multiplier = float(backtest_config["middle_multiplier"])
        high_offset = float(backtest_config["high_offset"])
        tick_size = float(backtest_config["tick_size"])
        bid_parameters = {
            "no_order_below": no_order_below,
            "multiplier_until": multiplier_until,
            "middle_multiplier": middle_multiplier,
            "high_offset": high_offset,
            "tick_size": tick_size,
        }
        primary_metric = "validation accepted-sample forced mean_accepted_pnl for p_side piecewise bid"
    else:
        raise ValueError(f"Unsupported bid_policy: {bid_policy}")
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")

    outputs: dict[str, Any] = {}
    for split, source_key, prediction_key in [
        ("train", "train_dataset", "predictions_train"),
        ("validation", "validation_dataset", "predictions_validation"),
    ]:
        frame = pd.read_parquet(resolve_path(config["paths"][source_key]))
        if bid_policy == "fixed_absolute":
            all_result, accepted_result, accepted, all_metrics, accepted_metrics = evaluate_fixed_bid_frame(
                frame, fixed_bid
            )
            if not np.all(all_result.bid == fixed_bid):
                raise AssertionError("Not every source row received the configured fixed bid")
        elif bid_policy == "p_side_multiplier":
            all_result, accepted_result, accepted, all_metrics, accepted_metrics = evaluate_pside_multiplier_bid_frame(
                frame, multiplier, tick_size
            )
            expected_bid = floor_to_tick(multiplier * frame["p_side"].to_numpy(dtype=float), tick_size)
            if not np.array_equal(all_result.bid, expected_bid):
                raise AssertionError("Not every source row received the configured p_side multiplier bid")
        else:
            all_result, accepted_result, accepted, all_metrics, accepted_metrics = evaluate_pside_piecewise_bid_frame(
                frame,
                no_order_below,
                multiplier_until,
                middle_multiplier,
                high_offset,
                tick_size,
            )
        write_predictions(frame, all_result, resolve_path(config["paths"][prediction_key]))
        outputs[split] = {
            "all_metrics": all_metrics,
            "accepted_metrics": accepted_metrics,
            "all_window": window(frame),
            "accepted_window": window(accepted),
            "all_bid_diagnostics": bid_policy_diagnostics(frame, all_result, bid_policy, bid_parameters),
            "accepted_bid_diagnostics": bid_policy_diagnostics(
                accepted, accepted_result, bid_policy, bid_parameters
            ),
        }

    min_coverage = float(config.get("objective", {}).get("min_coverage", 0.70))
    report = {
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit(),
        "git_commit_at_evaluation": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "summary_metrics.json"),
        "primary_metric": primary_metric,
        "bid_policy": bid_policy,
        "bid_parameters": bid_parameters,
        "bid_scope": "all_source_rows_before_accepted_filter",
        "pnl_scope": "threshold_accepted_rows",
        "wrong_fill_policy": "forced",
        "train_metrics": outputs["train"]["accepted_metrics"],
        "train_all_validation_diagnostic": outputs["train"]["all_metrics"],
        "train_window": outputs["train"]["accepted_window"],
        "validation_metrics": outputs["validation"]["accepted_metrics"],
        "validation_all_samples_diagnostic": outputs["validation"]["all_metrics"],
        "train_bid_policy_diagnostics": outputs["train"]["accepted_bid_diagnostics"],
        "validation_bid_policy_diagnostics": outputs["validation"]["accepted_bid_diagnostics"],
        "validation_all_samples_bid_policy_diagnostics": outputs["validation"]["all_bid_diagnostics"],
        "validation_window": outputs["validation"]["accepted_window"],
        "signal_coverage": outputs["validation"]["accepted_metrics"]["coverage"],
        "coverage_constraint_satisfied": bool(
            outputs["validation"]["accepted_metrics"]["coverage"] >= min_coverage
        ),
        "deploy_training_mode": "not_applicable_fixed_bid_replay",
        "offline_validation_metric_source": config["paths"].get("validation_dataset"),
        "artifacts": {
            "config_snapshot": str(reports_dir / "config_used.yaml"),
            "train_predictions": str(resolve_path(config["paths"]["predictions_train"])),
            "validation_predictions": str(resolve_path(config["paths"]["predictions_validation"])),
        },
    }
    write_json(reports_dir / "summary_metrics.json", report)
    print({"validation_metrics": report["validation_metrics"], "bid_policy": bid_policy, **bid_parameters})


if __name__ == "__main__":
    main()
