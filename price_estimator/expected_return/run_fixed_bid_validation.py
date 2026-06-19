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
            all_result, _, accepted, all_metrics, accepted_metrics = evaluate_fixed_bid_frame(frame, fixed_bid)
            if not np.all(all_result.bid == fixed_bid):
                raise AssertionError("Not every source row received the configured fixed bid")
        else:
            all_result, _, accepted, all_metrics, accepted_metrics = evaluate_pside_multiplier_bid_frame(
                frame, multiplier, tick_size
            )
            expected_bid = floor_to_tick(multiplier * frame["p_side"].to_numpy(dtype=float), tick_size)
            if not np.array_equal(all_result.bid, expected_bid):
                raise AssertionError("Not every source row received the configured p_side multiplier bid")
        if not np.all(all_result.bid > 0.0):
            raise AssertionError("Every source row must receive a positive bid")
        write_predictions(frame, all_result, resolve_path(config["paths"][prediction_key]))
        outputs[split] = {
            "all_metrics": all_metrics,
            "accepted_metrics": accepted_metrics,
            "all_window": window(frame),
            "accepted_window": window(accepted),
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
