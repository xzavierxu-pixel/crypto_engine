#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))

from price_estimator_common import load_config, resolve_path  # noqa: E402

from expected_return_common import git_commit, write_json  # noqa: E402
from train_low_cdf_and_backtest import backtest_metrics, backtest_with_bid, write_predictions  # noqa: E402


def window(frame: pd.DataFrame) -> dict[str, object]:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    return {"row_count": len(frame), "start": str(timestamp.min()), "end": str(timestamp.max())}


def add_policy_columns(frame: pd.DataFrame, p_side_bins: list[float]) -> pd.DataFrame:
    out = frame.copy()
    timestamp = pd.to_datetime(out["timestamp"], utc=True)
    out["date"] = timestamp.dt.strftime("%Y-%m-%d")
    out["day"] = timestamp.dt.day.astype("int64")
    out["hour"] = timestamp.dt.hour.astype("int64")
    out["minute"] = timestamp.dt.minute.astype("int64")
    out["dow"] = timestamp.dt.dayofweek.astype("int64")
    out["session"] = pd.cut(
        timestamp.dt.hour,
        bins=[-1, 7, 15, 23],
        labels=["asia", "europe", "us"],
        include_lowest=True,
    ).astype("string")
    out["p_side_bin"] = pd.cut(
        pd.to_numeric(out["p_side"], errors="coerce"),
        bins=p_side_bins,
        right=False,
        include_lowest=True,
    ).astype("string")
    return out


def realized_pnl_for_bid(frame: pd.DataFrame, bid: float) -> tuple[float, int]:
    low = pd.to_numeric(frame["chosen_low"], errors="coerce").to_numpy(dtype=float)
    correct = frame["correct"].astype(bool).to_numpy()
    filled = (correct & np.isfinite(low) & (low <= bid + 1e-12)) | ((~correct) & (bid > 0.0))
    pnl = np.zeros(len(frame), dtype=float)
    pnl[filled & correct] = 1.0 - bid
    pnl[filled & (~correct)] = -bid
    return float(pnl.sum()), int(filled.sum())


def group_key(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    return frame[list(columns)].astype("string").agg("|".join, axis=1)


def fit_group_policy(
    frame: pd.DataFrame,
    group_columns: list[str],
    bid_grid: np.ndarray,
    min_group_count: int,
    min_group_pnl: float,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    policies: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    keys = group_key(frame, group_columns)
    for name, group in frame.groupby(keys, observed=True):
        if len(group) < min_group_count:
            continue
        best_pnl = float("-inf")
        best_bid = 0.0
        best_trade_count = 0
        for bid in bid_grid:
            pnl, trade_count = realized_pnl_for_bid(group, float(bid))
            if pnl > best_pnl:
                best_pnl = pnl
                best_bid = float(bid)
                best_trade_count = trade_count
        row = {
            "group_key": name,
            "row_count": int(len(group)),
            "selected_bid": best_bid,
            "sum_pnl": best_pnl,
            "trade_count": int(best_trade_count),
            "enabled": bool(best_pnl >= min_group_pnl),
        }
        rows.append(row)
        if best_pnl >= min_group_pnl:
            policies[str(name)] = row
    return policies, rows


def apply_group_policy(frame: pd.DataFrame, group_columns: list[str], policies: dict[str, dict[str, float]]) -> np.ndarray:
    bids = np.zeros(len(frame), dtype=float)
    keys = group_key(frame, group_columns)
    for name, policy in policies.items():
        bids[keys.eq(name).to_numpy()] = float(policy["selected_bid"])
    return bids


def score_policy(
    frame: pd.DataFrame,
    group_columns: list[str],
    bid_grid: np.ndarray,
    min_group_count: int,
    min_group_pnl: float,
) -> tuple[float, dict[str, dict[str, float]], list[dict[str, object]]]:
    policies, rows = fit_group_policy(frame, group_columns, bid_grid, min_group_count, min_group_pnl)
    bids = apply_group_policy(frame, group_columns, policies)
    result = backtest_with_bid(frame, bids)
    return float(result.pnl.sum()), policies, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")

    train_all = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    validation_all = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    p_side_bins = [float(v) for v in config["policy_search"]["p_side_bins"]]
    train_all = add_policy_columns(train_all, p_side_bins)
    validation_all = add_policy_columns(validation_all, p_side_bins)
    train = train_all.loc[train_all["threshold_accepted"].astype(bool)].copy()
    validation = validation_all.loc[validation_all["threshold_accepted"].astype(bool)].copy()

    bid_grid = np.round(
        np.arange(
            float(config["policy_search"]["bid_min"]),
            float(config["policy_search"]["bid_max"]) + 1e-12,
            float(config["policy_search"]["bid_step"]),
        ),
        10,
    )
    min_group_count = int(config["policy_search"]["min_group_count"])
    min_group_pnl = float(config["policy_search"]["min_group_pnl"])
    candidates = [list(value) for value in config["policy_search"]["group_candidates"]]
    selection_source = str(config["policy_search"].get("selection_source", "validation"))
    selector_frame = validation if selection_source == "validation" else train

    search_rows: list[dict[str, object]] = []
    selected: tuple[float, list[str], dict[str, dict[str, float]], list[dict[str, object]]] | None = None
    for columns in candidates:
        score, policies, rows = score_policy(selector_frame, columns, bid_grid, min_group_count, min_group_pnl)
        search_rows.append(
            {
                "group_columns": ",".join(columns),
                "selection_source": selection_source,
                "sum_pnl": score,
                "enabled_group_count": len(policies),
                "searched_group_count": len(rows),
            }
        )
        if selected is None or score > selected[0]:
            selected = (score, columns, policies, rows)
    if selected is None:
        raise ValueError("No policy candidate was evaluated")

    _, selected_columns, selected_policies, selected_group_rows = selected
    train_bid = apply_group_policy(train, selected_columns, selected_policies)
    validation_bid = apply_group_policy(validation, selected_columns, selected_policies)
    train_result = backtest_with_bid(train, train_bid)
    validation_result = backtest_with_bid(validation, validation_bid)
    validation_metrics = backtest_metrics(validation, validation_result, len(validation_all))

    pd.DataFrame(search_rows).to_csv(reports_dir / "group_policy_search.csv", index=False)
    pd.DataFrame(selected_group_rows).to_csv(reports_dir / "selected_group_policy.csv", index=False)
    write_predictions(
        validation,
        validation_result,
        reports_dir / "predictions_validation.parquet",
        validation["p_side"].to_numpy(dtype=float),
    )
    metrics = {
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "summary_metrics.json"),
        "primary_metric": "validation sum_pnl for expected_return group bid policy",
        "objective": config["objective"],
        "baseline": config["baseline"],
        "policy": {
            "type": "validation_tuned_group_fixed_bid",
            "selection_source": selection_source,
            "selected_group_columns": selected_columns,
            "min_group_count": min_group_count,
            "min_group_pnl": min_group_pnl,
            "bid_min": float(config["policy_search"]["bid_min"]),
            "bid_max": float(config["policy_search"]["bid_max"]),
            "bid_step": float(config["policy_search"]["bid_step"]),
            "enabled_group_count": len(selected_policies),
        },
        "validation_optimism_note": (
            "This experiment intentionally tunes the expected_return order policy on validation. "
            "Use as a validation objective search result, not as independent holdout evidence."
        ),
        "train_metrics": backtest_metrics(train, train_result, len(train_all)),
        "train_window": window(train_all),
        "validation_metrics": validation_metrics,
        "validation_window": window(validation_all),
        "signal_coverage": validation_metrics["coverage"],
        "coverage_constraint_satisfied": bool(validation_metrics["coverage"] >= float(config["objective"]["min_coverage"])),
        "deploy_training_mode": config["metadata"]["deploy_training_mode"],
        "offline_validation_metric_source": config["metadata"]["offline_validation_metric_source"],
        "artifacts": {
            "group_policy_search": str(reports_dir / "group_policy_search.csv"),
            "selected_group_policy": str(reports_dir / "selected_group_policy.csv"),
            "validation_predictions": str(reports_dir / "predictions_validation.parquet"),
        },
    }
    write_json(reports_dir / "summary_metrics.json", metrics)


if __name__ == "__main__":
    main()
