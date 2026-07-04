#!/usr/bin/env python3
"""Frozen B_test milestone for the hierarchical Bayesian bid policy."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet"
TEST_PATH = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet"
HAZARD_PATH = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt"
SEED = 20260704


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


policy = load("hierarchical_bayes_btest", SESSION / "run_hierarchical_bayesian_bid.py")


def main() -> None:
    summary = json.loads((SESSION / "hierarchical_bayesian_bid_summary.json").read_text(encoding="utf-8"))
    variant = str(summary["winner"]["variant"])
    match = re.fullmatch(r"h([0-9.]+)_(q|side|hour)_b([0-9.]+)_z([0-9.]+)", variant)
    if match is None:
        raise ValueError(f"unrecognized frozen variant: {variant}")
    half_life, level, blend, z = float(match.group(1)), match.group(2), float(match.group(3)), float(match.group(4))

    prepared = policy.joint.prepare(TRAIN_PATH, TEST_PATH, HAZARD_PATH, SEED)
    test, accepted, qs, gc, grid, _, _ = prepared
    train = pd.read_parquet(TRAIN_PATH)
    train = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    action_idx = np.arange(1, min(len(grid), 86))
    bids = grid[action_idx]
    q = qs["raw_tree_blend"]
    analytic = q[:, None] * gc[:, action_idx] * (1.0 - bids[None, :]) - (1.0 - q[:, None]) * bids[None, :]
    post_mean, post_std = policy.posterior_surfaces(train, accepted, bids, half_life)[level]
    score = (1.0 - blend) * analytic + blend * post_mean - z * post_std
    masked = np.where(gc[:, action_idx] >= 0.85, score, -np.inf)
    idx = np.argmax(masked, axis=1)
    value = masked[np.arange(len(accepted)), idx]
    submit = np.isfinite(value) & (value >= 0.01)
    bid = np.where(submit, bids[idx], 0.0)
    fill = np.where(submit, gc[np.arange(len(accepted)), action_idx[idx]], 0.0)
    result = policy.joint.backtest_with_bid(accepted, bid, np.where(submit, value, 0.0), fill)
    metrics = policy.joint.backtest_metrics(accepted, result, len(test))

    diagnostics = accepted[["timestamp", "decision_time", "selected_side", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["posterior_mean"] = post_mean[np.arange(len(accepted)), idx]
    diagnostics["posterior_std"] = post_std[np.arange(len(accepted)), idx]
    diagnostics["bid"] = bid
    diagnostics["pnl"] = result.pnl
    diagnostics.to_parquet(SESSION / "btest_hierarchical_bayesian_diagnostics.parquet", index=False)
    payload = {
        "evaluation_kind": "single frozen B_test milestone after B_dev-only selection",
        "selection_was_bdev_only": True, "variant": variant,
        "bdev_winner": summary["winner"], "baseline_sum_pnl": 27.44,
        "previous_best_btest_sum_pnl": 42.43, "metrics": metrics,
        "btest_window": {
            "start": str(pd.to_datetime(test["timestamp"], utc=True).min()),
            "end": str(pd.to_datetime(test["timestamp"], utc=True).max()),
        },
        "btest_evaluation_count": 1,
        "leakage_guard": "posterior cells use p_side/side/hour/bid only; correct/chosen_low are training outcomes",
    }
    (SESSION / "btest_hierarchical_bayesian_milestone_once.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
