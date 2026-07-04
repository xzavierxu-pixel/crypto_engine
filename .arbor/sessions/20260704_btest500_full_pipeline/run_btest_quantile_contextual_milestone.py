#!/usr/bin/env python3
"""Frozen B_test milestone for the B_dev-selected quantile admission policy."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

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


policy = load("quantile_admission_btest", SESSION / "run_quantile_contextual_admission.py")


def main() -> None:
    summary = json.loads((SESSION / "quantile_contextual_admission_summary.json").read_text(encoding="utf-8"))
    variant = str(summary["winner"]["variant"])
    match = re.fullmatch(r"(.+)_k([0-9.]+)", variant)
    if match is None:
        raise ValueError(f"unrecognized frozen variant: {variant}")
    score_name, keep = match.group(1), float(match.group(2))

    prepared = policy.joint.prepare(TRAIN_PATH, TEST_PATH, HAZARD_PATH, SEED)
    test, accepted, _, _, grid, _, _ = prepared
    train = pd.read_parquet(TRAIN_PATH)
    train = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    models = policy.fit_reward_models(train, grid)
    base_bid, base_ev, base_fill = policy.analytic_bid(prepared)
    scores = policy.contextual_scores(models, accepted, base_bid)
    bid, ev, fill, cutoff = policy.admitted_bid(base_bid, base_ev, base_fill, scores[score_name], keep)
    result = policy.joint.backtest_with_bid(accepted, bid, ev, fill)
    metrics = policy.joint.backtest_metrics(accepted, result, len(test))
    metrics["contextual_score_cutoff"] = cutoff

    diagnostics = accepted[["timestamp", "decision_time", "selected_side", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["base_bid"] = base_bid
    diagnostics["contextual_score"] = scores[score_name]
    diagnostics["admitted_bid"] = bid
    diagnostics["pnl"] = result.pnl
    diagnostics.to_parquet(SESSION / "btest_quantile_contextual_diagnostics.parquet", index=False)
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
        "leakage_guard": "explicit decision-time features; correct/chosen_low only construct training potential outcomes",
    }
    (SESSION / "btest_quantile_contextual_milestone_once.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
