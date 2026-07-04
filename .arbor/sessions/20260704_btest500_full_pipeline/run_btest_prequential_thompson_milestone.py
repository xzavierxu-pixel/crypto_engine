#!/usr/bin/env python3
"""Frozen online B_test milestone for settlement-safe Thompson expert selection."""
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


policy = load("prequential_thompson_btest", SESSION / "run_prequential_thompson_experts.py")


def main() -> None:
    summary = json.loads((SESSION / "prequential_thompson_experts_summary.json").read_text(encoding="utf-8"))
    variant = str(summary["winner"]["variant"])
    match = re.fullmatch(r"(global|q|qside)_p([0-9.]+)_d([0-9.]+)_t([0-9.]+)", variant)
    if match is None:
        raise ValueError(f"unrecognized frozen variant: {variant}")
    level, prior, decay, temperature = match.group(1), float(match.group(2)), float(match.group(3)), float(match.group(4))

    prepared = policy.joint.prepare(TRAIN_PATH, TEST_PATH, HAZARD_PATH, SEED)
    test, accepted = prepared[0], prepared[1]
    experts = policy.build_experts(prepared)
    metrics, diag = policy.thompson_policy(experts, level, prior, decay, temperature)
    diagnostics = accepted[["timestamp", "decision_time", "selected_side", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["expert_index"] = diag["chosen"]
    diagnostics["expert_name"] = [experts["names"][i] for i in diag["chosen"]]
    diagnostics["bid"] = diag["bid"]
    diagnostics["pnl"] = diag["pnl"]
    diagnostics.to_parquet(SESSION / "btest_prequential_thompson_diagnostics.parquet", index=False)
    payload = {
        "evaluation_kind": "single frozen online B_test milestone after B_dev-only selection",
        "selection_was_bdev_only": True, "variant": variant,
        "bdev_winner": summary["winner"], "baseline_sum_pnl": 27.44,
        "previous_best_btest_sum_pnl": 42.43, "metrics": metrics,
        "daily_choices": diag["daily_choices"],
        "btest_window": {
            "start": str(pd.to_datetime(test["timestamp"], utc=True).min()),
            "end": str(pd.to_datetime(test["timestamp"], utc=True).max()),
        },
        "btest_evaluation_count": 1,
        "settlement_guard": "expert posteriors update only after an entire UTC day is resolved",
    }
    (SESSION / "btest_prequential_thompson_milestone_once.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"variant": variant, "metrics": metrics, "btest_window": payload["btest_window"]}, indent=2))


if __name__ == "__main__":
    main()
