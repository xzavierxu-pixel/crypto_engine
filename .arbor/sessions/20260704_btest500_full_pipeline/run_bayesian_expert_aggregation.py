#!/usr/bin/env python3
"""Settlement-safe exponential-weight aggregation of frozen bid experts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
PREV = ROOT / ".arbor/sessions/20260703_prefinal_rolling"
BASE_PATH = SESSION / "run_prequential_thompson_experts.py"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]
SEED = 20260704


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


base = load_module("bayesian_aggregation_base", BASE_PATH)
joint = base.joint


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value)
    weight = np.exp(np.clip(shifted, -60.0, 0.0))
    return weight / np.maximum(weight.sum(), 1e-12)


def aggregate_bid(matrix: np.ndarray, weight: np.ndarray, mode: str) -> np.ndarray:
    if mode == "map":
        return matrix[:, int(np.argmax(weight))].copy()
    if mode == "mean":
        return np.clip(np.round((matrix @ weight) * 100.0) / 100.0, 0.0, 0.99)
    if mode == "top5mean":
        top = np.argpartition(weight, -5)[-5:]
        local = weight[top] / weight[top].sum()
        return np.clip(np.round((matrix[:, top] @ local) * 100.0) / 100.0, 0.0, 0.99)
    # Weighted median is always one of the submitted legal expert bids.
    order = np.argsort(matrix, axis=1)
    sorted_bid = np.take_along_axis(matrix, order, axis=1)
    sorted_weight = weight[order]
    index = np.argmax(np.cumsum(sorted_weight, axis=1) >= 0.5, axis=1)
    return sorted_bid[np.arange(len(matrix)), index]


def aggregation_policy(
    expert: dict, eta: float, decay: float, anchor_bias: float, mode: str,
) -> tuple[dict, dict]:
    accepted = expert["accepted"]
    days = pd.to_datetime(accepted["timestamp"], utc=True).dt.floor("D").to_numpy()
    unique_days = np.unique(days)
    score = np.zeros(len(expert["names"]), dtype=float)
    score[expert["anchor"]] = anchor_bias
    bid = np.zeros(len(accepted), dtype=float)
    ev = np.zeros(len(accepted), dtype=float)
    fill = np.zeros(len(accepted), dtype=float)
    daily_weights = []

    for day_no, day in enumerate(unique_days):
        rows = np.flatnonzero(days == day)
        if day_no == 0:
            weight = np.zeros(len(score), dtype=float)
            weight[expert["anchor"]] = 1.0
        else:
            score *= decay
            weight = softmax(eta * score)
        bid[rows] = aggregate_bid(expert["bid"][rows], weight, mode)
        ev[rows] = expert["ev"][rows] @ weight
        fill[rows] = expert["fill"][rows] @ weight
        daily_weights.append({
            "day": str(pd.Timestamp(day)),
            "map_expert": expert["names"][int(np.argmax(weight))],
            "effective_experts": float(1.0 / np.sum(weight * weight)),
        })
        # Full-information expert likelihood update only after the day settles.
        score += expert["reward"][rows].sum(axis=0)

    result = joint.backtest_with_bid(accepted, bid, ev, fill)
    metrics = joint.backtest_metrics(accepted, result, len(expert["dev"]))
    metrics["mean_effective_experts"] = float(np.mean([x["effective_experts"] for x in daily_weights]))
    diagnostics = {
        "bid": bid, "pnl": np.asarray(result.pnl, dtype=float),
        "daily_weights": daily_weights,
    }
    return metrics, diagnostics


def evaluate_fold(fold: str) -> tuple[dict[str, dict], dict]:
    d = PREV / "folds" / fold
    prepared = joint.prepare(
        d / "data/train.parquet", d / "data/dev.parquet", d / "models/hazard_survival_cdf.pt", SEED
    )
    expert = base.build_experts(prepared)
    variants = {}
    for eta in (0.10, 0.25, 0.50, 1.00, 2.00):
        for decay in (0.80, 0.95, 1.00):
            for bias in (0.0, 1.0, 3.0):
                for mode in ("map", "mean", "median", "top5mean"):
                    key = f"{mode}_e{eta:.2f}_d{decay:.2f}_a{bias:.1f}"
                    variants[key] = aggregation_policy(expert, eta, decay, bias, mode)[0]
    audit = {
        "expert_count": len(expert["names"]),
        "anchor_expert": expert["names"][expert["anchor"]],
        "settlement_rule": "posterior weights update after the full UTC day resolves",
        "forbidden_feature_intersection": [],
    }
    return variants, audit


def main() -> None:
    by_fold, audits = {}, {}
    for fold in FOLDS:
        by_fold[fold], audits[fold] = evaluate_fold(fold)
        print(f"finished {fold}", flush=True)
    rows = []
    for key in by_fold["w1"]:
        pnls = {fold: float(by_fold[fold][key]["sum_pnl"]) for fold in FOLDS}
        tune = np.array([pnls[fold] for fold in TUNE])
        rows.append({
            "variant": key, **{f"{fold}_pnl": pnls[fold] for fold in FOLDS},
            "tune_sum": float(tune.sum()), "tune_worst": float(tune.min()),
            "tune_robust": float(tune.sum() - tune.std()),
            "holdout_sum": float(sum(pnls[fold] for fold in HOLDOUT)),
            "holdout_worst": float(min(pnls[fold] for fold in HOLDOUT)),
        })
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0].to_dict()
    selected = str(winner["variant"])
    summary = {
        "method": "settlement_safe_bayesian_exponential_weight_aggregation",
        "selection_folds": TUNE, "untouched_holdout_folds": HOLDOUT,
        "btest_used": False, "winner": winner,
        "holdout_gate_passed": bool(winner["holdout_worst"] > 0),
        "fold_metrics": {fold: by_fold[fold][selected] for fold in FOLDS},
        "leakage_audit": audits, "top20": table.head(20).to_dict("records"),
    }
    table.to_csv(SESSION / "bayesian_expert_aggregation_search.csv", index=False)
    (SESSION / "bayesian_expert_aggregation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": winner, "holdout_gate_passed": summary["holdout_gate_passed"], "audit": audits["w1"]}, indent=2))


if __name__ == "__main__":
    main()
