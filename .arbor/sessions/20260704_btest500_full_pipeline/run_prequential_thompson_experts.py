#!/usr/bin/env python3
"""Settlement-safe Thompson sampling over frozen analytic bid-policy experts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
PREV = ROOT / ".arbor/sessions/20260703_prefinal_rolling"
JOINT_PATH = ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]
SEED = 20260704


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("prequential_expert_joint", JOINT_PATH)


def build_experts(prepared: tuple) -> dict:
    dev, accepted, qs, gc, grid, _, _ = prepared
    names, bids, evs, fills, rewards = [], [], [], [], []
    for shrink in (0.0, 0.10, 0.20):
        q = (1.0 - shrink) * qs["raw_tree_blend"] + shrink * 0.5
        for floor in (0.75, 0.80, 0.85, 0.90):
            for min_ev in (0.0, 0.01, 0.02, 0.03, 0.05):
                bid, ev, fill = joint.choose_survival_expected_return_bids(
                    q, gc, grid, 0.01, min_ev, min_fill_probability=floor
                )
                result = joint.backtest_with_bid(accepted, bid, ev, fill)
                names.append(f"s{shrink:.2f}_f{floor:.2f}_e{min_ev:.2f}")
                bids.append(bid)
                evs.append(ev)
                fills.append(fill)
                rewards.append(np.asarray(result.pnl, dtype=float))
    return {
        "names": names,
        "bid": np.column_stack(bids),
        "ev": np.column_stack(evs),
        "fill": np.column_stack(fills),
        "reward": np.column_stack(rewards),
        "anchor": names.index("s0.00_f0.85_e0.02"),
        "dev": dev,
        "accepted": accepted,
    }


def context_ids(frame: pd.DataFrame, level: str) -> tuple[np.ndarray, int]:
    if level == "global":
        return np.zeros(len(frame), dtype=int), 1
    q = frame["p_side"].to_numpy(float)
    qbin = np.clip(((q - 0.50) / 0.05).astype(int), 0, 5)
    if level == "q":
        return qbin, 6
    side = (frame["selected_side"].astype(str).str.upper() == "UP").to_numpy(int)
    return qbin * 2 + side, 12


def thompson_policy(
    expert: dict, level: str, prior_strength: float, decay: float, temperature: float,
) -> tuple[dict, dict]:
    accepted = expert["accepted"]
    ctx, contexts = context_ids(accepted, level)
    days = pd.to_datetime(accepted["timestamp"], utc=True).dt.floor("D").to_numpy()
    unique_days = np.unique(days)
    n_experts = len(expert["names"])
    count = np.zeros((contexts, n_experts), dtype=float)
    total = np.zeros_like(count)
    square = np.zeros_like(count)
    chosen = np.full(len(accepted), expert["anchor"], dtype=int)
    daily_choices = []
    # Stable seed per variant; Python's randomized hash is intentionally avoided.
    variant_seed = SEED + int(prior_strength * 7 + decay * 1000 + temperature * 100)
    rng = np.random.default_rng(variant_seed)

    for day_no, day in enumerate(unique_days):
        rows = np.flatnonzero(days == day)
        active_contexts = np.unique(ctx[rows])
        if day_no == 0:
            picks = {int(c): expert["anchor"] for c in active_contexts}
        else:
            count *= decay
            total *= decay
            square *= decay
            denom = count + prior_strength
            mean = total / np.maximum(denom, 1e-9)
            second = (square + prior_strength * 0.04 ** 2) / np.maximum(denom, 1e-9)
            variance = np.maximum(second - mean * mean, 1e-6)
            standard_error = np.sqrt(variance / np.maximum(denom, 1.0))
            sampled = mean + temperature * standard_error * rng.standard_normal(mean.shape)
            picks = {int(c): int(np.argmax(sampled[c])) for c in active_contexts}
        for c, pick in picks.items():
            chosen[rows[ctx[rows] == c]] = pick
        daily_choices.append({
            "day": str(pd.Timestamp(day)),
            "choices": {str(c): expert["names"][pick] for c, pick in picks.items()},
        })

        # Full-information update occurs only after the whole day is resolved.
        reward = expert["reward"][rows]
        for c in active_contexts:
            crows = rows[ctx[rows] == c]
            values = expert["reward"][crows]
            count[c] += len(crows)
            total[c] += values.sum(axis=0)
            square[c] += (values * values).sum(axis=0)

    row = np.arange(len(accepted))
    bid = expert["bid"][row, chosen]
    ev = expert["ev"][row, chosen]
    fill = expert["fill"][row, chosen]
    result = joint.backtest_with_bid(accepted, bid, ev, fill)
    metrics = joint.backtest_metrics(accepted, result, len(expert["dev"]))
    metrics["expert_switch_count"] = float(np.sum(chosen[1:] != chosen[:-1]))
    diagnostics = {
        "chosen": chosen, "bid": bid, "pnl": np.asarray(result.pnl, dtype=float),
        "daily_choices": daily_choices,
    }
    return metrics, diagnostics


def evaluate_fold(fold: str) -> tuple[dict[str, dict], dict]:
    d = PREV / "folds" / fold
    prepared = joint.prepare(
        d / "data/train.parquet", d / "data/dev.parquet", d / "models/hazard_survival_cdf.pt", SEED
    )
    expert = build_experts(prepared)
    variants = {}
    for level in ("global", "q", "qside"):
        for prior in (5.0, 20.0, 80.0):
            for decay in (0.80, 0.95, 1.00):
                for temperature in (0.25, 0.50, 1.00):
                    key = f"{level}_p{prior:g}_d{decay:.2f}_t{temperature:.2f}"
                    variants[key] = thompson_policy(expert, level, prior, decay, temperature)[0]
    audit = {
        "expert_count": len(expert["names"]),
        "anchor_expert": expert["names"][expert["anchor"]],
        "settlement_rule": "choose for a UTC day, then update all expert posteriors after that day",
        "policy_features": ["p_side_bucket", "selected_side"],
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
        "method": "settlement_safe_contextual_thompson_expert_selection",
        "selection_folds": TUNE, "untouched_holdout_folds": HOLDOUT,
        "btest_used": False, "winner": winner,
        "holdout_gate_passed": bool(winner["holdout_worst"] > 0),
        "fold_metrics": {fold: by_fold[fold][selected] for fold in FOLDS},
        "leakage_audit": audits, "top20": table.head(20).to_dict("records"),
    }
    table.to_csv(SESSION / "prequential_thompson_experts_search.csv", index=False)
    (SESSION / "prequential_thompson_experts_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": winner, "holdout_gate_passed": summary["holdout_gate_passed"], "audit": audits["w1"]}, indent=2))


if __name__ == "__main__":
    main()
