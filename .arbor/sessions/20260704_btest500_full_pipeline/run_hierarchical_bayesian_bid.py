#!/usr/bin/env python3
"""Hierarchical empirical-Bayes posterior policy over legal bid rewards."""
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


joint = load_module("hierarchical_bayes_joint", JOINT_PATH)


def action_rewards(frame: pd.DataFrame, bids: np.ndarray) -> np.ndarray:
    correct = frame["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(frame["chosen_low"], errors="coerce").fillna(1.0).to_numpy(float)
    return np.where(
        correct[:, None],
        np.where(low[:, None] <= bids[None, :], 1.0 - bids[None, :], 0.0),
        -bids[None, :],
    )


def group_stats(group: np.ndarray, groups: int, reward: np.ndarray, weight: np.ndarray):
    count = np.bincount(group, weights=weight, minlength=groups).astype(float)
    total = np.column_stack([
        np.bincount(group, weights=weight * reward[:, j], minlength=groups)
        for j in range(reward.shape[1])
    ])
    square = np.column_stack([
        np.bincount(group, weights=weight * reward[:, j] ** 2, minlength=groups)
        for j in range(reward.shape[1])
    ])
    return count, total, square


def update_posterior(
    count: np.ndarray, total: np.ndarray, square: np.ndarray,
    parent_mean: np.ndarray, parent_var: np.ndarray, kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    denom = count[:, None] + kappa
    mean = (total + kappa * parent_mean) / np.maximum(denom, 1e-9)
    second = (square + kappa * (parent_var + parent_mean * parent_mean)) / np.maximum(denom, 1e-9)
    variance = np.maximum(second - mean * mean, 1e-6)
    mean_std = np.sqrt(variance / np.maximum(denom, 1.0))
    return mean, mean_std


def context_cells(train: pd.DataFrame, dev: pd.DataFrame):
    q_train = train["p_side"].to_numpy(float)
    edges = np.unique(np.quantile(q_train, [0.2, 0.4, 0.6, 0.8]))
    qtr = np.searchsorted(edges, q_train, side="right")
    qdv = np.searchsorted(edges, dev["p_side"].to_numpy(float), side="right")
    bins = len(edges) + 1
    str_ = (train["selected_side"].astype(str).str.upper() == "UP").to_numpy(int)
    sdv = (dev["selected_side"].astype(str).str.upper() == "UP").to_numpy(int)
    htr = pd.to_datetime(train["timestamp"], utc=True).dt.hour.to_numpy() // 6
    hdv = pd.to_datetime(dev["timestamp"], utc=True).dt.hour.to_numpy() // 6
    return {
        "q": (qtr, qdv, bins),
        "side": (qtr * 2 + str_, qdv * 2 + sdv, bins * 2),
        "hour": ((qtr * 2 + str_) * 4 + htr, (qdv * 2 + sdv) * 4 + hdv, bins * 8),
    }


def posterior_surfaces(
    train: pd.DataFrame, dev: pd.DataFrame, bids: np.ndarray, half_life: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    reward = action_rewards(train, bids)
    ts = pd.to_datetime(train["timestamp"], utc=True)
    age_days = (ts.max() - ts).dt.total_seconds().to_numpy() / 86400.0
    weight = np.exp2(-age_days / half_life) if half_life < 1000 else np.ones(len(train))
    cells = context_cells(train, dev)

    global_count = float(weight.sum())
    global_total = np.sum(weight[:, None] * reward, axis=0)
    global_square = np.sum(weight[:, None] * reward * reward, axis=0)
    global_mean = global_total / max(global_count, 1.0)
    global_var = np.maximum(global_square / max(global_count, 1.0) - global_mean * global_mean, 1e-6)
    surfaces = {}

    qtr, qdv, qgroups = cells["q"]
    qcount, total, square = group_stats(qtr, qgroups, reward, weight)
    qmean, qstd = update_posterior(
        qcount, total, square,
        np.broadcast_to(global_mean, (qgroups, len(bids))),
        np.broadcast_to(global_var, (qgroups, len(bids))), 50.0,
    )
    surfaces["q"] = qmean[qdv], qstd[qdv]

    str_, sdv, sgroups = cells["side"]
    scount, total, square = group_stats(str_, sgroups, reward, weight)
    parent_q = np.arange(sgroups) // 2
    qvar = qstd ** 2 * (qcount[:, None] + 50.0)
    smean, sstd = update_posterior(scount, total, square, qmean[parent_q], qvar[parent_q], 25.0)
    surfaces["side"] = smean[sdv], sstd[sdv]

    htr, hdv, hgroups = cells["hour"]
    hcount, total, square = group_stats(htr, hgroups, reward, weight)
    parent_side = np.arange(hgroups) // 4
    svar = sstd ** 2 * (scount[:, None] + 25.0)
    hmean, hstd = update_posterior(hcount, total, square, smean[parent_side], svar[parent_side], 12.0)
    surfaces["hour"] = hmean[hdv], hstd[hdv]
    return surfaces


def evaluate_fold(fold: str) -> tuple[dict[str, dict], dict]:
    d = PREV / "folds" / fold
    prepared = joint.prepare(
        d / "data/train.parquet", d / "data/dev.parquet", d / "models/hazard_survival_cdf.pt", SEED
    )
    dev, accepted, qs, gc, grid, _, _ = prepared
    train = pd.read_parquet(d / "data/train.parquet")
    train = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    action_idx = np.arange(1, min(len(grid), 86))
    bids = grid[action_idx]
    q = qs["raw_tree_blend"]
    analytic = q[:, None] * gc[:, action_idx] * (1.0 - bids[None, :]) - (1.0 - q[:, None]) * bids[None, :]
    legal = gc[:, action_idx] >= 0.85
    variants = {}
    for half_life in (14.0, 30.0, 60.0, 9999.0):
        surfaces = posterior_surfaces(train, accepted, bids, half_life)
        for level, (post_mean, post_std) in surfaces.items():
            for blend in (0.25, 0.50, 0.75, 1.00):
                for z in (0.0, 0.5, 1.0, 1.5):
                    score = (1.0 - blend) * analytic + blend * post_mean - z * post_std
                    masked = np.where(legal, score, -np.inf)
                    idx = np.argmax(masked, axis=1)
                    value = masked[np.arange(len(accepted)), idx]
                    submit = np.isfinite(value) & (value >= 0.01)
                    bid = np.where(submit, bids[idx], 0.0)
                    fill = np.where(submit, gc[np.arange(len(accepted)), action_idx[idx]], 0.0)
                    result = joint.backtest_with_bid(accepted, bid, np.where(submit, value, 0.0), fill)
                    key = f"h{half_life:g}_{level}_b{blend:.2f}_z{z:.1f}"
                    variants[key] = joint.backtest_metrics(accepted, result, len(dev))
    audit = {
        "training_rows": len(train), "action_count": len(bids),
        "context_axes": ["p_side_quantile", "selected_side", "utc_6h_bucket"],
        "forbidden_policy_features": [],
        "reward_only_columns": ["correct", "chosen_low"],
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
        "method": "hierarchical_empirical_bayes_reward_posterior",
        "selection_folds": TUNE, "untouched_holdout_folds": HOLDOUT,
        "btest_used": False, "winner": winner,
        "holdout_gate_passed": bool(winner["holdout_worst"] > 0),
        "fold_metrics": {fold: by_fold[fold][selected] for fold in FOLDS},
        "leakage_audit": audits, "top20": table.head(20).to_dict("records"),
    }
    table.to_csv(SESSION / "hierarchical_bayesian_bid_search.csv", index=False)
    (SESSION / "hierarchical_bayesian_bid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": winner, "holdout_gate_passed": summary["holdout_gate_passed"], "audit": audits["w1"]}, indent=2))


if __name__ == "__main__":
    main()
