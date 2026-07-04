#!/usr/bin/env python3
"""Analytic-EV-anchored contextual action-value ensemble on rolling B_dev."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

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


joint = load_module("anchored_joint", JOINT_PATH)

FEATURES = [
    "p_side", "direction_confidence", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "sl_return_5s", "sl_return_30s", "sl_rv_30s", "sl_taker_imbalance_30s",
    "sl_signed_dollar_flow_ratio_30s", "sl_directional_efficiency_30s",
    "sl_choppiness_30s", "sl_price_minus_vwap_30s", "sl_volume_burst_30s",
    "ret_1", "ret_3", "ret_5", "rv_5", "relative_volume_5", "volume_z_5",
]


def context(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for name in FEATURES:
        if name in frame:
            values.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(float))
        else:
            values.append(np.zeros(len(frame)))
    values.append((frame["selected_side"].astype(str).str.upper() == "UP").to_numpy(float))
    return np.nan_to_num(np.column_stack(values), nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def fit_fold(fold: str) -> tuple[dict[str, dict], dict]:
    d = PREV / "folds" / fold
    prepared = joint.prepare(
        d / "data/train.parquet", d / "data/dev.parquet", d / "models/hazard_survival_cdf.pt", SEED
    )
    dev, accepted, qs, gc, grid, _, _ = prepared
    train = pd.read_parquet(d / "data/train.parquet")
    train = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    xtr, xdv = context(train), context(accepted)

    action_idx = np.arange(4, min(len(grid), 86), 4)
    bids = grid[action_idx]
    correct = train["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(train["chosen_low"], errors="coerce").fillna(1.0).to_numpy(float)
    rewards = np.where(
        correct[:, None],
        np.where(low[:, None] <= bids[None, :], 1.0 - bids[None, :], 0.0),
        -bids[None, :],
    ).astype("float32")
    n, actions = rewards.shape
    train_x = np.column_stack([
        np.repeat(xtr, actions, axis=0), np.tile(bids, n), np.tile(bids * bids, n)
    ])
    dev_x = np.column_stack([
        np.repeat(xdv, actions, axis=0), np.tile(bids, len(xdv)), np.tile(bids * bids, len(xdv))
    ])

    predictions = []
    for member in range(5):
        model = LGBMRegressor(
            objective="huber", alpha=0.8, n_estimators=300, learning_rate=0.03,
            num_leaves=16, max_depth=5, min_child_samples=250,
            subsample=0.75, subsample_freq=1, colsample_bytree=0.75,
            reg_lambda=30.0, reg_alpha=5.0, random_state=SEED + member,
            verbosity=-1, n_jobs=-1,
        )
        model.fit(train_x, rewards.reshape(-1))
        predictions.append(model.predict(dev_x).reshape(len(xdv), actions))
    pred = np.stack(predictions)
    reward_mean, reward_std = pred.mean(axis=0), pred.std(axis=0)

    q = qs["raw_tree_blend"]
    analytic_ev = q[:, None] * gc[:, action_idx] * (1.0 - bids[None, :]) - (1.0 - q[:, None]) * bids[None, :]
    legal = gc[:, action_idx] >= 0.85
    base_masked = np.where(legal, analytic_ev, -np.inf)
    base_idx = np.argmax(base_masked, axis=1)
    base_value = base_masked[np.arange(len(xdv)), base_idx]

    variants: dict[str, dict] = {}
    for alpha in (0.0, 0.10, 0.20, 0.35, 0.50):
        for beta in (0.0, 0.25, 0.50, 1.0):
            for max_step in (0, 2, 4, 8):
                distance_ok = np.abs(np.arange(actions)[None, :] - base_idx[:, None]) <= max_step
                score = (1.0 - alpha) * analytic_ev + alpha * reward_mean - beta * reward_std
                masked = np.where(legal & distance_ok, score, -np.inf)
                idx = np.argmax(masked, axis=1)
                value = masked[np.arange(len(xdv)), idx]
                submit = np.isfinite(value) & (value >= 0.02)
                bid = np.where(submit, bids[idx], 0.0)
                result = joint.backtest_with_bid(accepted, bid, np.where(submit, value, 0.0), np.where(submit, gc[np.arange(len(xdv)), action_idx[idx]], 0.0))
                key = f"a{alpha:.2f}_u{beta:.2f}_d{max_step}"
                variants[key] = joint.backtest_metrics(accepted, result, len(dev))

    audit = {
        "training_rows": n, "action_count": actions, "ensemble_members": len(predictions),
        "feature_allowlist": FEATURES + ["selected_side_up"],
        "forbidden_feature_intersection": sorted(set(FEATURES) & {
            "target", "correct", "chosen_low", "winner", "pnl", "trade_time", "endDate",
            "condition_id", "market_id", "slug", "outcome",
        }),
        "base_submitted": int(np.sum(base_value >= 0.02)),
    }
    return variants, audit


def main() -> None:
    by_fold, audits = {}, {}
    for fold in FOLDS:
        by_fold[fold], audits[fold] = fit_fold(fold)
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
        "method": "analytic_ev_anchored_contextual_lcb", "selection_folds": TUNE,
        "untouched_holdout_folds": HOLDOUT, "btest_used": False, "winner": winner,
        "holdout_gate_passed": bool(winner["holdout_worst"] > 0),
        "fold_metrics": {fold: by_fold[fold][selected] for fold in FOLDS},
        "leakage_audit": audits, "top20": table.head(20).to_dict("records"),
    }
    table.to_csv(SESSION / "anchored_contextual_lcb_search.csv", index=False)
    (SESSION / "anchored_contextual_lcb_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": winner, "holdout_gate_passed": summary["holdout_gate_passed"], "audit": audits["w1"]}, indent=2))


if __name__ == "__main__":
    main()
