#!/usr/bin/env python3
"""Single frozen B_test milestone for the B_dev-selected anchored contextual policy."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

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


joint = load("anchored_btest_joint", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
anchor = load("anchored_btest_policy", SESSION / "run_anchored_contextual_lcb.py")


def main() -> None:
    summary = json.loads((SESSION / "anchored_contextual_lcb_summary.json").read_text(encoding="utf-8"))
    if not summary["holdout_gate_passed"]:
        raise RuntimeError("B_dev holdout gate failed; B_test evaluation is forbidden")
    variant = str(summary["winner"]["variant"])
    match = re.fullmatch(r"a([0-9.]+)_u([0-9.]+)_d(\d+)", variant)
    if match is None:
        raise ValueError(f"unrecognized frozen variant: {variant}")
    alpha, beta, max_step = float(match.group(1)), float(match.group(2)), int(match.group(3))

    train = pd.read_parquet(TRAIN_PATH)
    prepared = joint.prepare(TRAIN_PATH, TEST_PATH, HAZARD_PATH, SEED)
    test, accepted, qs, gc, grid, _, _ = prepared
    train = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    xtr, xte = anchor.context(train), anchor.context(accepted)
    action_idx = np.arange(4, min(len(grid), 86), 4)
    bids = grid[action_idx]
    correct = train["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(train["chosen_low"], errors="coerce").fillna(1.0).to_numpy(float)
    rewards = np.where(correct[:, None], np.where(low[:, None] <= bids[None, :], 1.0 - bids[None, :], 0.0), -bids[None, :]).astype("float32")
    n, actions = rewards.shape
    train_x = np.column_stack([np.repeat(xtr, actions, axis=0), np.tile(bids, n), np.tile(bids * bids, n)])
    test_x = np.column_stack([np.repeat(xte, actions, axis=0), np.tile(bids, len(xte)), np.tile(bids * bids, len(xte))])
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
        predictions.append(model.predict(test_x).reshape(len(xte), actions))
    pred = np.stack(predictions)
    reward_mean, reward_std = pred.mean(axis=0), pred.std(axis=0)

    q = qs["raw_tree_blend"]
    analytic_ev = q[:, None] * gc[:, action_idx] * (1.0 - bids[None, :]) - (1.0 - q[:, None]) * bids[None, :]
    legal = gc[:, action_idx] >= 0.85
    base_idx = np.argmax(np.where(legal, analytic_ev, -np.inf), axis=1)
    distance_ok = np.abs(np.arange(actions)[None, :] - base_idx[:, None]) <= max_step
    score = (1.0 - alpha) * analytic_ev + alpha * reward_mean - beta * reward_std
    masked = np.where(legal & distance_ok, score, -np.inf)
    idx = np.argmax(masked, axis=1)
    value = masked[np.arange(len(xte)), idx]
    submit = np.isfinite(value) & (value >= 0.02)
    bid = np.where(submit, bids[idx], 0.0)
    fill = np.where(submit, gc[np.arange(len(xte)), action_idx[idx]], 0.0)
    result = joint.backtest_with_bid(accepted, bid, np.where(submit, value, 0.0), fill)
    metrics = joint.backtest_metrics(accepted, result, len(test))

    diagnostics = accepted[["timestamp", "decision_time", "selected_side", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["bid"] = bid
    diagnostics["policy_score"] = np.where(submit, value, 0.0)
    diagnostics["pnl"] = result.pnl
    diagnostics.to_parquet(SESSION / "btest_anchored_contextual_diagnostics.parquet", index=False)
    payload = {
        "evaluation_kind": "single milestone after B_dev selection and untouched holdout gate",
        "selection_was_bdev_only": True, "variant": variant,
        "bdev_winner": summary["winner"], "baseline_sum_pnl": 27.44,
        "previous_best_btest_sum_pnl": 42.43, "metrics": metrics,
        "btest_window": {"start": str(pd.to_datetime(test["timestamp"], utc=True).min()), "end": str(pd.to_datetime(test["timestamp"], utc=True).max())},
        "btest_evaluation_count": 1,
        "leakage_guard": "explicit decision-time feature allowlist; correct/chosen_low used only as training rewards",
    }
    (SESSION / "btest_anchored_contextual_milestone_once.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
