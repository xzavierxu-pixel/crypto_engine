#!/usr/bin/env python3
"""One frozen B_test comparison for the selected Bayesian and contextual policies."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

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


joint = load("btest_policy_joint", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
minimal = load("btest_policy_minimal", ROOT / ".arbor/sessions/20260703_trade_btc_feature_research/run_minimal_q.py")
trade_features = load("btest_policy_features", ROOT / ".arbor/sessions/20260703_trade_btc_feature_research/run_feature_research.py")
bayes = load("btest_policy_bayes", SESSION / "run_bayesian_bid_policy.py")
contextual = load("btest_policy_contextual", SESSION / "run_contextual_action_value.py")


def minimal_q(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    lookup = trade_features.load_trades([train, test])
    tr_trade = trade_features.engineer_trades(train, lookup)
    te_trade = trade_features.engineer_trades(test, lookup)
    tr_mask = train["threshold_accepted"].astype(bool)
    te_mask = test["threshold_accepted"].astype(bool)
    tr = train.loc[tr_mask].reset_index(drop=True)
    te = test.loc[te_mask].reset_index(drop=True)
    xtr = minimal.design(tr, tr_trade.loc[tr_mask].reset_index(drop=True))
    xte = minimal.design(te, te_trade.loc[te_mask].reset_index(drop=True))
    ts = pd.to_datetime(tr["timestamp"], utc=True)
    fit_mask = (ts < ts.max() - pd.Timedelta(days=7)).to_numpy()
    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.025, num_leaves=12, max_depth=4,
        min_child_samples=150, subsample=0.8, colsample_bytree=0.65,
        reg_lambda=15, reg_alpha=4, random_state=20260703, verbosity=-1, n_jobs=-1,
    )
    model.fit(xtr.loc[fit_mask], tr.loc[fit_mask, "correct"].astype(int))
    q_minimal = model.predict_proba(xte)[:, 1]
    return 0.75 * te["p_side"].to_numpy(float) + 0.25 * q_minimal


def contextual_policy(train: pd.DataFrame, accepted: pd.DataFrame, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    xtr, xte = contextual.context(train), contextual.context(accepted)
    action_idx = np.arange(4, min(len(grid), 86), 4)
    bids = grid[action_idx]
    correct = train["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(train["chosen_low"], errors="coerce").fillna(1.0).to_numpy(float)
    reward = np.where(correct[:, None], np.where(low[:, None] <= bids[None, :], 1.0 - bids[None, :], 0.0), -bids[None, :]).astype("float32")
    n, actions = reward.shape
    train_x = np.column_stack([np.repeat(xtr, actions, axis=0), np.tile(bids, n), np.tile(bids * bids, n)])
    model = LGBMRegressor(
        objective="regression_l1", n_estimators=350, learning_rate=0.035,
        num_leaves=20, max_depth=5, min_child_samples=200,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=20.0, reg_alpha=3.0,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    model.fit(train_x, reward.reshape(-1))
    nt = len(xte)
    test_x = np.column_stack([np.repeat(xte, actions, axis=0), np.tile(bids, nt), np.tile(bids * bids, nt)])
    values = model.predict(test_x).reshape(nt, actions)
    legal = bids <= 0.52  # frozen B_dev winner v0.000_b0.52
    masked = np.where(legal[None, :], values, -np.inf)
    idx = np.argmax(masked, axis=1)
    value = masked[np.arange(nt), idx]
    bid = np.where(value >= 0.0, bids[idx], 0.0)
    return bid, np.maximum(value, 0.0)


def evaluate(accepted: pd.DataFrame, total_rows: int, bid: np.ndarray, ev: np.ndarray, fill: np.ndarray) -> tuple[dict, object]:
    result = joint.backtest_with_bid(accepted, bid, ev, fill)
    return joint.backtest_metrics(accepted, result, total_rows), result


def main() -> None:
    train, test = pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)
    prepared = joint.prepare(TRAIN_PATH, TEST_PATH, HAZARD_PATH, SEED)
    _, accepted, _, gc, grid, _, _ = prepared

    theta = np.asarray(json.loads((SESSION / "bayesian_bid_policy_summary.json").read_text())["theta"], dtype=float)
    q = minimal_q(train, test)
    bayes_payload = {"accepted": accepted, "dev": test, "qs": {"raw_minimal_0.25": q}, "gcs": {"hazard": gc}, "grid": grid}
    bayes_bid, bayes_ev, bayes_fill = bayes.policy(bayes_payload, theta)
    bayes_metrics, bayes_result = evaluate(accepted, len(test), bayes_bid, bayes_ev, bayes_fill)

    ctx_bid, ctx_ev = contextual_policy(train, accepted, grid)
    # Direct action-value policies do not produce a Gc probability forecast.
    ctx_metrics, ctx_result = evaluate(accepted, len(test), ctx_bid, ctx_ev, np.full(len(accepted), np.nan))

    diagnostics = accepted[["timestamp", "decision_time", "selected_side", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["bayesian_bid"] = bayes_bid
    diagnostics["bayesian_pnl"] = bayes_result.pnl
    diagnostics["contextual_bid"] = ctx_bid
    diagnostics["contextual_pnl"] = ctx_result.pnl
    diagnostics.to_parquet(SESSION / "btest_bid_policy_comparison_diagnostics.parquet", index=False)

    payload = {
        "btest_window": {"start": str(pd.to_datetime(test["timestamp"], utc=True).min()), "end": str(pd.to_datetime(test["timestamp"], utc=True).max())},
        "selection_was_bdev_only": True,
        "baseline": {"name": "expected_return_h14", "sum_pnl": 27.44},
        "bayesian": {"policy": "frozen gaussian-process winner", "bdev_robust_sum_pnl": 164.29521496589103, "metrics": bayes_metrics},
        "contextual": {"policy": "frozen full-information action-value v0.000_b0.52", "bdev_robust_sum_pnl": 17.13460190063472, "metrics": ctx_metrics},
        "winner_on_btest": "bayesian" if bayes_metrics["sum_pnl"] > ctx_metrics["sum_pnl"] else "contextual",
        "btest_evaluation_count": 1,
        "leakage_guard": "policy inputs use explicit decision-time allowlists; correct/chosen_low are training rewards only",
    }
    (SESSION / "btest_bid_policy_comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
