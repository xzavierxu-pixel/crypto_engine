#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
OLD = ROOT / ".arbor/sessions/20260703_prefinal_rolling"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("online_test_joint", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")


def lgb(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=350, learning_rate=0.03, num_leaves=15, max_depth=5,
        min_child_samples=150, colsample_bytree=0.4, reg_lambda=12, reg_alpha=3,
        verbosity=-1, n_jobs=-1, random_state=seed,
    )


def main() -> None:
    train_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet"
    dev_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet"
    hazard_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt"
    raw_train = pd.read_parquet(train_path, columns=["timestamp"])
    raw_dev = pd.read_parquet(dev_path, columns=["timestamp"])
    pre = pd.read_parquet(OLD / "pretest_both_side_lows.parquet")
    test = pd.read_parquet(OLD / "btest_both_side_lows.parquet")
    train_ts = pd.to_datetime(raw_train["timestamp"], utc=True)
    dev_ts = pd.to_datetime(raw_dev["timestamp"], utc=True)
    tr = pre.loc[pd.to_datetime(pre["timestamp"], utc=True).isin(train_ts)].copy().reset_index(drop=True)
    dv = test.loc[pd.to_datetime(test["timestamp"], utc=True).isin(dev_ts)].copy().reset_index(drop=True)
    if len(tr) != len(raw_train) or len(dv) != len(raw_dev):
        raise RuntimeError(f"counterfactual alignment failed train={len(tr)}/{len(raw_train)} test={len(dv)}/{len(raw_dev)}")

    cols, prep, grid, hazard = joint.load_hazard(hazard_path)
    xtr, xdv = joint.matrix(tr, cols), joint.matrix(dv, cols)
    y = tr["target"].astype(int)
    static_lgb = lgb(20260704).fit(xtr, y)
    static_cat = CatBoostClassifier(
        iterations=350, depth=5, learning_rate=0.035, l2_leaf_reg=12,
        loss_function="Logloss", verbose=False, allow_writing_files=False,
        random_seed=20260704,
    ).fit(xtr, y)
    static = 0.5 * static_lgb.predict_proba(xdv)[:, 1] + 0.5 * static_cat.predict_proba(xdv)[:, 1]

    days = pd.to_datetime(dv["timestamp"], utc=True).dt.floor("D")
    online = np.zeros(len(dv), dtype=float)
    training_rows_by_day: dict[str, int] = {}
    for day in sorted(days.unique()):
        current = (days == day).to_numpy()
        prior = (days < day).to_numpy()
        history = pd.concat([tr, dv.loc[prior]], ignore_index=True)
        model = lgb(20260704).fit(joint.matrix(history, cols), history["target"].astype(int))
        online[current] = model.predict_proba(xdv.loc[current])[:, 1]
        training_rows_by_day[str(day)] = int(len(history))

    p = 0.5 * static + 0.5 * online
    side = np.where(p >= 0.5, "UP", "DOWN")
    original = dv["selected_side"].astype(str).str.upper().to_numpy()
    dv["selected_side"] = side
    dv["p_up"] = p
    dv["p_side"] = np.maximum(p, 1.0 - p)
    dv["direction_confidence"] = np.abs(p - 0.5)
    truth = np.where(dv["target"].astype(int).to_numpy() == 1, "UP", "DOWN")
    dv["correct"] = side == truth
    alternative = np.where(side == "UP", dv["up_low"], dv["down_low"])
    dv["chosen_low"] = np.where(side == original, dv["chosen_low"], alternative)
    gc = joint.predict_hazard(hazard, prep.transform(dv), torch.device("cpu"), 512)[1]
    q = dv["p_side"].to_numpy(float)
    bid, ev, fill = joint.choose_survival_expected_return_bids(q, gc, grid, 0.01, 0.0, min_fill_probability=0.80)
    bid[(q < 0.55) | ~np.isfinite(dv["chosen_low"].to_numpy(float))] = 0.0
    result = joint.backtest_with_bid(dv, bid, ev, fill)
    metrics = joint.backtest_metrics(dv, result, len(dv))
    diagnostics = dv[["timestamp", "decision_time", "selected_side", "target", "p_up", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["static_p_up"] = static
    diagnostics["online_p_up"] = online
    diagnostics["bid"] = bid
    diagnostics["expected_ev"] = ev
    diagnostics["realized_pnl"] = result.pnl
    diagnostics.to_parquet(SESSION / "btest_online_full_direction_diagnostics.parquet", index=False)
    payload = {
        "candidate": "node 1.1 settlement-safe static_online_all direction; min_q=0.55 floor=0.80 min_ev=0",
        "selection_evidence": {"w1_w4_sum_pnl": 260.53, "w5_w6_sum_pnl": 165.61,
                               "daily_update_uses_only_prior_settled_days": True},
        "baseline_anchor": 27.44, "previous_research_best": 42.43,
        "training_rows_by_day": training_rows_by_day,
        "metrics": metrics, "btest_evaluation_count_this_session": 3,
    }
    (SESSION / "btest_online_full_direction_milestone_once.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
