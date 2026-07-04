#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("btest_gc_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
feat = load_module("btest_gc_feat", SESSION / "run_feature_research.py")
gcmod = load_module("btest_gc_helpers", SESSION / "run_gc_feature_research.py")


def main() -> None:
    train_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet"
    dev_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet"
    hazard_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    prepared = joint.prepare(train_path, dev_path, hazard_path, 20260703)
    lookup = feat.load_trades([train, dev])
    tr_trade, dv_trade = feat.engineer_trades(train, lookup), feat.engineer_trades(dev, lookup)
    hazard_cols, prep, grid, hazard = joint.load_hazard(hazard_path)
    q_trade = feat.fit_predict(pd.concat([train, tr_trade], axis=1), pd.concat([dev, dv_trade], axis=1),
                               hazard_cols + list(tr_trade.columns), 20260703)
    q = 0.5 * prepared[2]["raw_tree_blend"] + 0.5 * q_trade
    gc_train_all = joint.predict_hazard(hazard, prep.transform(train), torch.device("cpu"), 512)[1]
    tr_mask = train["threshold_accepted"].astype(bool).to_numpy()
    dv_mask = dev["threshold_accepted"].astype(bool).to_numpy()
    tr, dv = train.loc[tr_mask].reset_index(drop=True), dev.loc[dv_mask].reset_index(drop=True)
    tr_trade, dv_trade = tr_trade.loc[tr_mask].reset_index(drop=True), dv_trade.loc[dv_mask].reset_index(drop=True)
    gc_train, gc_base = gc_train_all[tr_mask], prepared[3]
    correct_train = np.flatnonzero(tr["correct"].astype(bool).to_numpy())
    sample_grid = np.unique(np.r_[np.arange(0, len(grid), 4), len(grid) - 1])
    xtr = gcmod.expand(gc_train, grid, tr, tr_trade, correct_train, sample_grid)
    lows = tr.iloc[correct_train]["chosen_low"].to_numpy(float)
    ytr = (lows[:, None] <= grid[sample_grid][None, :]).astype("uint8").reshape(-1)
    monotone = [1, 1] + [0] * (xtr.shape[1] - 2)
    model = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=5,
                           min_child_samples=200, subsample=0.8, colsample_bytree=0.5,
                           reg_lambda=15, reg_alpha=3, random_state=20260703, verbosity=-1,
                           n_jobs=-1, monotone_constraints=monotone)
    model.fit(xtr, ytr)
    xdv = gcmod.expand(gc_base, grid, dv, dv_trade, np.arange(len(dv)), np.arange(len(grid)))
    gc_new = model.predict_proba(xdv)[:, 1].reshape(len(dv), len(grid))
    gc_new = np.maximum.accumulate(np.clip(gc_new, 0.0, 1.0), axis=1)
    bid, ev, fill = joint.choose_survival_expected_return_bids(q, gc_new, grid, 0.01, 0.05, min_fill_probability=0.90)
    backtest = joint.backtest_with_bid(dv, bid, ev, fill)
    metrics = joint.backtest_metrics(dv, backtest, len(dev))
    diagnostics = dv[["timestamp", "decision_time", "selected_side", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["q_trade_blend"] = q
    diagnostics["bid"] = bid
    diagnostics["expected_ev"] = ev
    diagnostics["model_fill_probability"] = fill
    diagnostics["realized_pnl"] = backtest.pnl
    diagnostics["filled"] = backtest.filled
    diagnostics.to_parquet(SESSION / "btest_trade_gc_diagnostics.parquet", index=False)
    payload = {"candidate": "frozen node 1.1 trade-conditioned monotone Gc; floor=0.90; min_ev=0.05",
               "selection_evidence": {"w1_w4_sum_pnl": 290.31, "w5_w6_sum_pnl": 138.58},
               "baseline_anchor": 27.44, "previous_research_best": 42.43,
               "metrics": metrics, "btest_evaluation_count_this_session": 2}
    (SESSION / "btest_trade_gc_milestone_once.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
