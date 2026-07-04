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


joint = load_module("path_test_joint", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
base = load_module("path_test_base", ROOT / ".arbor/sessions/20260703_trade_btc_feature_research/run_feature_research.py")
pathmod = load_module("path_test_mod", SESSION / "run_path_signature_policy.py")


def main() -> None:
    train_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet"
    dev_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet"
    hazard_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    prepared = joint.prepare(train_path, dev_path, hazard_path, 20260704)
    lookup = base.load_trades([train, dev])
    tr_basic, dv_basic = base.engineer_trades(train, lookup), base.engineer_trades(dev, lookup)
    tr_path, dv_path = pathmod.engineer_path(train, lookup), pathmod.engineer_path(dev, lookup)
    tr_extra, dv_extra = pd.concat([tr_basic, tr_path], axis=1), pd.concat([dv_basic, dv_path], axis=1)
    hazard_cols, prep, grid, hazard = joint.load_hazard(hazard_path)
    q_path = pathmod.fit_q(pd.concat([train, tr_extra], axis=1), pd.concat([dev, dv_extra], axis=1),
                           hazard_cols + list(tr_extra.columns))
    q = 0.5 * prepared[2]["raw"] + 0.5 * q_path
    gc_train_all = joint.predict_hazard(hazard, prep.transform(train), torch.device("cpu"), 512)[1]
    gc_base = prepared[3]
    tr_mask = train["threshold_accepted"].astype(bool).to_numpy()
    dv_mask = dev["threshold_accepted"].astype(bool).to_numpy()
    tr, dv = train.loc[tr_mask].reset_index(drop=True), dev.loc[dv_mask].reset_index(drop=True)
    tr_extra, dv_extra = tr_extra.loc[tr_mask].reset_index(drop=True), dv_extra.loc[dv_mask].reset_index(drop=True)
    gc_train = gc_train_all[tr_mask]
    correct_train = np.flatnonzero(tr["correct"].astype(bool).to_numpy())
    sample_grid = np.unique(np.r_[np.arange(0, len(grid), 4), len(grid) - 1])

    def expand(gc: np.ndarray, frame: pd.DataFrame, extra: pd.DataFrame, indices: np.ndarray, gi: np.ndarray) -> np.ndarray:
        static = np.c_[frame.iloc[indices]["p_side"].to_numpy(float),
                       (frame.iloc[indices]["selected_side"].astype(str).str.upper() == "UP").to_numpy(float),
                       extra.iloc[indices].to_numpy(float)].astype("float32")
        return np.c_[np.tile(grid[gi], len(indices)), gc[indices][:, gi].reshape(-1),
                     np.repeat(static, len(gi), axis=0)].astype("float32")

    xtr = expand(gc_train, tr, tr_extra, correct_train, sample_grid)
    lows = tr.iloc[correct_train]["chosen_low"].to_numpy(float)
    ytr = (lows[:, None] <= grid[sample_grid][None, :]).astype("uint8").reshape(-1)
    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.025, num_leaves=15, max_depth=5,
        min_child_samples=250, subsample=0.8, colsample_bytree=0.25,
        reg_lambda=20, reg_alpha=4, random_state=20260704, verbosity=-1, n_jobs=-1,
        monotone_constraints=[1, 1] + [0] * (xtr.shape[1] - 2),
    )
    model.fit(xtr, ytr)
    idx = np.arange(len(dv))
    gc = model.predict_proba(expand(gc_base, dv, dv_extra, idx, np.arange(len(grid))))[:, 1].reshape(len(dv), len(grid))
    gc = np.maximum.accumulate(np.clip(gc, 0, 1), axis=1)
    bid, ev, fill = joint.choose_survival_expected_return_bids(q, gc, grid, 0.01, 0.02, min_fill_probability=0.85)
    result = joint.backtest_with_bid(dv, bid, ev, fill)
    metrics = joint.backtest_metrics(dv, result, len(dev))
    diagnostics = dv[["timestamp", "decision_time", "selected_side", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["q_path_blend"] = q
    diagnostics["bid"] = bid
    diagnostics["expected_ev"] = ev
    diagnostics["model_fill_probability"] = fill
    diagnostics["realized_pnl"] = result.pnl
    diagnostics["filled"] = result.filled
    diagnostics.to_parquet(SESSION / "btest_path_signature_diagnostics.parquet", index=False)
    payload = {
        "candidate": "frozen node 2.1 path-signature q+Gc; floor=0.85; min_ev=0.02",
        "selection_evidence": {"w1_w4_sum_pnl": 288.89, "w5_w6_sum_pnl": 139.0,
                               "q_and_gc_brier_improved_all_six_folds": True},
        "baseline_anchor": 27.44, "previous_research_best": 42.43,
        "metrics": metrics, "btest_evaluation_count_this_session": 1,
    }
    (SESSION / "btest_path_signature_milestone_once.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
