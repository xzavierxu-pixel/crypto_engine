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
SOURCE = ROOT / ".arbor/sessions/20260703_prefinal_rolling"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("btest_full_direction_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")


def main() -> None:
    train = pd.read_parquet(SOURCE / "pretest_both_side_lows.parquet")
    dev = pd.read_parquet(SOURCE / "btest_both_side_lows.parquet")
    checkpoint = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt"
    cols, prep, grid, hazard = joint.load_hazard(checkpoint)
    xtr, xdv = joint.matrix(train, cols), joint.matrix(dev, cols)
    y = train["target"].astype(int).to_numpy()
    lgb = LGBMClassifier(n_estimators=500, learning_rate=0.025, num_leaves=15, max_depth=5,
                         min_child_samples=150, colsample_bytree=0.4, reg_lambda=12, reg_alpha=3,
                         verbosity=-1, n_jobs=-1, random_state=20260703).fit(xtr, y)
    cat = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.03, l2_leaf_reg=12,
                             loss_function="Logloss", verbose=False, allow_writing_files=False,
                             random_seed=20260703).fit(xtr, y)
    p = 0.5 * lgb.predict_proba(xdv)[:, 1] + 0.5 * cat.predict_proba(xdv)[:, 1]
    side = np.where(p >= 0.5, "UP", "DOWN")
    work = dev.copy()
    original = work["selected_side"].astype(str).str.upper().to_numpy()
    work["selected_side"] = side
    work["p_up"] = p
    work["p_side"] = np.maximum(p, 1 - p)
    work["direction_confidence"] = np.abs(p - 0.5)
    work["correct"] = side == np.where(work["target"].astype(int).to_numpy() == 1, "UP", "DOWN")
    alternative = np.where(side == "UP", work["up_low"], work["down_low"])
    work["chosen_low"] = np.where(side == original, work["chosen_low"], alternative)
    gc = joint.predict_hazard(hazard, prep.transform(work), torch.device("cpu"), 512)[1]
    q = work["p_side"].to_numpy(float)
    bid, ev, fill = joint.choose_survival_expected_return_bids(q, gc, grid, 0.01, 0.0, min_fill_probability=0.80)
    bid[~np.isfinite(work["chosen_low"].to_numpy(float))] = 0.0
    backtest = joint.backtest_with_bid(work, bid, ev, fill)
    metrics = joint.backtest_metrics(work, backtest, len(work))
    diagnostics = work[["timestamp", "decision_time", "selected_side", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["bid"] = bid
    diagnostics["expected_ev"] = ev
    diagnostics["model_fill_probability"] = fill
    diagnostics["realized_pnl"] = backtest.pnl
    diagnostics["filled"] = backtest.filled
    diagnostics.to_parquet(SESSION / "btest_full_universe_direction_diagnostics.parquet", index=False)
    payload = {
        "candidate": "frozen node 7 full-universe tree direction; Gc floor=0.80; min_ev=0",
        "selection_evidence": {"w1_w4_sum_pnl": 283.59, "w5_w6_sum_pnl": 173.10},
        "baseline_anchor": {"experiment_id": "20260619_expected_return_h14_h2_gc_gt_0p75", "sum_pnl": 27.44},
        "previous_research_best": 42.43,
        "direction_changed_count": int((side != original).sum()),
        "direction_coverage": 1.0,
        "metrics": metrics,
        "btest_evaluation_kind": "new full-universe action-space milestone",
    }
    (SESSION / "btest_full_universe_direction_milestone.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
