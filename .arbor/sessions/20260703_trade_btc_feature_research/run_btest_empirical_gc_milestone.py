#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("btest_emp_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
feat = load_module("btest_emp_feat", SESSION / "run_feature_research.py")
emp = load_module("btest_emp_model", SESSION / "run_empirical_gc.py")


def main() -> None:
    train_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet"
    dev_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet"
    hazard_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    prepared = joint.prepare(train_path, dev_path, hazard_path, 20260703)
    lookup = feat.load_trades([train, dev])
    tr_trade, dv_trade = feat.engineer_trades(train, lookup), feat.engineer_trades(dev, lookup)
    hazard_cols, _, grid, _ = joint.load_hazard(hazard_path)
    q_trade = feat.fit_predict(pd.concat([train, tr_trade], axis=1), pd.concat([dev, dv_trade], axis=1),
                               hazard_cols + list(tr_trade.columns), 20260703)
    q = 0.5 * prepared[2]["raw_tree_blend"] + 0.5 * q_trade
    tr = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    dv = dev.loc[dev["threshold_accepted"].astype(bool)].reset_index(drop=True)
    gc = emp.empirical_gc(tr, dv, grid, 28, "pbin")
    bid, ev, fill = joint.choose_survival_expected_return_bids(q, gc, grid, 0.01, 0.005, min_fill_probability=0.80)
    backtest = joint.backtest_with_bid(dv, bid, ev, fill)
    metrics = joint.backtest_metrics(dv, backtest, len(dev))
    diagnostics = dv[["timestamp", "decision_time", "selected_side", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["q_trade_blend"] = q
    diagnostics["bid"] = bid
    diagnostics["expected_ev"] = ev
    diagnostics["model_fill_probability"] = fill
    diagnostics["realized_pnl"] = backtest.pnl
    diagnostics["filled"] = backtest.filled
    diagnostics.to_parquet(SESSION / "btest_empirical_gc_diagnostics.parquet", index=False)
    payload = {"candidate": "frozen node 5 empirical 28-day p_side-bin Gc; floor=0.80; min_ev=0.005",
               "selection_evidence": {"w1_w4_sum_pnl": 272.94, "w5_w6_sum_pnl": 148.22},
               "baseline_anchor": 27.44, "previous_research_best": 42.43,
               "metrics": metrics, "btest_evaluation_count_this_session": 3}
    (SESSION / "btest_empirical_gc_milestone_once.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
