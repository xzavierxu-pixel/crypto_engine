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


joint = load_module("joint_btest_trade", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
features = load_module("feature_btest_trade", SESSION / "run_feature_research.py")


def main() -> None:
    train_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet"
    dev_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet"
    hazard_path = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    prepared = joint.prepare(train_path, dev_path, hazard_path, 20260703)
    lookup = features.load_trades([train, dev])
    tr_trade, dv_trade = features.engineer_trades(train, lookup), features.engineer_trades(dev, lookup)
    hazard_columns = joint.load_hazard(hazard_path)[0]
    q_trade = features.fit_predict(
        pd.concat([train, tr_trade], axis=1), pd.concat([dev, dv_trade], axis=1),
        hazard_columns + list(tr_trade.columns), 20260703,
    )
    q = 0.5 * prepared[2]["raw_tree_blend"] + 0.5 * q_trade
    dev_all, accepted, _, gc, grid, _, _ = prepared
    bid, ev, fill = joint.choose_survival_expected_return_bids(
        q, gc, grid, 0.01, 0.02, min_fill_probability=0.80,
    )
    backtest = joint.backtest_with_bid(accepted, bid, ev, fill)
    diagnostics = accepted[["timestamp", "decision_time", "selected_side", "p_side", "correct", "chosen_low"]].copy()
    diagnostics["q_trade_blend"] = q
    diagnostics["q_trade_lgbm"] = q_trade
    diagnostics["q_old_tree_blend"] = prepared[2]["raw_tree_blend"]
    diagnostics["q_raw"] = prepared[2]["raw"]
    diagnostics["candidate_bid"] = bid
    diagnostics["candidate_ev"] = ev
    diagnostics["candidate_fill_probability"] = fill
    diagnostics["realized_pnl"] = backtest.pnl
    diagnostics["filled"] = backtest.filled
    diagnostics["printed_filled"] = backtest.printed_filled
    diagnostics.to_parquet(SESSION / "btest_q_milestone_diagnostics.parquet", index=False)
    metrics = joint.backtest_metrics(accepted, backtest, len(dev_all))
    payload = {
        "candidate": "frozen node-1 oldnew_trade_lgbm_blend; base Gc floor=0.80; min_ev=0.02",
        "selection_evidence": {"w1_w4_sum_pnl": 266.34, "w5_w6_sum_pnl": 128.49},
        "baseline_anchor": {"experiment_id": "20260619_expected_return_h14_h2_gc_gt_0p75", "sum_pnl": 27.44},
        "metrics": metrics, "btest_evaluation_count_this_session": 1,
        "leakage_guard": "trade_time <= decision_time; stage1_sample_weight and label-derived columns excluded",
    }
    (SESSION / "btest_q_milestone_once.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
