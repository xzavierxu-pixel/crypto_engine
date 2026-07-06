#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SESSION = Path(__file__).resolve().parent
OUT = SESSION / "experiments" / "S3_combination_not_triggered"
OUT.mkdir(parents=True, exist_ok=True)

(OUT / "REPORT.md").write_text(
    "# S3 combination — not triggered\n\n"
    "S3 was conditional on S1 showing stable positive value. S1 improved zero of four tune weeks and failed its selection gate, so combining that stop with G or M would violate the research plan. B_test reads: `0`.\n",
    encoding="utf-8",
)
(OUT / "config_used.yaml").write_text(
    "experiment_id: S3_combination_not_triggered\ntrigger: S1_success\ntrigger_value: false\nbtest_reads: 0\n",
    encoding="utf-8",
)
(OUT / "feature_manifest.json").write_text(json.dumps({"feature_columns": [], "reason": "conditional node not executed"}, indent=2), encoding="utf-8")
(OUT / "leakage_check.json").write_text(json.dumps({"feature_intersection": [], "passed": True, "reason": "no model or decision executed"}, indent=2), encoding="utf-8")
(OUT / "metrics_bdev.json").write_text(json.dumps({"status": "not_triggered", "trigger": "S1_success", "trigger_value": False, "evidence": {"s1_tune_positive_delta_weeks": 0}}, indent=2), encoding="utf-8")
(OUT / "metrics_btest.json").write_text(json.dumps({"status": "not_triggered", "btest_evaluation_count": 0, "sum_pnl": None}, indent=2), encoding="utf-8")
pd.DataFrame(columns=["sample_id", "decision_time", "selected_side", "q", "bid", "market_price", "expected_ev", "fill_probability", "fill_flag", "filled", "correct", "realized_pnl"]).to_parquet(OUT / "predictions_btest.parquet", index=False)

ledger = SESSION / "gc_market_stop_btest_ledger.csv"
old = pd.read_csv(ledger)
old = old.loc[old["experiment_id"] != "S3_combination_not_triggered"]
row = {"experiment_id": "S3_combination_not_triggered", "track": "S", "btest_status": "not_triggered", "btest_sum_pnl": None, "anchor_same_universe": 42.43, "delta": None, "btest_reads": 0, "holdout_gate_passed": False}
pd.concat([old, pd.DataFrame([row])], ignore_index=True, sort=False).to_csv(ledger, index=False)
print(json.dumps(row, indent=2))
