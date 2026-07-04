#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
module_path = ROOT/".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
spec = importlib.util.spec_from_file_location("joint", module_path)
joint = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(joint)

summary = json.loads((SESSION/"rolling_policy_summary.json").read_text(encoding="utf-8"))
if not summary["holdout_gate_passed"]:
    raise RuntimeError("Untouched pre-test holdout gate failed; B_test is forbidden")
winner = summary["winner"]
prepared = joint.prepare(
    ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet",
    ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet",
    ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt",
    20260703,
)
metrics = joint.evaluate(prepared, winner["q_model"], float(winner["shrink"]), float(winner["gc_floor"]), float(winner["min_ev"]))
payload = {"evaluation_kind":"single milestone after six-fold pre-test rolling selection",
           "baseline_sum_pnl":27.44,"target_sum_pnl":100.0,"candidate":winner,"metrics":metrics,
           "q_calibration":prepared[5][winner["q_model"]],"btest_evaluation_count_this_session":1}
(SESSION/"btest_milestone_once.json").write_text(json.dumps(payload,indent=2,allow_nan=True),encoding="utf-8")
print(json.dumps(payload,indent=2,allow_nan=True))
