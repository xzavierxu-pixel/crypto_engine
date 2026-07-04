#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
SOURCE_MODULE = ROOT/".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
spec = importlib.util.spec_from_file_location("joint", SOURCE_MODULE)
joint = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(joint)

FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE = FOLDS[:4]
HOLDOUT = FOLDS[4:]


def main() -> None:
    prepared = {}
    for fold in FOLDS:
        d = SESSION/"folds"/fold
        prepared[fold] = joint.prepare(d/"data/train.parquet", d/"data/dev.parquet", d/"models/hazard_survival_cdf.pt", 20260703)

    # Exact policy baseline under the same rolling checkpoints.
    baseline = {f: joint.evaluate(p, "raw", 0.0, 0.75, 0.0)["sum_pnl"] for f,p in prepared.items()}
    rows = []
    for q_name in prepared["w1"][2]:
        for shrink in [0.0, .1, .2, .3]:
            for floor in [.70, .75, .80, .85, .90]:
                for min_ev in [0.0, .005, .01, .02, .03, .05]:
                    pnl = {f: joint.evaluate(p, q_name, shrink, floor, min_ev)["sum_pnl"] for f,p in prepared.items()}
                    delta = {f: pnl[f] - baseline[f] for f in FOLDS}
                    tune_values = np.array([pnl[f] for f in TUNE])
                    tune_deltas = np.array([delta[f] for f in TUNE])
                    rows.append({"q_model":q_name,"shrink":shrink,"gc_floor":floor,"min_ev":min_ev,
                                 **{f"{f}_pnl":pnl[f] for f in FOLDS}, **{f"{f}_delta":delta[f] for f in FOLDS},
                                 "tune_sum":tune_values.sum(), "tune_worst":tune_values.min(),
                                 "tune_positive_delta_weeks":int((tune_deltas>0).sum()),
                                 "tune_robust_score":float(tune_values.sum() - tune_values.std()),
                                 "holdout_sum":sum(pnl[f] for f in HOLDOUT),
                                 "holdout_delta":sum(delta[f] for f in HOLDOUT)})
    table = pd.DataFrame(rows)
    # Winner is fixed using w1-w4 only. w5-w6 are a genuine untouched gate.
    eligible = table.loc[table["tune_positive_delta_weeks"] >= 3]
    winner = eligible.sort_values(["tune_robust_score","tune_sum","tune_worst"], ascending=False).iloc[0]
    table.sort_values(["tune_robust_score","tune_sum"], ascending=False).to_csv(SESSION/"rolling_policy_search.csv", index=False)
    payload = {"experiment_count":len(table), "selection_folds":TUNE, "untouched_gate_folds":HOLDOUT,
               "baseline_by_week":baseline, "baseline_tune_sum":sum(baseline[f] for f in TUNE),
               "baseline_holdout_sum":sum(baseline[f] for f in HOLDOUT), "winner":winner.to_dict(),
               "holdout_gate_passed":bool(winner["holdout_delta"] > 0 and winner["w5_pnl"] > 0 and winner["w6_pnl"] > 0),
               "top20_tune_only":table.sort_values(["tune_robust_score","tune_sum"], ascending=False).head(20).to_dict("records")}
    (SESSION/"rolling_policy_summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__ == "__main__":
    main()
