#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = Path(__file__).resolve().parent
MODULE = SESSION / "run_joint_fill_value.py"
spec = importlib.util.spec_from_file_location("base", MODULE)
base = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(base)


def main() -> None:
    tune = {fold: base.fit_fold(fold) for fold in base.TUNE}
    rows = []
    for q_shrink in np.round(np.arange(-0.60, 0.051, 0.05), 2):
        for h_power in np.round(np.arange(0.70, 1.31, 0.10), 2):
            for h_scale in np.round(np.arange(0.40, 0.91, 0.05), 2):
                for min_ev in np.round(np.arange(-0.005, 0.031, 0.005), 3):
                    params = {"q_shrink": float(q_shrink), "h_power": float(h_power),
                              "h_scale": float(h_scale), "min_ev": float(min_ev)}
                    fold_pnl = {fold: base.pnl(tune[fold], base.choose(tune[fold], params)[0]) for fold in base.TUNE}
                    values = np.asarray(list(fold_pnl.values()))
                    rows.append({**params, **{f"{fold}_pnl": value for fold, value in fold_pnl.items()},
                                 "tune_sum": values.sum(), "tune_worst": values.min(),
                                 "positive_weeks": int((values > 0).sum()), "robust_score": values.sum() - values.std()})
    table = pd.DataFrame(rows).sort_values(["positive_weeks", "robust_score", "tune_sum"], ascending=False)
    table.to_csv(SESSION / "cycle5_joint_fill_refine.csv", index=False)
    winner = table.loc[table["positive_weeks"] >= 3].iloc[0].to_dict()
    params = {key: winner[key] for key in ["q_shrink", "h_power", "h_scale", "min_ev"]}
    holdout_data = {fold: base.fit_fold(fold) for fold in base.HOLDOUT}
    holdout = {fold: base.full_metrics(fold, holdout_data[fold], params) for fold in base.HOLDOUT}
    payload = {"experiment_count": len(table), "winner": winner, "winner_params": params,
               "holdout_metrics": holdout, "holdout_sum": float(sum(v["sum_pnl"] for v in holdout.values())),
               "holdout_gate_passed": bool(all(v["sum_pnl"] > 0 for v in holdout.values())),
               "top20": table.head(20).to_dict("records")}
    (SESSION / "cycle5_joint_fill_refine_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
