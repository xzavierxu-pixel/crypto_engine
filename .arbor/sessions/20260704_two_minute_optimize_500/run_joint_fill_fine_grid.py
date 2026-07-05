#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("base", SESSION / "run_joint_fill_value.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(base)
FINE_BIDS = np.round(np.arange(0.01, 0.851, 0.01), 2)


def fine_data(fold: str) -> dict:
    data = base.fit_fold(fold)
    data = dict(data)
    data["h_fine"] = np.vstack([np.interp(FINE_BIDS, base.BIDS, row, left=row[0], right=row[-1]) for row in data["h"]])
    return data


def choose(data: dict, params: dict):
    q = np.clip((1 - params["q_shrink"]) * data["q"] + params["q_shrink"] * 0.5, 0.001, 0.999)
    h = np.clip(params["h_scale"] * data["h_fine"] ** params["h_power"], 0, 1)
    ev = h * (1 - FINE_BIDS[None, :]) - (1 - q[:, None]) * FINE_BIDS[None, :]
    best = np.argmax(ev, axis=1)
    best_ev = ev[np.arange(len(ev)), best]
    bid = FINE_BIDS[best].copy()
    bid[best_ev < params["min_ev"]] = 0.0
    return bid, best_ev, h[np.arange(len(h)), best]


def full_metrics(fold: str, data: dict, params: dict) -> dict:
    _, dev, accepted, _ = base.load_frames(fold)
    bid, ev, fill = choose(data, params)
    return base.joint.backtest_metrics(accepted, base.joint.backtest_with_bid(accepted, bid, ev, fill), len(dev))


def main() -> None:
    tune = {fold: fine_data(fold) for fold in base.TUNE}
    rows = []
    for q_shrink in np.round(np.arange(-0.35, 0.01, 0.05), 2):
        for h_power in np.round(np.arange(0.75, 1.11, 0.05), 2):
            for h_scale in np.round(np.arange(0.55, 0.91, 0.05), 2):
                for min_ev in np.round(np.arange(-0.005, 0.021, 0.005), 3):
                    params = {"q_shrink": float(q_shrink), "h_power": float(h_power),
                              "h_scale": float(h_scale), "min_ev": float(min_ev)}
                    values = np.asarray([base.pnl(tune[fold], choose(tune[fold], params)[0]) for fold in base.TUNE])
                    rows.append({**params, **{f"{fold}_pnl": values[i] for i, fold in enumerate(base.TUNE)},
                                 "tune_sum": values.sum(), "tune_worst": values.min(),
                                 "positive_weeks": int((values > 0).sum()), "robust_score": values.sum() - values.std()})
    table = pd.DataFrame(rows).sort_values(["positive_weeks", "robust_score", "tune_sum"], ascending=False)
    table.to_csv(SESSION / "cycle6_fine_grid_search.csv", index=False)
    winner = table.loc[table["positive_weeks"] >= 3].iloc[0].to_dict()
    params = {key: winner[key] for key in ["q_shrink", "h_power", "h_scale", "min_ev"]}
    holdout_data = {fold: fine_data(fold) for fold in base.HOLDOUT}
    holdout = {fold: full_metrics(fold, holdout_data[fold], params) for fold in base.HOLDOUT}
    payload = {"winner": winner, "winner_params": params, "holdout_metrics": holdout,
               "holdout_sum": float(sum(v["sum_pnl"] for v in holdout.values())),
               "holdout_gate_passed": bool(all(v["sum_pnl"] > 0 for v in holdout.values())),
               "top20": table.head(20).to_dict("records")}
    (SESSION / "cycle6_fine_grid_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
