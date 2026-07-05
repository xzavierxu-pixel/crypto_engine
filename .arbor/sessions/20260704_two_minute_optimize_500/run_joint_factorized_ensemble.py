#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = Path(__file__).resolve().parent
SOURCE = SESSION.parent / "20260704_two_minute_shift"

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module

base = load_module("base", SESSION / "run_joint_fill_value.py")
joint = load_module("joint", base.JOINT_PATH)
TUNE, HOLDOUT = base.TUNE, base.HOLDOUT


def data(fold: str) -> dict:
    direct = base.fit_fold(fold)
    d = SOURCE / "folds" / fold
    factor = joint.prepare(d / "data/train.parquet", d / "data/dev.parquet", d / "models/hazard_survival_cdf.pt", 20260704)
    grid = np.asarray(factor[4], dtype=float)
    indices = [int(np.argmin(np.abs(grid - bid))) for bid in base.BIDS]
    return {"direct": direct, "accepted": factor[1], "qs": factor[2], "gc": factor[3][:, indices],
            "sample_count": len(factor[0])}


def choose(item: dict, params: dict):
    direct = item["direct"]
    q_other = item["qs"][params["q_source"]]
    q = np.clip(params["q_weight"] * direct["q"] + (1 - params["q_weight"]) * q_other, 0.001, 0.999)
    conditional_joint = q[:, None] * np.clip(item["gc"], 0, 1) ** params["gc_power"]
    h = np.clip(params["h_scale"] * (params["direct_weight"] * direct["h"] +
                (1 - params["direct_weight"]) * conditional_joint), 0, 1)
    ev = h * (1 - base.BIDS[None, :]) - (1 - q[:, None]) * base.BIDS[None, :]
    best = np.argmax(ev, axis=1)
    best_ev = ev[np.arange(len(ev)), best]
    bid = base.BIDS[best].copy()
    bid[best_ev < params["min_ev"]] = 0
    return bid, best_ev, h[np.arange(len(h)), best]


def score(item: dict, params: dict) -> float:
    return base.pnl(item["direct"], choose(item, params)[0])


def metrics(item: dict, params: dict) -> dict:
    bid, ev, fill = choose(item, params)
    return joint.backtest_metrics(item["accepted"], joint.backtest_with_bid(item["accepted"], bid, ev, fill), item["sample_count"])


def main() -> None:
    tune = {fold: data(fold) for fold in TUNE}
    rows = []
    for q_source in ["raw", "lgbm", "catboost", "tree_blend", "raw_tree_blend"]:
        for q_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for direct_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for gc_power in [1.0, 1.5, 2.0]:
                    for h_scale in [0.65, 0.8, 0.95, 1.1]:
                        for min_ev in [0.0, 0.005, 0.01, 0.02, 0.04]:
                            params = {"q_source": q_source, "q_weight": q_weight, "direct_weight": direct_weight,
                                      "gc_power": gc_power, "h_scale": h_scale, "min_ev": min_ev}
                            values = np.asarray([score(tune[fold], params) for fold in TUNE])
                            rows.append({**params, **{f"{fold}_pnl": values[i] for i, fold in enumerate(TUNE)},
                                         "tune_sum": values.sum(), "tune_worst": values.min(),
                                         "positive_weeks": int((values > 0).sum()), "robust_score": values.sum() - values.std()})
    table = pd.DataFrame(rows).sort_values(["positive_weeks", "robust_score", "tune_sum"], ascending=False)
    table.to_csv(SESSION / "cycle7_joint_factorized_ensemble.csv", index=False)
    winner = table.loc[table["positive_weeks"] >= 3].iloc[0].to_dict()
    keys = ["q_source", "q_weight", "direct_weight", "gc_power", "h_scale", "min_ev"]
    params = {key: winner[key] for key in keys}
    holdout_data = {fold: data(fold) for fold in HOLDOUT}
    holdout = {fold: metrics(holdout_data[fold], params) for fold in HOLDOUT}
    payload = {"winner": winner, "winner_params": params, "holdout_metrics": holdout,
               "holdout_sum": float(sum(v["sum_pnl"] for v in holdout.values())),
               "holdout_gate_passed": bool(all(v["sum_pnl"] > 0 for v in holdout.values())),
               "top20": table.head(20).to_dict("records")}
    (SESSION / "cycle7_joint_factorized_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
