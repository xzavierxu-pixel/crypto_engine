#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
SOURCE = ROOT / ".arbor/sessions/20260704_two_minute_shift"
JOINT_PATH = ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
spec = importlib.util.spec_from_file_location("joint", JOINT_PATH)
joint = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(joint)

TUNE = [f"w{i}" for i in range(1, 5)]
HOLDOUT = ["w5", "w6"]
THRESHOLDS = np.round(np.arange(0.50, 0.901, 0.005), 3)
BIDS = np.round(np.arange(0.01, 0.851, 0.01), 2)


def prepared(fold: str):
    d = SOURCE / "folds" / fold
    return joint.prepare(d / "data/train.parquet", d / "data/dev.parquet", d / "models/hazard_survival_cdf.pt", 20260704)


def returns(accepted: pd.DataFrame) -> np.ndarray:
    correct = accepted["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(accepted["chosen_low"], errors="coerce").to_numpy(float)
    fill = correct[:, None] & np.isfinite(low[:, None]) & (low[:, None] <= BIDS[None, :] + 1e-12)
    return fill * (1.0 - BIDS[None, :]) - (~correct)[:, None] * BIDS[None, :]


def grid(prep: tuple, q_model: str, shrink: float) -> np.ndarray:
    accepted, qs = prep[1], prep[2]
    q = np.clip((1 - shrink) * qs[q_model] + shrink * 0.5, 0, 1)
    value = returns(accepted)
    return np.stack([value[q >= threshold].sum(axis=0) for threshold in THRESHOLDS])


def metrics(prep: tuple, params: dict) -> dict:
    dev, accepted, qs = prep[0], prep[1], prep[2]
    q = np.clip((1 - params["shrink"]) * qs[params["q_model"]] + params["shrink"] * 0.5, 0, 1)
    bid = np.where(q >= params["q_threshold"], params["bid"], 0.0)
    return joint.backtest_metrics(accepted, joint.backtest_with_bid(accepted, bid), len(dev))


def main() -> None:
    tune = {fold: prepared(fold) for fold in TUNE}
    rows = []
    for q_model in tune["w1"][2]:
        for shrink in [-0.30, -0.15, 0.0, 0.15, 0.30]:
            cube = np.stack([grid(tune[fold], q_model, shrink) for fold in TUNE])
            sums = cube.sum(axis=0)
            std = cube.std(axis=0)
            positive = (cube > 0).sum(axis=0)
            robust = sums - std
            for ti, bi in np.ndindex(sums.shape):
                rows.append({"q_model": q_model, "shrink": shrink, "q_threshold": THRESHOLDS[ti], "bid": BIDS[bi],
                             **{f"{fold}_pnl": cube[index, ti, bi] for index, fold in enumerate(TUNE)},
                             "tune_sum": sums[ti, bi], "tune_worst": cube[:, ti, bi].min(),
                             "positive_weeks": int(positive[ti, bi]), "robust_score": robust[ti, bi]})
    table = pd.DataFrame(rows).sort_values(["positive_weeks", "robust_score", "tune_sum"], ascending=False)
    table.to_csv(SESSION / "cycle3_q_fixed_bid_search.csv", index=False)
    winner = table.loc[table["positive_weeks"] >= 3].iloc[0].to_dict()
    params = {key: winner[key] for key in ["q_model", "shrink", "q_threshold", "bid"]}
    holdout = {fold: metrics(prepared(fold), params) for fold in HOLDOUT}
    payload = {"experiment_count": len(table), "winner": winner, "winner_params": params,
               "holdout_metrics": holdout, "holdout_sum": float(sum(v["sum_pnl"] for v in holdout.values())),
               "holdout_gate_passed": bool(all(v["sum_pnl"] > 0 for v in holdout.values())),
               "top20": table.head(20).to_dict("records")}
    (SESSION / "cycle3_q_fixed_bid_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
