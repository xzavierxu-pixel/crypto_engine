#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
SOURCE = ROOT / ".arbor/sessions/20260704_two_minute_shift"
JOINT_PATH = ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("joint", JOINT_PATH)
joint = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(joint)

FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE = FOLDS[:4]
HOLDOUT = FOLDS[4:]


def prepare(folds: list[str]) -> dict[str, tuple]:
    return {
        fold: joint.prepare(
            SOURCE / "folds" / fold / "data/train.parquet",
            SOURCE / "folds" / fold / "data/dev.parquet",
            SOURCE / "folds" / fold / "models/hazard_survival_cdf.pt",
            20260704,
        )
        for fold in folds
    }


def bid_result(prepared: tuple, params: dict):
    dev, accepted, qs, gc, grid, _, _ = prepared
    q = (1.0 - params["shrink"]) * qs[params["q_model"]] + params["shrink"] * 0.5
    q = np.clip(q, 0.001, 0.999)
    calibrated_gc = np.clip(gc, 0.0, 1.0) ** params["gc_power"]
    cap_mask = grid <= params["bid_cap"] + 1e-12
    local_grid = grid[cap_mask]
    local_gc = calibrated_gc[:, cap_mask]
    bid, ev, fill = joint.choose_survival_expected_return_bids(
        q,
        local_gc,
        local_grid,
        0.01,
        params["min_ev"],
        min_fill_probability=params["gc_floor"],
    )
    return dev, accepted, bid, ev, fill


def evaluate_pnl(prepared: tuple, params: dict) -> float:
    _, accepted, bid, _, _ = bid_result(prepared, params)
    correct = accepted["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(accepted["chosen_low"], errors="coerce").to_numpy(float)
    submitted = bid > 0
    filled = submitted & (~correct | (np.isfinite(low) & (low <= bid + 1e-12)))
    pnl = np.zeros(len(bid), dtype=float)
    pnl[filled & correct] = 1.0 - bid[filled & correct]
    pnl[filled & ~correct] = -bid[filled & ~correct]
    return float(pnl.sum())


def evaluate(prepared: tuple, params: dict) -> dict:
    dev, accepted, bid, ev, fill = bid_result(prepared, params)
    return joint.backtest_metrics(accepted, joint.backtest_with_bid(accepted, bid, ev, fill), len(dev))


def search() -> dict:
    prepared = prepare(TUNE)
    rows = []
    q_models = list(prepared["w1"][2])
    rng = np.random.default_rng(20260704)
    selected_indices = set(rng.choice(31_500, size=3_000, replace=False).tolist())
    candidate_index = -1
    for q_model in q_models:
        for shrink in [-0.25, 0.0, 0.15, 0.30, 0.45]:
            for gc_power in [0.75, 1.0, 1.5, 2.0, 3.0]:
                for gc_floor in [0.0, 0.50, 0.70, 0.80, 0.90]:
                    for min_ev in [-0.02, 0.0, 0.01, 0.02, 0.04, 0.07]:
                        for bid_cap in [0.20, 0.30, 0.40, 0.50, 0.65, 0.85]:
                            candidate_index += 1
                            if candidate_index not in selected_indices:
                                continue
                            params = {
                                "q_model": q_model,
                                "shrink": shrink,
                                "gc_power": gc_power,
                                "gc_floor": gc_floor,
                                "min_ev": min_ev,
                                "bid_cap": bid_cap,
                            }
                            pnl = np.asarray([evaluate_pnl(prepared[fold], params) for fold in TUNE])
                            rows.append(
                                {
                                    **params,
                                    **{f"{fold}_pnl": pnl[index] for index, fold in enumerate(TUNE)},
                                    "tune_sum": float(pnl.sum()),
                                    "tune_worst": float(pnl.min()),
                                    "tune_mean": float(pnl.mean()),
                                    "tune_std": float(pnl.std()),
                                    "positive_weeks": int((pnl > 0).sum()),
                                    "robust_score": float(pnl.sum() - pnl.std()),
                                }
                            )
    table = pd.DataFrame(rows).sort_values(
        ["positive_weeks", "robust_score", "tune_sum", "tune_worst"], ascending=False
    )
    table.to_csv(SESSION / "cycle1_policy_search.csv", index=False)
    eligible = table.loc[table["positive_weeks"] >= 3]
    winner = (eligible if not eligible.empty else table).iloc[0].to_dict()
    params = {key: winner[key] for key in ["q_model", "shrink", "gc_power", "gc_floor", "min_ev", "bid_cap"]}
    holdout_prepared = prepare(HOLDOUT)
    holdout_metrics = {fold: evaluate(holdout_prepared[fold], params) for fold in HOLDOUT}
    payload = {
        "experiment_count": int(len(table)),
        "selection_folds": TUNE,
        "holdout_folds": HOLDOUT,
        "winner": winner,
        "winner_params": params,
        "holdout_metrics": holdout_metrics,
        "holdout_sum": float(sum(value["sum_pnl"] for value in holdout_metrics.values())),
        "holdout_gate_passed": bool(all(value["sum_pnl"] > 0 for value in holdout_metrics.values())),
        "top20": table.head(20).to_dict("records"),
    }
    (SESSION / "cycle1_policy_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--node-id", default="1")
    args = parser.parse_args()
    if args.split == "test":
        raise RuntimeError("B_test is not available through routine policy search")
    print(json.dumps(search(), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
