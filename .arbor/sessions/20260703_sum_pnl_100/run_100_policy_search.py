#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
OLD = ROOT / ".arbor" / "sessions" / "20260703_sum_pnl_no_leak"
ER = ROOT / "price_estimator" / "expected_return"
sys.path[:0] = [str(ROOT / "price_estimator" / "scripts"), str(ROOT / "price_estimator" / "upper_bound_mlp"), str(ER)]

from train_upper_bound_mlp import Preprocessor  # noqa: E402
from train_low_cdf_and_backtest import (  # noqa: E402
    HazardMLP, backtest_metrics, backtest_with_bid,
    choose_survival_expected_return_bids, predict_hazard,
)

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FLOORS = [0.65, 0.70, 0.75, 0.80, 0.85]
MIN_EVS = [0.0, 0.01, 0.02, 0.03]
BASELINES = {"main": 88.38, "earlier": 82.34}


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    forbidden = [c for c in ck["feature_columns"] if "sample_weight" in c.lower()]
    if forbidden:
        raise RuntimeError(f"Forbidden sample-weight features: {forbidden}")
    prep = Preprocessor(**ck["preprocessor"])
    grid = np.asarray(ck["tick_grid"], dtype=float)
    spec = ck["model"]
    model = HazardMLP(len(prep.output_columns), [int(v) for v in spec["hidden_dims"]], [float(v) for v in spec["dropout"]], len(grid))
    model.load_state_dict(ck["state_dict"])
    return prep, grid, model, len(ck["feature_columns"])


def prepare(train_path: Path, validation_path: Path, checkpoint_path: Path):
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    ts = pd.to_datetime(train["timestamp"], utc=True)
    calibration = train.loc[ts >= ts.max() - pd.Timedelta(days=7)].copy()
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        calibration["p_side"].to_numpy(float), calibration["correct"].astype(float).to_numpy()
    )
    prep, grid, model, feature_count = load_model(checkpoint_path)
    gc = predict_hazard(model, prep.transform(validation), torch.device("cpu"), 512)[1]
    mask = validation["threshold_accepted"].astype(bool).to_numpy()
    accepted = validation.loc[mask].copy()
    raw_q = accepted["p_side"].to_numpy(float)
    iso_q = np.asarray(iso.predict(raw_q), dtype=float)
    return validation, accepted, gc[mask], raw_q, iso_q, grid, feature_count


def evaluate(prepared, alpha: float, floor: float, min_ev: float):
    validation, accepted, gc, raw_q, iso_q, grid, _ = prepared
    q = (1.0 - alpha) * raw_q + alpha * iso_q
    bid, ev, fill_prob = choose_survival_expected_return_bids(
        q, gc, grid, 0.01, min_ev, min_fill_probability=floor
    )
    result = backtest_with_bid(accepted, bid, ev, fill_prob)
    return backtest_metrics(accepted, result, len(validation))


def dev_search():
    fold_paths = {
        "main": (
            OLD / "bdev_data/train_before_20260328.parquet",
            OLD / "bdev_data/dev_20260328_20260410.parquet",
            OLD / "bdev_h2/models/hazard_survival_cdf.pt",
        ),
        "earlier": (
            OLD / "experiments/3.1/data/train.parquet",
            OLD / "experiments/3.1/data/dev.parquet",
            OLD / "experiments/3.1/h2/models/hazard_survival_cdf.pt",
        ),
    }
    prepared = {name: prepare(*paths) for name, paths in fold_paths.items()}
    rows = []
    experiment_id = 0
    for alpha in ALPHAS:
        for floor in FLOORS:
            for min_ev in MIN_EVS:
                experiment_id += 1
                metrics = {name: evaluate(data, alpha, floor, min_ev) for name, data in prepared.items()}
                deltas = {name: metrics[name]["sum_pnl"] - BASELINES[name] for name in metrics}
                rows.append({
                    "experiment_id": experiment_id,
                    "q_isotonic_alpha": alpha,
                    "candidate_gc_floor": floor,
                    "min_ev": min_ev,
                    "main_sum_pnl": metrics["main"]["sum_pnl"],
                    "earlier_sum_pnl": metrics["earlier"]["sum_pnl"],
                    "main_delta": deltas["main"],
                    "earlier_delta": deltas["earlier"],
                    "worst_delta": min(deltas.values()),
                    "mean_delta": float(np.mean(list(deltas.values()))),
                    "main_order_count": metrics["main"]["order_count"],
                    "earlier_order_count": metrics["earlier"]["order_count"],
                })
    table = pd.DataFrame(rows).sort_values(["worst_delta", "mean_delta"], ascending=False).reset_index(drop=True)
    table.to_csv(SESSION / "dev_search_100.csv", index=False)
    winner = table.iloc[0].to_dict()
    payload = {
        "experiment_count": len(rows),
        "selection_rule": "maximize worst chronological-fold delta, then mean delta",
        "winner": winner,
        "top10": table.head(10).to_dict(orient="records"),
        "feature_counts": {name: data[-1] for name, data in prepared.items()},
        "forbidden_feature_check": "passed: no sample_weight-named checkpoint feature",
    }
    (SESSION / "dev_search_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def btest_once():
    summary = json.loads((SESSION / "dev_search_summary.json").read_text(encoding="utf-8"))
    winner = summary["winner"]
    prepared = prepare(
        ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet",
        ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet",
        ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt",
    )
    metrics = evaluate(prepared, float(winner["q_isotonic_alpha"]), float(winner["candidate_gc_floor"]), float(winner["min_ev"]))
    payload = {"candidate_source": "dev_search_summary.json winner", "baseline_sum_pnl": 27.44, "metrics": metrics}
    (SESSION / "final_btest_once.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--btest-once", action="store_true")
    args = parser.parse_args()
    btest_once() if args.btest_once else dev_search()
