#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[3]
ER = ROOT / "price_estimator" / "expected_return"
sys.path[:0] = [str(ROOT / "price_estimator" / "scripts"), str(ROOT / "price_estimator" / "upper_bound_mlp"), str(ER)]

from train_upper_bound_mlp import Preprocessor  # noqa: E402
from train_low_cdf_and_backtest import (  # noqa: E402
    HazardMLP, backtest_metrics, backtest_with_bid,
    choose_survival_expected_return_bids, predict_hazard,
)

SESSION = Path(__file__).resolve().parent
OLD = ROOT / ".arbor" / "sessions" / "20260703_sum_pnl_no_leak"
MIN_EVS = [0.0, 0.01, 0.02, 0.03, 0.05]
FLOORS = [0.0, 0.50, 0.60, 0.70, 0.75, 0.80]


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    prep = Preprocessor(**ck["preprocessor"])
    grid = np.asarray(ck["tick_grid"], dtype=float)
    spec = ck["model"]
    model = HazardMLP(len(prep.output_columns), [int(x) for x in spec["hidden_dims"]], [float(x) for x in spec["dropout"]], len(grid))
    model.load_state_dict(ck["state_dict"])
    return prep, grid, model


def split_tail(train: pd.DataFrame):
    ts = pd.to_datetime(train["timestamp"], utc=True)
    tail = train.loc[ts >= ts.max() - pd.Timedelta(days=7)].sort_values("timestamp").copy()
    mid = len(tail) // 2
    return tail.iloc[:mid].copy(), tail.iloc[mid:].copy()


def fit_q(method: str, fit: pd.DataFrame):
    x = fit["p_side"].to_numpy(float)
    y = fit["correct"].astype(float).to_numpy()
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)
    base = np.asarray(iso.predict(x), float)
    if method in {"isotonic", "shrink", "clip"}:
        def predict(frame):
            raw = frame["p_side"].to_numpy(float)
            q = np.asarray(iso.predict(raw), float)
            if method == "shrink": q = 0.5 * q + 0.5 * raw
            if method == "clip": q = np.clip(q, 0.55, 0.85)
            return q
        return predict
    group = (pd.to_datetime(fit["timestamp"], utc=True).dt.hour // 6 if method == "hour_group" else np.floor(x / 0.05).astype(int))
    residual = y - base
    stats = pd.DataFrame({"g": group, "r": residual}).groupby("g").r.agg(["mean", "count"])
    offsets = (stats["mean"] * stats["count"] / (stats["count"] + 50.0)).to_dict()
    def predict(frame):
        raw = frame["p_side"].to_numpy(float)
        q = np.asarray(iso.predict(raw), float)
        groups = (pd.to_datetime(frame["timestamp"], utc=True).dt.hour // 6 if method == "hour_group" else np.floor(raw / 0.05).astype(int))
        return np.clip(q + np.asarray([offsets.get(int(g), 0.0) for g in groups]), 0.01, 0.99)
    return predict


def evaluate_fold(name: str, train_path: Path, dev_path: Path, checkpoint: Path, methods=None):
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    q_fit, policy = split_tail(train)
    prep, grid, model = load_model(checkpoint)
    frames = {"q_fit": q_fit, "policy": policy, "dev": dev}
    gc = {k: predict_hazard(model, prep.transform(v), torch.device("cpu"), 512)[1] for k, v in frames.items()}
    output = {}
    for method in (methods or ["isotonic", "shrink", "clip", "hour_group", "pbin_group"]):
        q_fn = fit_q(method, q_fit)
        def run(frame, matrix, floor, min_ev):
            mask = frame["threshold_accepted"].astype(bool).to_numpy()
            accepted = frame.loc[mask].copy()
            q = q_fn(accepted)
            bid, ev, fp = choose_survival_expected_return_bids(q, matrix[mask], grid, 0.01, min_ev, min_fill_probability=floor)
            return accepted, backtest_with_bid(accepted, bid, ev, fp)
        candidates = []
        for floor in FLOORS:
            for min_ev in MIN_EVS:
                accepted, result = run(policy, gc["policy"], floor, min_ev)
                m = backtest_metrics(accepted, result, len(policy))
                if m["order_count"] >= 100:
                    candidates.append((m["mean_accepted_pnl"], m["order_count"], floor, min_ev))
        _, _, floor, min_ev = max(candidates)
        accepted, result = run(dev, gc["dev"], floor, min_ev)
        metrics = backtest_metrics(accepted, result, len(dev))
        output[method] = {"selected_floor": floor, "selected_min_ev": min_ev, "metrics": metrics}
    return output


def main():
    folds = {
        "main": (OLD / "bdev_data/train_before_20260328.parquet", OLD / "bdev_data/dev_20260328_20260410.parquet", OLD / "bdev_h2/models/hazard_survival_cdf.pt"),
        "earlier": (OLD / "experiments/3.1/data/train.parquet", OLD / "experiments/3.1/data/dev.parquet", OLD / "experiments/3.1/h2/models/hazard_survival_cdf.pt"),
    }
    results = {name: evaluate_fold(name, *paths) for name, paths in folds.items()}
    (SESSION / "batch_results.json").write_text(json.dumps(results, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({f"{fold}/{m}": v["metrics"]["sum_pnl"] for fold, rows in results.items() for m, v in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
