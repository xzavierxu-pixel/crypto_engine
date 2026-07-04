#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
OLD = ROOT / ".arbor/sessions/20260703_sum_pnl_no_leak"
ER = ROOT / "price_estimator/expected_return"
sys.path[:0] = [str(ROOT / "price_estimator/scripts"), str(ROOT / "price_estimator/upper_bound_mlp"), str(ER)]

from train_upper_bound_mlp import Preprocessor  # noqa: E402
from train_low_cdf_and_backtest import (  # noqa: E402
    HazardMLP, backtest_metrics, backtest_with_bid,
    choose_survival_expected_return_bids, predict_hazard,
)

FORBIDDEN_EXACT = {
    "target", "abs_return", "signed_return", "stage1_target", "stage2_target",
    "stage1_sample_weight", "chosen_low", "correct", "winner", "pnl",
    "trade_time", "endDate", "condition_id", "market_id", "slug", "outcome",
}
BASELINES = {"main": 88.38, "earlier": 82.34}


def load_hazard(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cols = list(ck["feature_columns"])
    bad = [c for c in cols if c in FORBIDDEN_EXACT or c.startswith("future_") or "sample_weight" in c.lower()]
    if bad:
        raise RuntimeError(f"forbidden checkpoint features: {bad}")
    prep = Preprocessor(**ck["preprocessor"])
    grid = np.asarray(ck["tick_grid"], dtype=float)
    spec = ck["model"]
    model = HazardMLP(len(prep.output_columns), [int(v) for v in spec["hidden_dims"]], [float(v) for v in spec["dropout"]], len(grid))
    model.load_state_dict(ck["state_dict"])
    return cols, prep, grid, model


def matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df.reindex(columns=cols).copy()
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def prepare(train_path: Path, dev_path: Path, hazard_path: Path, seed: int):
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    cols, prep, grid, hazard = load_hazard(hazard_path)
    accepted_train = train.loc[train["threshold_accepted"].astype(bool)].copy()
    accepted_dev = dev.loc[dev["threshold_accepted"].astype(bool)].copy()
    xtr, xdv = matrix(accepted_train, cols), matrix(accepted_dev, cols)
    ytr = accepted_train["correct"].astype(int).to_numpy()
    raw_train = accepted_train["p_side"].to_numpy(float)
    raw_dev = accepted_dev["p_side"].to_numpy(float)

    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw_train, ytr)
    logit = LogisticRegression(C=0.2, max_iter=500).fit(np.c_[raw_train, np.abs(raw_train - 0.5)], ytr)
    lgb = LGBMClassifier(n_estimators=350, learning_rate=.025, num_leaves=15, max_depth=5,
                         min_child_samples=120, subsample=.8, colsample_bytree=.35,
                         reg_lambda=8, reg_alpha=2, random_state=seed, verbosity=-1, n_jobs=-1)
    lgb.fit(xtr, ytr)
    cat = CatBoostClassifier(iterations=350, depth=5, learning_rate=.035, loss_function="Logloss",
                             l2_leaf_reg=10, random_seed=seed, verbose=False, thread_count=-1,
                             random_strength=1.0, allow_writing_files=False)
    cat.fit(xtr, ytr)

    q = {
        "raw": raw_dev,
        "isotonic": np.asarray(iso.predict(raw_dev)),
        "logistic": logit.predict_proba(np.c_[raw_dev, np.abs(raw_dev - 0.5)])[:, 1],
        "lgbm": lgb.predict_proba(xdv)[:, 1],
        "catboost": cat.predict_proba(xdv)[:, 1],
    }
    q["tree_blend"] = .5 * q["lgbm"] + .5 * q["catboost"]
    q["raw_tree_blend"] = .5 * q["raw"] + .25 * q["lgbm"] + .25 * q["catboost"]
    gc_all = predict_hazard(hazard, prep.transform(dev), torch.device("cpu"), 512)[1]
    gc = gc_all[dev["threshold_accepted"].astype(bool).to_numpy()]
    calibration = {k: {"brier": float(brier_score_loss(accepted_dev["correct"], v)),
                       "log_loss": float(log_loss(accepted_dev["correct"], np.clip(v, 1e-6, 1-1e-6)))} for k, v in q.items()}
    return dev, accepted_dev, q, gc, grid, calibration, len(cols)


def evaluate(prepared, q_name: str, shrink: float, floor: float, min_ev: float):
    dev, accepted, qs, gc, grid, _, _ = prepared
    q = (1 - shrink) * qs[q_name] + shrink * 0.5
    bid, ev, fill = choose_survival_expected_return_bids(q, gc, grid, 0.01, min_ev, min_fill_probability=floor)
    return backtest_metrics(accepted, backtest_with_bid(accepted, bid, ev, fill), len(dev))


def dev_search():
    folds = {
        "main": (OLD/"bdev_data/train_before_20260328.parquet", OLD/"bdev_data/dev_20260328_20260410.parquet", OLD/"bdev_h2/models/hazard_survival_cdf.pt"),
        "earlier": (OLD/"experiments/3.1/data/train.parquet", OLD/"experiments/3.1/data/dev.parquet", OLD/"experiments/3.1/h2/models/hazard_survival_cdf.pt"),
    }
    prepared = {k: prepare(*v, seed=20260703) for k, v in folds.items()}
    rows = []
    for q_name in prepared["main"][2]:
        for shrink in [0.0, .1, .2]:
            for floor in [.65, .70, .75, .80, .85, .90]:
                for min_ev in [0.0, .005, .01, .02]:
                    ms = {f: evaluate(p, q_name, shrink, floor, min_ev) for f, p in prepared.items()}
                    ds = {f: ms[f]["sum_pnl"] - BASELINES[f] for f in ms}
                    rows.append({"q_model": q_name, "shrink_to_half": shrink, "gc_floor": floor, "min_ev": min_ev,
                                 **{f"{f}_sum_pnl": ms[f]["sum_pnl"] for f in ms},
                                 **{f"{f}_delta": ds[f] for f in ds}, "worst_delta": min(ds.values()), "mean_delta": np.mean(list(ds.values()))})
    table = pd.DataFrame(rows).sort_values(["worst_delta", "mean_delta"], ascending=False).reset_index(drop=True)
    table.to_csv(SESSION/"dev_search.csv", index=False)
    payload = {"experiment_count": len(table), "winner": table.iloc[0].to_dict(), "top20": table.head(20).to_dict("records"),
               "calibration": {f: p[5] for f, p in prepared.items()}, "feature_counts": {f: p[6] for f,p in prepared.items()},
               "leakage_guard": "passed"}
    (SESSION/"dev_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["winner"], indent=2))


def btest_once():
    winner = json.loads((SESSION/"dev_summary.json").read_text(encoding="utf-8"))["winner"]
    prepared = prepare(
        ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet",
        ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet",
        ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt",
        seed=20260703,
    )
    metrics = evaluate(prepared, winner["q_model"], float(winner["shrink_to_half"]), float(winner["gc_floor"]), float(winner["min_ev"]))
    payload = {"candidate_source": "robust two-fold B_dev winner", "baseline_sum_pnl": 27.44,
               "winner": winner, "metrics": metrics, "calibration": prepared[5][winner["q_model"]],
               "btest_evaluation_count": 1}
    (SESSION/"final_btest_once.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--btest-once", action="store_true")
    args = parser.parse_args()
    btest_once() if args.btest_once else dev_search()
