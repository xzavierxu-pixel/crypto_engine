#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
PREV = ROOT / ".arbor/sessions/20260703_prefinal_rolling"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("direct_pnl_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
feat = load_module("direct_pnl_feat", SESSION / "run_feature_research.py")
jwin = load_module("direct_pnl_expand", SESSION / "run_joint_winfill.py")


def prepare_fold(fold: str) -> dict:
    d = PREV / "folds" / fold
    train_path, dev_path = d / "data/train.parquet", d / "data/dev.parquet"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    lookup = feat.load_trades([train, dev])
    tr_trade, dv_trade = feat.engineer_trades(train, lookup), feat.engineer_trades(dev, lookup)
    _, prep, grid, hazard = joint.load_hazard(d / "models/hazard_survival_cdf.pt")
    gc_train_all = joint.predict_hazard(hazard, prep.transform(train), torch.device("cpu"), 512)[1]
    gc_dev_all = joint.predict_hazard(hazard, prep.transform(dev), torch.device("cpu"), 512)[1]
    tr_mask = train["threshold_accepted"].astype(bool).to_numpy()
    dv_mask = dev["threshold_accepted"].astype(bool).to_numpy()
    tr, dv = train.loc[tr_mask].reset_index(drop=True), dev.loc[dv_mask].reset_index(drop=True)
    tr_trade, dv_trade = tr_trade.loc[tr_mask].reset_index(drop=True), dv_trade.loc[dv_mask].reset_index(drop=True)
    gc_train, gc_dev = gc_train_all[tr_mask], gc_dev_all[dv_mask]
    sample_grid = np.unique(np.r_[np.arange(0, len(grid), 3), len(grid) - 1])
    xtr = jwin.expand(gc_train, grid, tr, tr_trade, np.arange(len(tr)), sample_grid)
    bids = grid[sample_grid]
    correct = tr["correct"].astype(bool).to_numpy()
    low = tr["chosen_low"].to_numpy(float)
    filled = (~correct[:, None]) | (correct[:, None] & (low[:, None] <= bids[None, :]))
    pnl = np.zeros((len(tr), len(bids)), dtype="float32")
    pnl[filled & correct[:, None]] = np.broadcast_to(1.0 - bids[None, :], pnl.shape)[filled & correct[:, None]]
    pnl[filled & ~correct[:, None]] = np.broadcast_to(-bids[None, :], pnl.shape)[filled & ~correct[:, None]]
    xdv = jwin.expand(gc_dev, grid, dv, dv_trade, np.arange(len(dv)), np.arange(len(grid)))
    predictions = {}
    specs = {
        "mean": {"objective": "regression_l1"},
        "huber": {"objective": "huber", "alpha": 0.8},
        "q35": {"objective": "quantile", "alpha": 0.35},
    }
    for name, objective in specs.items():
        model = LGBMRegressor(
            n_estimators=500, learning_rate=0.025, num_leaves=15, max_depth=5,
            min_child_samples=250, subsample=0.8, colsample_bytree=0.55,
            reg_lambda=18, reg_alpha=4, random_state=20260703, verbosity=-1, n_jobs=-1,
            **objective,
        )
        model.fit(xtr, pnl.reshape(-1))
        predictions[name] = model.predict(xdv).reshape(len(dv), len(grid))
    predictions["mean_huber"] = 0.5 * predictions["mean"] + 0.5 * predictions["huber"]
    return {"dev": dev, "accepted": dv, "grid": grid, "predictions": predictions}


def evaluate(p: dict, model: str, max_bid: float, min_pred: float) -> dict:
    allowed = p["grid"] <= max_bid + 1e-9
    grid = p["grid"][allowed]
    pred = p["predictions"][model][:, allowed]
    idx = np.argmax(pred, axis=1)
    rows = np.arange(len(p["accepted"]))
    score = pred[rows, idx]
    bid = grid[idx]
    submit = score >= min_pred
    bid = np.where(submit, bid, 0.0)
    result = joint.backtest_with_bid(p["accepted"], bid, np.where(submit, score, 0.0), None)
    return joint.backtest_metrics(p["accepted"], result, len(p["dev"]))


def main() -> None:
    prepared = {f: prepare_fold(f) for f in FOLDS}
    rows = []
    for model in prepared["w1"]["predictions"]:
        for max_bid in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85):
            for min_pred in (-0.01, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
                metrics = {f: evaluate(prepared[f], model, max_bid, min_pred) for f in FOLDS}
                pnl = {f: metrics[f]["sum_pnl"] for f in FOLDS}
                tune = np.array([pnl[f] for f in TUNE])
                rows.append({"model": model, "max_bid": max_bid, "min_pred": min_pred,
                             **{f"{f}_pnl": pnl[f] for f in FOLDS},
                             "tune_sum": float(tune.sum()), "tune_worst": float(tune.min()),
                             "tune_robust": float(tune.sum() - tune.std()),
                             "holdout_sum": float(sum(pnl[f] for f in HOLDOUT)),
                             "holdout_worst": float(min(pnl[f] for f in HOLDOUT)),
                             "mean_bid_tune": float(np.mean([metrics[f]["mean_bid"] for f in TUNE]))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0]
    table.to_csv(SESSION / "direct_pnl_search.csv", index=False)
    summary = {"experiment_count": len(table), "winner": winner.to_dict(), "btest_used": False,
               "top20": table.head(20).to_dict("records")}
    (SESSION / "direct_pnl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
