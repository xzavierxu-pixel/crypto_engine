#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
PREV = ROOT / ".arbor/sessions/20260703_prefinal_rolling"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("joint_gc", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
features = load_module("feature_research", SESSION / "run_feature_research.py")
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]


def gc_matrix(base_gc: np.ndarray, bid_grid: np.ndarray, frame: pd.DataFrame,
              trade: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    n, m = len(indices), len(bid_grid)
    static = np.c_[
        frame.iloc[indices]["p_side"].to_numpy(float),
        (frame.iloc[indices]["selected_side"].astype(str).str.upper() == "UP").to_numpy(float),
        trade.iloc[indices].to_numpy(float),
    ].astype("float32")
    return np.c_[
        np.tile(bid_grid, n),
        base_gc[indices][:, np.searchsorted(features.np.asarray(bid_grid), bid_grid)].reshape(-1),
        np.repeat(static, m, axis=0),
    ].astype("float32")


def expand(base_gc: np.ndarray, grid: np.ndarray, frame: pd.DataFrame,
           trade: pd.DataFrame, indices: np.ndarray, selected_grid_indices: np.ndarray) -> np.ndarray:
    n, m = len(indices), len(selected_grid_indices)
    static = np.c_[
        frame.iloc[indices]["p_side"].to_numpy(float),
        (frame.iloc[indices]["selected_side"].astype(str).str.upper() == "UP").to_numpy(float),
        trade.iloc[indices].to_numpy(float),
    ].astype("float32")
    return np.c_[
        np.tile(grid[selected_grid_indices], n),
        base_gc[indices][:, selected_grid_indices].reshape(-1),
        np.repeat(static, m, axis=0),
    ].astype("float32")


def prepare_fold(fold: str) -> dict:
    d = PREV / "folds" / fold
    train_path, dev_path = d / "data/train.parquet", d / "data/dev.parquet"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    prepared = joint.prepare(train_path, dev_path, d / "models/hazard_survival_cdf.pt", 20260703)
    lookup = features.load_trades([train, dev])
    tr_trade, dv_trade = features.engineer_trades(train, lookup), features.engineer_trades(dev, lookup)
    hazard_cols = joint.load_hazard(d / "models/hazard_survival_cdf.pt")[0]
    q_trade = features.fit_predict(
        pd.concat([train, tr_trade], axis=1), pd.concat([dev, dv_trade], axis=1),
        hazard_cols + list(tr_trade.columns), 20260703,
    )
    q = 0.5 * prepared[2]["raw_tree_blend"] + 0.5 * q_trade

    _, prep, grid, hazard = joint.load_hazard(d / "models/hazard_survival_cdf.pt")
    gc_train_all = joint.predict_hazard(hazard, prep.transform(train), torch.device("cpu"), 512)[1]
    gc_dev = prepared[3]
    tr_mask = train["threshold_accepted"].astype(bool).to_numpy()
    dv_mask = dev["threshold_accepted"].astype(bool).to_numpy()
    tr = train.loc[tr_mask].reset_index(drop=True)
    dv = dev.loc[dv_mask].reset_index(drop=True)
    tr_trade = tr_trade.loc[tr_mask].reset_index(drop=True)
    dv_trade = dv_trade.loc[dv_mask].reset_index(drop=True)
    gc_train = gc_train_all[tr_mask]

    correct_train = np.flatnonzero(tr["correct"].astype(bool).to_numpy())
    sample_grid = np.unique(np.r_[np.arange(0, len(grid), 4), len(grid) - 1])
    xtr = expand(gc_train, grid, tr, tr_trade, correct_train, sample_grid)
    lows = tr.iloc[correct_train]["chosen_low"].to_numpy(float)
    ytr = (lows[:, None] <= grid[sample_grid][None, :]).astype("uint8").reshape(-1)
    monotone = [1, 1] + [0] * (xtr.shape[1] - 2)
    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=5,
        min_child_samples=200, subsample=0.8, colsample_bytree=0.5,
        reg_lambda=15, reg_alpha=3, random_state=20260703, verbosity=-1,
        n_jobs=-1, monotone_constraints=monotone,
    )
    model.fit(xtr, ytr)
    all_dev = np.arange(len(dv))
    xdv = expand(gc_dev, grid, dv, dv_trade, all_dev, np.arange(len(grid)))
    gc_new = model.predict_proba(xdv)[:, 1].reshape(len(dv), len(grid))
    gc_new = np.maximum.accumulate(np.clip(gc_new, 0.0, 1.0), axis=1)

    correct_dev = dv["correct"].astype(bool).to_numpy()
    targets = (dv.loc[correct_dev, "chosen_low"].to_numpy(float)[:, None] <= grid[None, :]).astype(float)
    return {
        "prepared": prepared, "q": q, "gc_base": gc_dev, "gc_new": gc_new, "grid": grid,
        "gc_brier_base": float(np.mean((gc_dev[correct_dev] - targets) ** 2)),
        "gc_brier_new": float(np.mean((gc_new[correct_dev] - targets) ** 2)),
    }


def evaluate(payload: dict, gc_name: str, floor: float, min_ev: float) -> dict:
    dev, accepted = payload["prepared"][0], payload["prepared"][1]
    gc = payload[gc_name]
    bid, ev, fill = joint.choose_survival_expected_return_bids(
        payload["q"], gc, payload["grid"], 0.01, min_ev, min_fill_probability=floor,
    )
    return joint.backtest_metrics(accepted, joint.backtest_with_bid(accepted, bid, ev, fill), len(dev))


def main() -> None:
    prepared = {f: prepare_fold(f) for f in FOLDS}
    rows = []
    for gc_name in ("gc_base", "gc_new"):
        for floor in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            for min_ev in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
                pnl = {f: evaluate(prepared[f], gc_name, floor, min_ev)["sum_pnl"] for f in FOLDS}
                tune = np.array([pnl[f] for f in TUNE])
                rows.append({"gc_model": gc_name, "gc_floor": floor, "min_ev": min_ev,
                             **{f"{f}_pnl": pnl[f] for f in FOLDS}, "tune_sum": float(tune.sum()),
                             "tune_worst": float(tune.min()), "tune_robust": float(tune.sum() - tune.std()),
                             "holdout_sum": float(sum(pnl[f] for f in HOLDOUT))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0]
    table.to_csv(SESSION / "gc_feature_policy_search.csv", index=False)
    summary = {
        "experiment_count": len(table), "winner": winner.to_dict(),
        "gc_brier": {f: {"base": p["gc_brier_base"], "new": p["gc_brier_new"]} for f, p in prepared.items()},
        "all_fold_gc_brier_improved": all(p["gc_brier_new"] < p["gc_brier_base"] for p in prepared.values()),
        "btest_used": False,
    }
    (SESSION / "gc_feature_research_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
