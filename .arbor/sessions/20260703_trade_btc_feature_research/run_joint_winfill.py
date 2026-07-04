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
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("joint_winfill_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
feat = load_module("joint_winfill_features", SESSION / "run_feature_research.py")


def expand(base_gc: np.ndarray, grid: np.ndarray, frame: pd.DataFrame,
           trade: pd.DataFrame, row_idx: np.ndarray, grid_idx: np.ndarray) -> np.ndarray:
    n, m = len(row_idx), len(grid_idx)
    p = frame.iloc[row_idx]["p_side"].to_numpy(float)
    static = np.c_[
        p,
        (frame.iloc[row_idx]["selected_side"].astype(str).str.upper() == "UP").to_numpy(float),
        trade.iloc[row_idx].to_numpy(float),
    ].astype("float32")
    chosen_gc = base_gc[row_idx][:, grid_idx]
    bids = grid[grid_idx]
    return np.c_[
        np.tile(bids, n),
        chosen_gc.reshape(-1),
        (p[:, None] * chosen_gc).reshape(-1),
        np.repeat(static, m, axis=0),
    ].astype("float32")


def prepare_fold(fold: str) -> dict:
    d = PREV / "folds" / fold
    train_path, dev_path = d / "data/train.parquet", d / "data/dev.parquet"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    prepared = joint.prepare(train_path, dev_path, d / "models/hazard_survival_cdf.pt", 20260703)
    lookup = feat.load_trades([train, dev])
    tr_trade, dv_trade = feat.engineer_trades(train, lookup), feat.engineer_trades(dev, lookup)
    hazard_columns, prep, grid, hazard = joint.load_hazard(d / "models/hazard_survival_cdf.pt")
    q_trade = feat.fit_predict(
        pd.concat([train, tr_trade], axis=1), pd.concat([dev, dv_trade], axis=1),
        hazard_columns + list(tr_trade.columns), 20260703,
    )
    q = 0.5 * prepared[2]["raw_tree_blend"] + 0.5 * q_trade
    gc_train_all = joint.predict_hazard(hazard, prep.transform(train), torch.device("cpu"), 512)[1]
    tr_mask = train["threshold_accepted"].astype(bool).to_numpy()
    dv_mask = dev["threshold_accepted"].astype(bool).to_numpy()
    tr = train.loc[tr_mask].reset_index(drop=True)
    dv = dev.loc[dv_mask].reset_index(drop=True)
    tr_trade = tr_trade.loc[tr_mask].reset_index(drop=True)
    dv_trade = dv_trade.loc[dv_mask].reset_index(drop=True)
    gc_train, gc_dev = gc_train_all[tr_mask], prepared[3]

    sample_grid = np.unique(np.r_[np.arange(0, len(grid), 4), len(grid) - 1])
    rows = np.arange(len(tr))
    xtr = expand(gc_train, grid, tr, tr_trade, rows, sample_grid)
    correct = tr["correct"].astype(bool).to_numpy()
    low = tr["chosen_low"].to_numpy(float)
    y = (correct[:, None] & (low[:, None] <= grid[sample_grid][None, :])).astype("uint8").reshape(-1)
    monotone = [1, 1, 1] + [0] * (xtr.shape[1] - 3)
    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.025, num_leaves=15, max_depth=5,
        min_child_samples=250, subsample=0.8, colsample_bytree=0.55,
        reg_lambda=18, reg_alpha=4, random_state=20260703, verbosity=-1,
        n_jobs=-1, monotone_constraints=monotone,
    )
    model.fit(xtr, y)
    all_rows, all_grid = np.arange(len(dv)), np.arange(len(grid))
    xdv = expand(gc_dev, grid, dv, dv_trade, all_rows, all_grid)
    r = model.predict_proba(xdv)[:, 1].reshape(len(dv), len(grid))
    r = np.maximum.accumulate(np.clip(r, 0.0, 1.0), axis=1)
    r = np.minimum(r, q[:, None])
    target = (
        dv["correct"].astype(bool).to_numpy()[:, None]
        & (dv["chosen_low"].to_numpy(float)[:, None] <= grid[None, :])
    ).astype(float)
    base_r = q[:, None] * gc_dev
    return {
        "prepared": prepared, "q": q, "r": r, "base_r": base_r, "grid": grid,
        "joint_brier": float(np.mean((r - target) ** 2)),
        "base_joint_brier": float(np.mean((base_r - target) ** 2)),
    }


def evaluate(payload: dict, r_name: str, fill_floor: float, min_ev: float) -> dict:
    dev, accepted = payload["prepared"][0], payload["prepared"][1]
    q, r, grid = payload["q"], payload[r_name], payload["grid"]
    ev_grid = r * (1.0 - grid[None, :]) - (1.0 - q[:, None]) * grid[None, :]
    idx = np.argmax(ev_grid, axis=1)
    rows = np.arange(len(q))
    bid = grid[idx].astype(float)
    ev = ev_grid[rows, idx]
    conditional_fill = np.divide(r[rows, idx], q, out=np.zeros_like(q), where=q > 1e-8)
    submit = (ev >= min_ev) & (conditional_fill >= fill_floor)
    bid = np.where(submit, bid, 0.0)
    return joint.backtest_metrics(
        accepted,
        joint.backtest_with_bid(accepted, bid, np.where(submit, ev, 0.0), np.where(submit, conditional_fill, 0.0)),
        len(dev),
    )


def main() -> None:
    prepared = {f: prepare_fold(f) for f in FOLDS}
    rows = []
    for r_name in ("base_r", "r"):
        for floor in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            for min_ev in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075):
                pnl = {f: evaluate(prepared[f], r_name, floor, min_ev)["sum_pnl"] for f in FOLDS}
                tune = np.array([pnl[f] for f in TUNE])
                rows.append({"r_model": r_name, "fill_floor": floor, "min_ev": min_ev,
                             **{f"{f}_pnl": pnl[f] for f in FOLDS},
                             "tune_sum": float(tune.sum()), "tune_worst": float(tune.min()),
                             "tune_robust": float(tune.sum() - tune.std()),
                             "holdout_sum": float(sum(pnl[f] for f in HOLDOUT))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0]
    table.to_csv(SESSION / "joint_winfill_search.csv", index=False)
    summary = {
        "experiment_count": len(table), "winner": winner.to_dict(),
        "calibration": {f: {"base_joint_brier": p["base_joint_brier"], "joint_brier": p["joint_brier"]} for f, p in prepared.items()},
        "all_fold_joint_brier_improved": all(p["joint_brier"] < p["base_joint_brier"] for p in prepared.values()),
        "btest_used": False,
    }
    (SESSION / "joint_winfill_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
