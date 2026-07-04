#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
SOURCE = ROOT / ".arbor/sessions/20260703_prefinal_rolling"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("full_direction_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")


def fit_fold(fold: str) -> dict:
    fd = SOURCE / "folds" / fold
    raw_train = pd.read_parquet(fd / "data/train.parquet", columns=["timestamp"])
    raw_dev = pd.read_parquet(fd / "data/dev.parquet", columns=["timestamp"])
    all_data = pd.read_parquet(SOURCE / "pretest_both_side_lows.parquet")
    ts = pd.to_datetime(all_data["timestamp"], utc=True)
    tr = all_data.loc[ts.isin(pd.to_datetime(raw_train["timestamp"], utc=True))].copy()
    dv = all_data.loc[ts.isin(pd.to_datetime(raw_dev["timestamp"], utc=True))].copy()
    cols, prep, grid, hazard = joint.load_hazard(fd / "models/hazard_survival_cdf.pt")
    xtr, xdv = joint.matrix(tr, cols), joint.matrix(dv, cols)
    y = tr["target"].astype(int).to_numpy()
    lgb = LGBMClassifier(n_estimators=500, learning_rate=0.025, num_leaves=15, max_depth=5,
                         min_child_samples=150, colsample_bytree=0.4, reg_lambda=12, reg_alpha=3,
                         verbosity=-1, n_jobs=-1, random_state=20260703).fit(xtr, y)
    cat = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.03, l2_leaf_reg=12,
                             loss_function="Logloss", verbose=False, allow_writing_files=False,
                             random_seed=20260703).fit(xtr, y)
    raw = dv["p_up"].to_numpy(float)
    lp, cp = lgb.predict_proba(xdv)[:, 1], cat.predict_proba(xdv)[:, 1]
    tree = 0.5 * lp + 0.5 * cp
    probs = {"raw": raw, "lgbm": lp, "catboost": cp, "tree": tree}
    for a in (0.25, 0.5, 0.75):
        probs[f"raw_tree_{a}"] = (1 - a) * raw + a * tree
    prepared = {}
    for name, p in probs.items():
        side = np.where(p >= 0.5, "UP", "DOWN")
        work = dv.copy()
        original = work["selected_side"].astype(str).str.upper().to_numpy()
        work["selected_side"] = side
        work["p_up"] = p
        work["p_side"] = np.maximum(p, 1 - p)
        work["direction_confidence"] = np.abs(p - 0.5)
        work["correct"] = side == np.where(work["target"].astype(int).to_numpy() == 1, "UP", "DOWN")
        alternative = np.where(side == "UP", work["up_low"], work["down_low"])
        work["chosen_low"] = np.where(side == original, work["chosen_low"], alternative)
        gc = joint.predict_hazard(hazard, prep.transform(work), torch.device("cpu"), 512)[1]
        prepared[name] = (work, gc, grid)
    return prepared


def evaluate(payload: tuple, min_q: float, floor: float, min_ev: float) -> dict:
    work, gc, grid = payload
    q = work["p_side"].to_numpy(float)
    bid, ev, fill = joint.choose_survival_expected_return_bids(q, gc, grid, 0.01, min_ev,
                                                               min_fill_probability=floor)
    bid[(q < min_q) | ~np.isfinite(work["chosen_low"].to_numpy(float))] = 0.0
    return joint.backtest_metrics(work, joint.backtest_with_bid(work, bid, ev, fill), len(work))


def main() -> None:
    folds = {f: fit_fold(f) for f in FOLDS}
    rows = []
    for direction in folds["w1"]:
      for min_q in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
       for floor in (0.70, 0.75, 0.80, 0.85, 0.90):
        for min_ev in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10):
            ms = {f: evaluate(folds[f][direction], min_q, floor, min_ev) for f in FOLDS}
            pnl = {f: ms[f]["sum_pnl"] for f in FOLDS}
            tune = np.array([pnl[f] for f in TUNE])
            rows.append({"direction_model": direction, "min_q": min_q, "gc_floor": floor, "min_ev": min_ev,
                         **{f"{f}_pnl": pnl[f] for f in FOLDS},
                         **{f"{f}_accuracy": ms[f]["accepted_sample_accuracy"] for f in FOLDS},
                         **{f"{f}_order_coverage": ms[f]["order_coverage"] for f in FOLDS},
                         "tune_sum": float(tune.sum()), "tune_worst": float(tune.min()),
                         "tune_robust": float(tune.sum() - tune.std()),
                         "holdout_sum": float(sum(pnl[f] for f in HOLDOUT)),
                         "holdout_worst": float(min(pnl[f] for f in HOLDOUT))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0]
    table.to_csv(SESSION / "full_universe_direction_search.csv", index=False)
    summary = {"experiment_count": len(table), "winner": winner.to_dict(), "direction_coverage": 1.0,
               "btest_used": False, "top20": table.head(20).to_dict("records")}
    (SESSION / "full_universe_direction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
