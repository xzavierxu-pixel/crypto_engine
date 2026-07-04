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


joint = load_module("online_direction_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")


def lgb_model() -> LGBMClassifier:
    return LGBMClassifier(n_estimators=350, learning_rate=0.03, num_leaves=15, max_depth=5,
                          min_child_samples=150, colsample_bytree=0.4, reg_lambda=12, reg_alpha=3,
                          verbosity=-1, n_jobs=-1, random_state=20260703)


def fit_fold(fold: str) -> dict:
    fd = SOURCE / "folds" / fold
    raw_train = pd.read_parquet(fd / "data/train.parquet", columns=["timestamp"])
    raw_dev = pd.read_parquet(fd / "data/dev.parquet", columns=["timestamp"])
    both = pd.read_parquet(SOURCE / "pretest_both_side_lows.parquet")
    ts = pd.to_datetime(both["timestamp"], utc=True)
    tr = both.loc[ts.isin(pd.to_datetime(raw_train["timestamp"], utc=True))].copy().reset_index(drop=True)
    dv = both.loc[ts.isin(pd.to_datetime(raw_dev["timestamp"], utc=True))].copy().reset_index(drop=True)
    cols, prep, grid, hazard = joint.load_hazard(fd / "models/hazard_survival_cdf.pt")
    xtr, xdv = joint.matrix(tr, cols), joint.matrix(dv, cols)
    y = tr["target"].astype(int).to_numpy()
    static_lgb = lgb_model().fit(xtr, y)
    static_cat = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.035, l2_leaf_reg=12,
                                    loss_function="Logloss", verbose=False, allow_writing_files=False,
                                    random_seed=20260703).fit(xtr, y)
    static = 0.5 * static_lgb.predict_proba(xdv)[:, 1] + 0.5 * static_cat.predict_proba(xdv)[:, 1]
    days = pd.to_datetime(dv["timestamp"], utc=True).dt.floor("D")
    online = {"online_all": np.zeros(len(dv)), "online_28": np.zeros(len(dv))}
    for day in sorted(days.unique()):
        current, prior = (days == day).to_numpy(), (days < day).to_numpy()
        history = pd.concat([tr, dv.loc[prior]], ignore_index=True)
        xhist = joint.matrix(history, cols)
        for name, window in (("online_all", None), ("online_28", 28)):
            use = np.ones(len(history), dtype=bool)
            if window is not None:
                hts = pd.to_datetime(history["timestamp"], utc=True)
                use = (hts >= pd.Timestamp(day) - pd.Timedelta(days=window)).to_numpy()
            model = lgb_model().fit(xhist.loc[use], history.loc[use, "target"].astype(int))
            online[name][current] = model.predict_proba(xdv.loc[current])[:, 1]
    probs = {"static_tree": static, **online,
             "static_online_all": 0.5 * static + 0.5 * online["online_all"],
             "static_online_28": 0.5 * static + 0.5 * online["online_28"]}
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
        alt = np.where(side == "UP", work["up_low"], work["down_low"])
        work["chosen_low"] = np.where(side == original, work["chosen_low"], alt)
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
    for model in folds["w1"]:
      for min_q in (0.50, 0.55, 0.60, 0.65, 0.70):
       for floor in (0.70, 0.75, 0.80, 0.85):
        for min_ev in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
            ms = {f: evaluate(folds[f][model], min_q, floor, min_ev) for f in FOLDS}
            pnl = {f: ms[f]["sum_pnl"] for f in FOLDS}
            tune = np.array([pnl[f] for f in TUNE])
            rows.append({"direction_model": model, "min_q": min_q, "floor": floor, "min_ev": min_ev,
                         **{f"{f}_pnl": pnl[f] for f in FOLDS},
                         **{f"{f}_accuracy": ms[f]["accepted_sample_accuracy"] for f in FOLDS},
                         "tune_sum": float(tune.sum()), "tune_worst": float(tune.min()),
                         "tune_robust": float(tune.sum() - tune.std()),
                         "holdout_sum": float(sum(pnl[f] for f in HOLDOUT)),
                         "holdout_worst": float(min(pnl[f] for f in HOLDOUT))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0]
    table.to_csv(SESSION / "online_full_direction_search.csv", index=False)
    summary = {"experiment_count": len(table), "winner": winner.to_dict(), "direction_coverage": 1.0,
               "causality": "Each UTC day retrains with base train plus earlier settled dev days only.",
               "btest_used": False, "top20": table.head(20).to_dict("records")}
    (SESSION / "online_full_direction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
