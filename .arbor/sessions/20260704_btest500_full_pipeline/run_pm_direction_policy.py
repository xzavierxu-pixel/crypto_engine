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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
OLD = ROOT / ".arbor/sessions/20260703_trade_btc_feature_research"
PREV = ROOT / ".arbor/sessions/20260703_prefinal_rolling"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("pm_dir_joint", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
trade = load_module("pm_dir_trade", OLD / "run_feature_research.py")
pathmod = load_module("pm_dir_path", SESSION / "run_path_signature_policy.py")


def absolute_pm_features(frame: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    absolute = frame.copy()
    absolute["selected_side"] = "UP"
    basic = trade.engineer_trades(absolute, lookup)
    path = pathmod.engineer_path(absolute, lookup)
    basic = basic.loc[:, ~basic.columns.str.contains("pside_minus")]
    renamed = {}
    for c in basic.columns:
        renamed[c] = c.replace("pm_trade_selected", "pm_up").replace("pm_trade_opposite", "pm_down")
    for c in path.columns:
        renamed[c] = c.replace("pm_path_selected", "pm_up_path").replace("pm_path_opposite", "pm_down_path")
    return pd.concat([basic, path], axis=1).rename(columns=renamed)


def fit_fold(fold: str) -> dict:
    fd = PREV / "folds" / fold
    raw_train = pd.read_parquet(fd / "data/train.parquet", columns=["timestamp"])
    raw_dev = pd.read_parquet(fd / "data/dev.parquet", columns=["timestamp"])
    both = pd.read_parquet(PREV / "pretest_both_side_lows.parquet")
    ts = pd.to_datetime(both["timestamp"], utc=True)
    tr = both.loc[ts.isin(pd.to_datetime(raw_train["timestamp"], utc=True))].copy().reset_index(drop=True)
    dv = both.loc[ts.isin(pd.to_datetime(raw_dev["timestamp"], utc=True))].copy().reset_index(drop=True)
    lookup = trade.load_trades([tr, dv])
    tr_pm, dv_pm = absolute_pm_features(tr, lookup), absolute_pm_features(dv, lookup)
    hazard_cols, prep, grid, hazard = joint.load_hazard(fd / "models/hazard_survival_cdf.pt")
    direction_derived = {"p_up", "p_side", "direction_confidence", "p_bin", "p_side_bucket", "selected_side"}
    base_cols = [c for c in hazard_cols if c not in direction_derived]
    cols = base_cols + list(tr_pm.columns)
    tr_plus, dv_plus = pd.concat([tr, tr_pm], axis=1), pd.concat([dv, dv_pm], axis=1)
    xtr, xdv = joint.matrix(tr_plus, cols), joint.matrix(dv_plus, cols)
    y = tr["target"].astype(int)
    models = {}
    for seed in (20260704, 20260705, 20260706):
        model = LGBMClassifier(
            n_estimators=650, learning_rate=0.02, num_leaves=15, max_depth=5,
            min_child_samples=150, colsample_bytree=0.30, subsample=0.8,
            reg_lambda=16, reg_alpha=4, verbosity=-1, n_jobs=-1, random_state=seed,
        ).fit(xtr, y)
        models[f"lgb_{seed}"] = model.predict_proba(xdv)[:, 1]
    lgb_avg = np.mean(list(models.values()), axis=0)
    cat = CatBoostClassifier(
        iterations=550, depth=5, learning_rate=0.025, l2_leaf_reg=14,
        loss_function="Logloss", verbose=False, allow_writing_files=False,
        random_seed=20260704,
    ).fit(xtr, y)
    cat_p = cat.predict_proba(xdv)[:, 1]
    probs = {"lgb_avg": lgb_avg, "cat": cat_p, "tree_blend": 0.5*lgb_avg + 0.5*cat_p}
    prepared = {}
    for name, p in probs.items():
        side = np.where(p >= 0.5, "UP", "DOWN")
        work = dv.copy()
        original = work["selected_side"].astype(str).str.upper().to_numpy()
        work["selected_side"] = side
        work["p_up"] = p
        work["p_side"] = np.maximum(p, 1-p)
        work["direction_confidence"] = np.abs(p-0.5)
        truth = np.where(work["target"].astype(int).to_numpy() == 1, "UP", "DOWN")
        work["correct"] = side == truth
        work["chosen_low"] = np.where(side == original, work["chosen_low"],
                                      np.where(side == "UP", work["up_low"], work["down_low"]))
        gc = joint.predict_hazard(hazard, prep.transform(work), torch.device("cpu"), 512)[1]
        prepared[name] = (work, gc, grid)
    direction_metrics = {name: {"accuracy": float(np.mean((p >= .5) == dv["target"].astype(bool).to_numpy())),
                                "auc": float(roc_auc_score(dv["target"], p)),
                                "brier": float(brier_score_loss(dv["target"], p)),
                                "logloss": float(log_loss(dv["target"], np.clip(p,1e-6,1-1e-6)))}
                         for name, p in probs.items()}
    return {"prepared": prepared, "direction_metrics": direction_metrics, "feature_count": len(cols)}


def evaluate(payload: tuple, min_q: float, floor: float, min_ev: float) -> dict:
    work, gc, grid = payload
    q = work["p_side"].to_numpy(float)
    bid, ev, fill = joint.choose_survival_expected_return_bids(q, gc, grid, 0.01, min_ev, min_fill_probability=floor)
    bid[(q < min_q) | ~np.isfinite(work["chosen_low"].to_numpy(float))] = 0.0
    return joint.backtest_metrics(work, joint.backtest_with_bid(work, bid, ev, fill), len(work))


def main() -> None:
    folds = {f: fit_fold(f) for f in FOLDS}
    rows = []
    for model in folds["w1"]["prepared"]:
        for min_q in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            for floor in (0.70, 0.75, 0.80, 0.85, 0.90):
                for min_ev in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
                    ms = {f: evaluate(folds[f]["prepared"][model], min_q, floor, min_ev) for f in FOLDS}
                    pnl = {f: ms[f]["sum_pnl"] for f in FOLDS}
                    tune = np.array([pnl[f] for f in TUNE])
                    rows.append({"direction_model": model, "min_q": min_q, "floor": floor, "min_ev": min_ev,
                                 **{f"{f}_pnl": pnl[f] for f in FOLDS},
                                 **{f"{f}_accuracy": ms[f]["accepted_sample_accuracy"] for f in FOLDS},
                                 "tune_sum": float(tune.sum()), "tune_worst": float(tune.min()),
                                 "tune_robust": float(tune.sum()-tune.std()),
                                 "holdout_sum": float(sum(pnl[f] for f in HOLDOUT))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    summary = {"experiment_count": len(table), "winner": table.iloc[0].to_dict(),
               "selection_folds": TUNE, "untouched_folds": HOLDOUT, "btest_used": False,
               "feature_count": folds["w1"]["feature_count"],
               "direction_metrics": {f: v["direction_metrics"] for f,v in folds.items()},
               "top20": table.head(20).to_dict("records")}
    table.to_csv(SESSION / "pm_direction_search.csv", index=False)
    (SESSION / "pm_direction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": summary["winner"], "feature_count": summary["feature_count"],
                      "direction_metrics": summary["direction_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
