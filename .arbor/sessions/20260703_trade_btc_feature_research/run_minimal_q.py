#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

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


joint = load_module("minimal_q_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
feat = load_module("minimal_q_feat", SESSION / "run_feature_research.py")
emp = load_module("minimal_q_emp", SESSION / "run_empirical_gc.py")


def design(frame: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    out = trade.copy()
    out.insert(0, "p_side", frame["p_side"].to_numpy(float))
    out.insert(1, "side_up", (frame["selected_side"].astype(str).str.upper() == "UP").to_numpy(float))
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def prepare_fold(fold: str) -> dict:
    d = PREV / "folds" / fold
    train, dev = pd.read_parquet(d / "data/train.parquet"), pd.read_parquet(d / "data/dev.parquet")
    lookup = feat.load_trades([train, dev])
    tr_trade, dv_trade = feat.engineer_trades(train, lookup), feat.engineer_trades(dev, lookup)
    tr_mask, dv_mask = train["threshold_accepted"].astype(bool), dev["threshold_accepted"].astype(bool)
    tr, dv = train.loc[tr_mask].reset_index(drop=True), dev.loc[dv_mask].reset_index(drop=True)
    tr_trade, dv_trade = tr_trade.loc[tr_mask].reset_index(drop=True), dv_trade.loc[dv_mask].reset_index(drop=True)
    xtr, xdv = design(tr, tr_trade), design(dv, dv_trade)
    ts = pd.to_datetime(tr["timestamp"], utc=True)
    cutoff = ts.max() - pd.Timedelta(days=7)
    fit_mask, cal_mask = (ts < cutoff).to_numpy(), (ts >= cutoff).to_numpy()
    model = LGBMClassifier(n_estimators=400, learning_rate=0.025, num_leaves=12, max_depth=4,
                           min_child_samples=150, subsample=0.8, colsample_bytree=0.65,
                           reg_lambda=15, reg_alpha=4, random_state=20260703, verbosity=-1, n_jobs=-1)
    model.fit(xtr.loc[fit_mask], tr.loc[fit_mask, "correct"].astype(int))
    qcal = model.predict_proba(xtr.loc[cal_mask])[:, 1]
    qdev = model.predict_proba(xdv)[:, 1]
    ycal = tr.loc[cal_mask, "correct"].astype(int).to_numpy()
    raw_cal = tr.loc[cal_mask, "p_side"].to_numpy(float)
    raw_dev = dv["p_side"].to_numpy(float)
    def logit(x):
        x = np.clip(x, 1e-5, 1 - 1e-5)
        return np.log(x / (1 - x))
    platt = LogisticRegression(C=0.2, max_iter=500).fit(np.c_[logit(qcal), logit(raw_cal)], ycal)
    qplatt = platt.predict_proba(np.c_[logit(qdev), logit(raw_dev)])[:, 1]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(qcal, ycal)
    qiso = iso.predict(qdev)
    qs = {"raw": raw_dev, "minimal": qdev, "minimal_platt": qplatt, "minimal_iso": qiso}
    for name, q in (("minimal", qdev), ("platt", qplatt)):
        for w in (0.25, 0.5, 0.75):
            qs[f"raw_{name}_{w:.2f}"] = (1 - w) * raw_dev + w * q
    _, prep, grid, hazard = joint.load_hazard(d / "models/hazard_survival_cdf.pt")
    gc_all = joint.predict_hazard(hazard, prep.transform(dev), torch.device("cpu"), 512)[1]
    gcs = {"hazard": gc_all[dv_mask.to_numpy()], "emp_all_pbin": emp.empirical_gc(tr, dv, grid, None, "pbin")}
    ydev = dv["correct"].astype(int).to_numpy()
    calibration = {name: {"brier": float(brier_score_loss(ydev, q)),
                          "log_loss": float(log_loss(ydev, np.clip(q, 1e-6, 1 - 1e-6)))} for name, q in qs.items()}
    return {"dev": dev, "accepted": dv, "qs": qs, "gcs": gcs, "grid": grid, "calibration": calibration}


def evaluate(p: dict, q_name: str, gc_name: str, floor: float, min_ev: float) -> dict:
    bid, ev, fill = joint.choose_survival_expected_return_bids(
        p["qs"][q_name], p["gcs"][gc_name], p["grid"], 0.01, min_ev, min_fill_probability=floor)
    return joint.backtest_metrics(p["accepted"], joint.backtest_with_bid(p["accepted"], bid, ev, fill), len(p["dev"]))


def main() -> None:
    prepared = {f: prepare_fold(f) for f in FOLDS}
    rows = []
    for q_name in prepared["w1"]["qs"]:
      for gc_name in prepared["w1"]["gcs"]:
       for floor in (0.60, 0.70, 0.75, 0.80, 0.85):
        for min_ev in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
            metrics = {f: evaluate(prepared[f], q_name, gc_name, floor, min_ev) for f in FOLDS}
            pnl = {f: metrics[f]["sum_pnl"] for f in FOLDS}
            tune = np.array([pnl[f] for f in TUNE])
            rows.append({"q_model": q_name, "gc_model": gc_name, "floor": floor, "min_ev": min_ev,
                         **{f"{f}_pnl": pnl[f] for f in FOLDS}, "tune_sum": float(tune.sum()),
                         "tune_worst": float(tune.min()), "tune_robust": float(tune.sum() - tune.std()),
                         "holdout_sum": float(sum(pnl[f] for f in HOLDOUT)),
                         "holdout_worst": float(min(pnl[f] for f in HOLDOUT))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0]
    table.to_csv(SESSION / "minimal_q_search.csv", index=False)
    summary = {"experiment_count": len(table), "winner": winner.to_dict(),
               "calibration": {f: p["calibration"] for f, p in prepared.items()}, "btest_used": False,
               "top20": table.head(20).to_dict("records")}
    (SESSION / "minimal_q_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": summary["winner"], "calibration": summary["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
