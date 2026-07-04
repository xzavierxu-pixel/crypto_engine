#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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


joint = load_module("online_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
feat = load_module("online_feat", SESSION / "run_feature_research.py")
emp = load_module("online_emp", SESSION / "run_empirical_gc.py")


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
    _, _, grid, _ = joint.load_hazard(d / "models/hazard_survival_cdf.pt")
    dev_days = pd.to_datetime(dv["timestamp"], utc=True).dt.floor("D")
    q_store = {name: np.zeros(len(dv)) for name in ("raw", "online_all", "online_28", "raw_online_all", "raw_online_28")}
    gc_store = {name: np.zeros((len(dv), len(grid))) for name in ("emp_all", "emp_28")}
    for day in sorted(dev_days.unique()):
        current = (dev_days == day).to_numpy()
        prior = (dev_days < day).to_numpy()
        history = pd.concat([tr, dv.loc[prior]], ignore_index=True)
        history_trade = pd.concat([tr_trade, dv_trade.loc[prior]], ignore_index=True)
        x_current = design(dv.loc[current].reset_index(drop=True), dv_trade.loc[current].reset_index(drop=True))
        raw = dv.loc[current, "p_side"].to_numpy(float)
        q_store["raw"][current] = raw
        for suffix, days in (("all", None), ("28", 28)):
            use = np.ones(len(history), dtype=bool)
            if days is not None:
                hts = pd.to_datetime(history["timestamp"], utc=True)
                use = (hts >= pd.Timestamp(day) - pd.Timedelta(days=days)).to_numpy()
            xh = design(history.loc[use].reset_index(drop=True), history_trade.loc[use].reset_index(drop=True))
            yh = history.loc[use, "correct"].astype(int)
            model = LGBMClassifier(n_estimators=180, learning_rate=0.035, num_leaves=10, max_depth=4,
                                   min_child_samples=120, subsample=0.8, colsample_bytree=0.7,
                                   reg_lambda=15, reg_alpha=4, random_state=20260703, verbosity=-1, n_jobs=-1)
            model.fit(xh, yh)
            qp = model.predict_proba(x_current)[:, 1]
            q_store[f"online_{suffix}"][current] = qp
            q_store[f"raw_online_{suffix}"][current] = 0.5 * raw + 0.5 * qp
            gc_store[f"emp_{suffix}"][current] = emp.empirical_gc(
                history.loc[use].reset_index(drop=True), dv.loc[current].reset_index(drop=True), grid, None, "pbin")
    y = dv["correct"].astype(int).to_numpy()
    calibration = {name: {"brier": float(brier_score_loss(y, q)),
                          "log_loss": float(log_loss(y, np.clip(q, 1e-6, 1 - 1e-6)))} for name, q in q_store.items()}
    return {"dev": dev, "accepted": dv, "qs": q_store, "gcs": gc_store, "grid": grid,
            "calibration": calibration}


def evaluate(p: dict, q_name: str, gc_name: str, floor: float, min_ev: float) -> dict:
    bid, ev, fill = joint.choose_survival_expected_return_bids(
        p["qs"][q_name], p["gcs"][gc_name], p["grid"], 0.01, min_ev,
        min_fill_probability=floor)
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
    table.to_csv(SESSION / "online_adaptation_search.csv", index=False)
    summary = {"experiment_count": len(table), "winner": winner.to_dict(),
               "calibration": {f: p["calibration"] for f, p in prepared.items()},
               "causality": "Each UTC day uses base-train plus only earlier settled dev days; current/future-day labels excluded.",
               "btest_used": False, "top20": table.head(20).to_dict("records")}
    (SESSION / "online_adaptation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": summary["winner"], "calibration": summary["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
