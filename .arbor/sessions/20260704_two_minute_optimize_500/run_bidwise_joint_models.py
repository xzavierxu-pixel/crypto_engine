#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

SESSION = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("base", SESSION / "run_joint_fill_value.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(base)


def fit_fold(fold: str) -> dict:
    cache = SESSION / "bidwise_joint_cache" / f"{fold}.npz"
    if cache.exists():
        saved = np.load(cache)
        return {key: saved[key] for key in saved.files}
    train, dev, accepted, cols = base.load_frames(fold)
    xtr = base.joint.matrix(train, cols)
    xdv = base.joint.matrix(accepted, cols)
    correct = train["correct"].astype(bool).to_numpy()
    q_model = LGBMClassifier(n_estimators=500, learning_rate=.025, num_leaves=31, max_depth=7,
                             min_child_samples=100, colsample_bytree=.45, reg_alpha=2, reg_lambda=10,
                             random_state=20260704, verbosity=-1, n_jobs=-1)
    q_model.fit(xtr, correct.astype(int))
    top = np.argsort(q_model.feature_importances_)[-180:]
    q = q_model.predict_proba(xdv)[:, 1]
    xtr = xtr.iloc[:, top]
    xdv = xdv.iloc[:, top]
    low = pd.to_numeric(train["chosen_low"], errors="coerce").to_numpy(float)
    predictions = []
    for index, bid in enumerate(base.BIDS):
        y = (correct & np.isfinite(low) & (low <= bid + 1e-12)).astype(int)
        model = LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=24, max_depth=6,
                               min_child_samples=150, colsample_bytree=.55, reg_alpha=2, reg_lambda=12,
                               random_state=20260704 + index, verbosity=-1, n_jobs=-1)
        model.fit(xtr, y)
        predictions.append(model.predict_proba(xdv)[:, 1])
    h = np.maximum.accumulate(np.column_stack(predictions), axis=1)
    dev_correct = accepted["correct"].astype(bool).to_numpy()
    dev_low = pd.to_numeric(accepted["chosen_low"], errors="coerce").to_numpy(float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, q=q, h=h, correct=dev_correct, low=dev_low,
                        sample_count=np.asarray([len(dev)]))
    return {"q": q, "h": h, "correct": dev_correct, "low": dev_low, "sample_count": np.asarray([len(dev)])}


def choose(data: dict, params: dict):
    q = np.clip((1 - params["q_shrink"]) * data["q"] + params["q_shrink"] * .5, .001, .999)
    h = np.clip(params["h_scale"] * data["h"] ** params["h_power"], 0, 1)
    ev = h * (1 - base.BIDS[None, :]) - (1 - q[:, None]) * base.BIDS[None, :]
    best = np.argmax(ev, axis=1)
    best_ev = ev[np.arange(len(ev)), best]
    bid = base.BIDS[best].copy()
    bid[best_ev < params["min_ev"]] = 0
    return bid, best_ev, h[np.arange(len(h)), best]


def metrics(fold: str, data: dict, params: dict) -> dict:
    _, dev, accepted, _ = base.load_frames(fold)
    bid, ev, fill = choose(data, params)
    return base.joint.backtest_metrics(accepted, base.joint.backtest_with_bid(accepted, bid, ev, fill), len(dev))


def main() -> None:
    tune = {fold: fit_fold(fold) for fold in base.TUNE}
    rows = []
    for q_shrink in [-.4, -.25, -.1, 0, .15]:
        for h_power in [.7, .85, 1, 1.2, 1.5]:
            for h_scale in [.5, .65, .8, .95, 1.1]:
                for min_ev in [0, .005, .01, .02, .04]:
                    params = {"q_shrink": q_shrink, "h_power": h_power, "h_scale": h_scale, "min_ev": min_ev}
                    values = np.asarray([base.pnl(tune[fold], choose(tune[fold], params)[0]) for fold in base.TUNE])
                    rows.append({**params, **{f"{fold}_pnl": values[i] for i, fold in enumerate(base.TUNE)},
                                 "tune_sum": values.sum(), "tune_worst": values.min(),
                                 "positive_weeks": int((values > 0).sum()), "robust_score": values.sum() - values.std()})
    table = pd.DataFrame(rows).sort_values(["positive_weeks", "robust_score", "tune_sum"], ascending=False)
    table.to_csv(SESSION / "cycle8_bidwise_joint_search.csv", index=False)
    winner = table.loc[table["positive_weeks"] >= 3].iloc[0].to_dict()
    params = {key: winner[key] for key in ["q_shrink", "h_power", "h_scale", "min_ev"]}
    holdout_data = {fold: fit_fold(fold) for fold in base.HOLDOUT}
    holdout = {fold: metrics(fold, holdout_data[fold], params) for fold in base.HOLDOUT}
    payload = {"winner": winner, "winner_params": params, "holdout_metrics": holdout,
               "holdout_sum": float(sum(v["sum_pnl"] for v in holdout.values())),
               "holdout_gate_passed": bool(all(v["sum_pnl"] > 0 for v in holdout.values())),
               "top20": table.head(20).to_dict("records")}
    (SESSION / "cycle8_bidwise_joint_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
