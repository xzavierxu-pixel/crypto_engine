#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
SOURCE = ROOT / ".arbor/sessions/20260704_two_minute_shift"
JOINT_PATH = ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
spec = importlib.util.spec_from_file_location("joint", JOINT_PATH)
joint = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(joint)

FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]
BIDS = np.round(np.arange(0.05, 0.851, 0.05), 2)


def load_frames(fold: str):
    d = SOURCE / "folds" / fold
    train = pd.read_parquet(d / "data/train.parquet")
    dev = pd.read_parquet(d / "data/dev.parquet")
    train = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    accepted = dev.loc[dev["threshold_accepted"].astype(bool)].reset_index(drop=True)
    cols = joint.load_hazard(d / "models/hazard_survival_cdf.pt")[0]
    return train, dev, accepted, cols


def fit_fold(fold: str) -> dict:
    cache = SESSION / "joint_fill_cache" / f"{fold}.npz"
    if cache.exists():
        saved = np.load(cache)
        return {key: saved[key] for key in saved.files}
    train, dev, accepted, cols = load_frames(fold)
    xtr = joint.matrix(train, cols)
    xdv = joint.matrix(accepted, cols)
    y_correct = train["correct"].astype(int).to_numpy()
    q_model = LGBMClassifier(
        n_estimators=450, learning_rate=0.025, num_leaves=31, max_depth=7, min_child_samples=100,
        subsample=0.85, colsample_bytree=0.45, reg_alpha=2.0, reg_lambda=10.0,
        random_state=20260704, verbosity=-1, n_jobs=-1,
    )
    q_model.fit(xtr, y_correct)
    importance = np.asarray(q_model.feature_importances_)
    top_index = np.argsort(importance)[-120:]
    q = q_model.predict_proba(xdv)[:, 1]
    xtr_top = xtr.iloc[:, top_index].to_numpy(np.float32)
    xdv_top = xdv.iloc[:, top_index].to_numpy(np.float32)
    repeated = np.repeat(xtr_top, len(BIDS), axis=0)
    bid_column = np.tile(BIDS.astype(np.float32), len(train))[:, None]
    expanded_x = np.concatenate([repeated, bid_column], axis=1)
    correct = train["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(train["chosen_low"], errors="coerce").to_numpy(float)
    expanded_y = (
        correct[:, None] & np.isfinite(low[:, None]) & (low[:, None] <= BIDS[None, :] + 1e-12)
    ).astype(np.uint8).reshape(-1)
    constraints = [0] * xtr_top.shape[1] + [1]
    model = LGBMClassifier(
        n_estimators=550, learning_rate=0.025, num_leaves=31, max_depth=7, min_child_samples=250,
        subsample=0.85, colsample_bytree=0.60, reg_alpha=2.0, reg_lambda=12.0,
        monotone_constraints=constraints, random_state=20260704, verbosity=-1, n_jobs=-1,
    )
    model.fit(expanded_x, expanded_y)
    dev_expanded = np.concatenate(
        [np.repeat(xdv_top, len(BIDS), axis=0), np.tile(BIDS.astype(np.float32), len(accepted))[:, None]], axis=1
    )
    h = model.predict_proba(dev_expanded)[:, 1].reshape(len(accepted), len(BIDS))
    h = np.maximum.accumulate(h, axis=1)
    dev_correct = accepted["correct"].astype(bool).to_numpy()
    dev_low = pd.to_numeric(accepted["chosen_low"], errors="coerce").to_numpy(float)
    event = dev_correct[:, None] & np.isfinite(dev_low[:, None]) & (dev_low[:, None] <= BIDS[None, :] + 1e-12)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, q=q, h=h, event=event, correct=dev_correct, low=dev_low,
                        sample_count=np.asarray([len(dev)]), accepted_count=np.asarray([len(accepted)]))
    return {"q": q, "h": h, "event": event, "correct": dev_correct, "low": dev_low,
            "sample_count": np.asarray([len(dev)]), "accepted_count": np.asarray([len(accepted)])}


def choose(data: dict, params: dict):
    q = np.clip((1 - params["q_shrink"]) * data["q"] + params["q_shrink"] * 0.5, 0.001, 0.999)
    h = np.clip(params["h_scale"] * np.clip(data["h"], 0, 1) ** params["h_power"], 0, 1)
    ev = h * (1.0 - BIDS[None, :]) - (1.0 - q[:, None]) * BIDS[None, :]
    best = np.argmax(ev, axis=1)
    best_ev = ev[np.arange(len(ev)), best]
    bid = BIDS[best].copy()
    bid[best_ev < params["min_ev"]] = 0.0
    return bid, best_ev, h[np.arange(len(h)), best]


def pnl(data: dict, bid: np.ndarray) -> float:
    correct, low = data["correct"].astype(bool), data["low"]
    filled = (bid > 0) & (~correct | (np.isfinite(low) & (low <= bid + 1e-12)))
    out = np.zeros(len(bid))
    out[filled & correct] = 1 - bid[filled & correct]
    out[filled & ~correct] = -bid[filled & ~correct]
    return float(out.sum())


def full_metrics(fold: str, data: dict, params: dict) -> dict:
    _, dev, accepted, _ = load_frames(fold)
    bid, ev, fill = choose(data, params)
    return joint.backtest_metrics(accepted, joint.backtest_with_bid(accepted, bid, ev, fill), len(dev))


def main() -> None:
    tune = {fold: fit_fold(fold) for fold in TUNE}
    rows = []
    for q_shrink in [-0.2, 0.0, 0.15, 0.30]:
        for h_power in [0.7, 1.0, 1.3, 1.7, 2.2]:
            for h_scale in [0.75, 0.9, 1.0, 1.1, 1.25]:
                for min_ev in [-0.01, 0.0, 0.005, 0.01, 0.02, 0.04, 0.07]:
                    params = {"q_shrink": q_shrink, "h_power": h_power, "h_scale": h_scale, "min_ev": min_ev}
                    fold_pnl = {fold: pnl(tune[fold], choose(tune[fold], params)[0]) for fold in TUNE}
                    values = np.asarray(list(fold_pnl.values()))
                    rows.append({**params, **{f"{fold}_pnl": value for fold, value in fold_pnl.items()},
                                 "tune_sum": values.sum(), "tune_worst": values.min(),
                                 "positive_weeks": int((values > 0).sum()), "robust_score": values.sum() - values.std()})
    table = pd.DataFrame(rows).sort_values(["positive_weeks", "robust_score", "tune_sum"], ascending=False)
    table.to_csv(SESSION / "cycle4_joint_fill_search.csv", index=False)
    winner = table.loc[table["positive_weeks"] >= 3].iloc[0].to_dict()
    params = {key: winner[key] for key in ["q_shrink", "h_power", "h_scale", "min_ev"]}
    holdout_data = {fold: fit_fold(fold) for fold in HOLDOUT}
    holdout = {fold: full_metrics(fold, holdout_data[fold], params) for fold in HOLDOUT}
    brier = {fold: float(brier_score_loss(tune[fold]["event"].reshape(-1), tune[fold]["h"].reshape(-1))) for fold in TUNE}
    payload = {"winner": winner, "winner_params": params, "joint_event_brier": brier,
               "holdout_metrics": holdout, "holdout_sum": float(sum(v["sum_pnl"] for v in holdout.values())),
               "holdout_gate_passed": bool(all(v["sum_pnl"] > 0 for v in holdout.values())),
               "top20": table.head(20).to_dict("records")}
    (SESSION / "cycle4_joint_fill_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
