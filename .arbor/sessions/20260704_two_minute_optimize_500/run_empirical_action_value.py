#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
SOURCE = ROOT / ".arbor/sessions/20260704_two_minute_shift"
JOINT_PATH = ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
spec = importlib.util.spec_from_file_location("joint", JOINT_PATH)
joint = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(joint)

FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE = FOLDS[:4]
HOLDOUT = FOLDS[4:]
ACTIONS = np.round(np.arange(0.01, 0.86, 0.01), 2)


def load_fold(fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = SOURCE / "folds" / fold / "data"
    train = pd.read_parquet(base / "train.parquet")
    dev = pd.read_parquet(base / "dev.parquet")
    return (
        train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True),
        dev.loc[dev["threshold_accepted"].astype(bool)].reset_index(drop=True),
    )


def keys(frame: pd.DataFrame, bins: int, use_side: bool, use_session: bool) -> pd.Series:
    p = np.clip(pd.to_numeric(frame["p_side"], errors="coerce").fillna(0.5), 0.5, 0.999999)
    pbin = np.minimum(((p - 0.5) * 2 * bins).astype(int), bins - 1).astype(str)
    result = "p" + pbin
    if use_side:
        result = result + "_" + frame["selected_side"].astype(str)
    if use_session:
        hour = pd.to_datetime(frame["decision_time"], utc=True).dt.hour
        session = pd.cut(hour, [-1, 7, 15, 23], labels=["asia", "europe", "us"]).astype(str)
        result = result + "_" + session
    return result


def action_values(frame: pd.DataFrame) -> np.ndarray:
    correct = frame["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(frame["chosen_low"], errors="coerce").to_numpy(float)
    correct_fill = correct[:, None] & np.isfinite(low[:, None]) & (low[:, None] <= ACTIONS[None, :] + 1e-12)
    return correct_fill * (1.0 - ACTIONS[None, :]) - (~correct)[:, None] * ACTIONS[None, :]


def policy(train: pd.DataFrame, dev: pd.DataFrame, params: dict) -> np.ndarray:
    dev_start = pd.to_datetime(dev["timestamp"], utc=True).min()
    if params["lookback_days"] > 0:
        cutoff = dev_start - pd.Timedelta(days=params["lookback_days"])
        train = train.loc[pd.to_datetime(train["timestamp"], utc=True) >= cutoff].reset_index(drop=True)
    cap_mask = ACTIONS <= params["bid_cap"] + 1e-12
    actions = ACTIONS[cap_mask]
    values = action_values(train)[:, cap_mask]
    global_mean = values.mean(axis=0)
    train_key = keys(train, params["bins"], params["use_side"], params["use_session"])
    dev_key = keys(dev, params["bins"], params["use_side"], params["use_session"])
    chosen: dict[str, float] = {}
    for group, index in train_key.groupby(train_key).groups.items():
        group_values = values[np.asarray(list(index), dtype=int)]
        n = len(group_values)
        mean = (group_values.sum(axis=0) + params["alpha"] * global_mean) / (n + params["alpha"])
        best = int(np.argmax(mean))
        chosen[str(group)] = float(actions[best]) if mean[best] >= params["min_value"] else 0.0
    global_best = int(np.argmax(global_mean))
    fallback = float(actions[global_best]) if global_mean[global_best] >= params["min_value"] else 0.0
    return dev_key.map(chosen).fillna(fallback).to_numpy(float)


def pnl(frame: pd.DataFrame, bid: np.ndarray) -> float:
    correct = frame["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(frame["chosen_low"], errors="coerce").to_numpy(float)
    submitted = bid > 0
    filled = submitted & (~correct | (np.isfinite(low) & (low <= bid + 1e-12)))
    result = np.zeros(len(frame))
    result[filled & correct] = 1.0 - bid[filled & correct]
    result[filled & ~correct] = -bid[filled & ~correct]
    return float(result.sum())


def main() -> None:
    tune_data = {fold: load_fold(fold) for fold in TUNE}
    all_params = [
        dict(zip(["bins", "use_side", "use_session", "lookback_days", "alpha", "min_value", "bid_cap"], values))
        for values in itertools.product(
            [4, 6, 10, 15], [False, True], [False, True], [0, 14, 28, 56], [10.0, 30.0, 75.0, 150.0],
            [-0.01, 0.0, 0.005, 0.01, 0.02], [0.30, 0.50, 0.65, 0.85]
        )
    ]
    rng = np.random.default_rng(20260704)
    selected = rng.choice(len(all_params), size=min(700, len(all_params)), replace=False)
    rows = []
    for index in selected:
        params = all_params[int(index)]
        fold_pnl = {fold: pnl(tune_data[fold][1], policy(*tune_data[fold], params)) for fold in TUNE}
        values = np.asarray(list(fold_pnl.values()))
        rows.append({**params, **{f"{fold}_pnl": value for fold, value in fold_pnl.items()},
                     "tune_sum": float(values.sum()), "tune_worst": float(values.min()),
                     "positive_weeks": int((values > 0).sum()), "robust_score": float(values.sum() - values.std())})
    table = pd.DataFrame(rows).sort_values(["positive_weeks", "robust_score", "tune_sum"], ascending=False)
    table.to_csv(SESSION / "cycle2_empirical_action_search.csv", index=False)
    eligible = table.loc[table["positive_weeks"] >= 3]
    winner = (eligible if not eligible.empty else table).iloc[0].to_dict()
    param_keys = ["bins", "use_side", "use_session", "lookback_days", "alpha", "min_value", "bid_cap"]
    params = {key: winner[key] for key in param_keys}
    params.update({"bins": int(params["bins"]), "use_side": bool(params["use_side"]),
                   "use_session": bool(params["use_session"]), "lookback_days": int(params["lookback_days"])})
    holdout = {}
    for fold in HOLDOUT:
        train, dev = load_fold(fold)
        bid = policy(train, dev, params)
        result = joint.backtest_with_bid(dev, bid)
        holdout[fold] = joint.backtest_metrics(dev, result, len(pd.read_parquet(SOURCE / "folds" / fold / "data/dev.parquet")))
    payload = {"experiment_count": len(table), "winner": winner, "winner_params": params,
               "holdout_metrics": holdout, "holdout_sum": float(sum(v["sum_pnl"] for v in holdout.values())),
               "holdout_gate_passed": bool(all(v["sum_pnl"] > 0 for v in holdout.values())),
               "top20": table.head(20).to_dict("records")}
    (SESSION / "cycle2_empirical_action_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
