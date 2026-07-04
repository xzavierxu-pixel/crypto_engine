#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
OLD = ROOT / ".arbor/sessions/20260703_trade_btc_feature_research"
PREV = ROOT / ".arbor/sessions/20260703_prefinal_rolling"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]
WINDOWS = (15, 30, 60, 120)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("joint_path", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
base = load_module("base_trade", OLD / "run_feature_research.py")


def path_one(group: pd.DataFrame | None, decision: pd.Timestamp, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    eligible = group.loc[group["trade_time"] <= decision] if group is not None and len(group) else None
    for window in WINDOWS:
        names = (
            "interarrival_mean", "interarrival_cv", "path_efficiency", "up_move_share",
            "reversal_share", "delta_autocorr", "early_late_shift", "max_drawdown",
            "max_runup", "q10", "q25", "q75", "q90", "last_rank", "duration_share",
        )
        if eligible is None:
            vals = [0.0] * len(names)
        else:
            g = eligible.loc[eligible["trade_time"] > decision - pd.Timedelta(seconds=window)]
            p = g["price"].astype(float).to_numpy()
            if len(p) == 0:
                vals = [0.0] * len(names)
            else:
                t = (g["trade_time"] - g["trade_time"].iloc[0]).dt.total_seconds().to_numpy(float)
                dt = np.diff(t)
                dp = np.diff(p)
                total_var = float(np.abs(dp).sum())
                efficiency = float(abs(p[-1] - p[0]) / total_var) if total_var > 1e-12 else 0.0
                signs = np.sign(dp)
                reversals = float(np.mean(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0.0
                autocorr = float(np.corrcoef(dp[1:], dp[:-1])[0, 1]) if len(dp) > 2 and np.std(dp[1:]) > 0 and np.std(dp[:-1]) > 0 else 0.0
                running_max = np.maximum.accumulate(p)
                running_min = np.minimum.accumulate(p)
                half = max(1, len(p) // 2)
                vals = [
                    float(dt.mean()) if len(dt) else 0.0,
                    float(dt.std() / max(dt.mean(), 1e-6)) if len(dt) else 0.0,
                    efficiency,
                    float(np.mean(dp > 0)) if len(dp) else 0.0,
                    reversals,
                    autocorr,
                    float(p[half:].mean() - p[:half].mean()) if len(p) > 1 else 0.0,
                    float(np.max(running_max - p)),
                    float(np.max(p - running_min)),
                    *[float(v) for v in np.quantile(p, [0.10, 0.25, 0.75, 0.90])],
                    float(np.mean(p <= p[-1])),
                    float((t[-1] - t[0]) / window) if len(t) > 1 else 0.0,
                ]
        for name, value in zip(names, vals):
            out[f"{prefix}_{name}_{window}s"] = float(np.nan_to_num(value))
    return out


def engineer_path(frame: pd.DataFrame, lookup: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for row in frame[["condition_id", "selected_side", "decision_time"]].itertuples(index=False):
        selected = str(row.selected_side).upper()
        opposite = "DOWN" if selected == "UP" else "UP"
        decision = pd.Timestamp(row.decision_time)
        a = path_one(lookup.get((str(row.condition_id), selected)), decision, "pm_path_selected")
        b = path_one(lookup.get((str(row.condition_id), opposite)), decision, "pm_path_opposite")
        values = {**a, **b}
        for window in WINDOWS:
            for name in ("path_efficiency", "up_move_share", "reversal_share", "early_late_shift", "max_drawdown", "max_runup", "last_rank"):
                values[f"pm_path_diff_{name}_{window}s"] = a[f"pm_path_selected_{name}_{window}s"] - b[f"pm_path_opposite_{name}_{window}s"]
        rows.append(values)
    return pd.DataFrame(rows, index=frame.index, dtype="float32")


def fit_q(train: pd.DataFrame, dev: pd.DataFrame, cols: list[str]) -> np.ndarray:
    tr = train.loc[train["threshold_accepted"].astype(bool)]
    dv = dev.loc[dev["threshold_accepted"].astype(bool)]
    model = LGBMClassifier(
        n_estimators=700, learning_rate=0.018, num_leaves=15, max_depth=5,
        min_child_samples=120, subsample=0.8, colsample_bytree=0.3,
        reg_lambda=18, reg_alpha=4, random_state=20260704, verbosity=-1, n_jobs=-1,
    )
    model.fit(joint.matrix(tr, cols), tr["correct"].astype(int))
    return model.predict_proba(joint.matrix(dv, cols))[:, 1]


def prepare_fold(fold: str) -> dict:
    d = PREV / "folds" / fold
    train_path, dev_path = d / "data/train.parquet", d / "data/dev.parquet"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    prepared = joint.prepare(train_path, dev_path, d / "models/hazard_survival_cdf.pt", 20260704)
    lookup = base.load_trades([train, dev])
    tr_basic, dv_basic = base.engineer_trades(train, lookup), base.engineer_trades(dev, lookup)
    tr_path, dv_path = engineer_path(train, lookup), engineer_path(dev, lookup)
    tr_extra, dv_extra = pd.concat([tr_basic, tr_path], axis=1), pd.concat([dv_basic, dv_path], axis=1)
    hazard_cols = joint.load_hazard(d / "models/hazard_survival_cdf.pt")[0]
    train_plus, dev_plus = pd.concat([train, tr_extra], axis=1), pd.concat([dev, dv_extra], axis=1)
    q_path = fit_q(train_plus, dev_plus, hazard_cols + list(tr_extra.columns))
    qs = {
        "raw": prepared[2]["raw"],
        "old_tree": prepared[2]["raw_tree_blend"],
        "path": q_path,
        "raw_path": 0.5 * prepared[2]["raw"] + 0.5 * q_path,
        "old_path": 0.5 * prepared[2]["raw_tree_blend"] + 0.5 * q_path,
    }

    _, prep, grid, hazard = joint.load_hazard(d / "models/hazard_survival_cdf.pt")
    gc_train_all = joint.predict_hazard(hazard, prep.transform(train), torch.device("cpu"), 512)[1]
    gc_dev = prepared[3]
    tr_mask = train["threshold_accepted"].astype(bool).to_numpy()
    dv_mask = dev["threshold_accepted"].astype(bool).to_numpy()
    tr, dv = train.loc[tr_mask].reset_index(drop=True), dev.loc[dv_mask].reset_index(drop=True)
    tr_extra, dv_extra = tr_extra.loc[tr_mask].reset_index(drop=True), dv_extra.loc[dv_mask].reset_index(drop=True)
    gc_train = gc_train_all[tr_mask]
    correct_train = np.flatnonzero(tr["correct"].astype(bool).to_numpy())
    sample_grid = np.unique(np.r_[np.arange(0, len(grid), 4), len(grid) - 1])

    def expand(gc: np.ndarray, frame: pd.DataFrame, extra: pd.DataFrame, indices: np.ndarray, gi: np.ndarray) -> np.ndarray:
        static = np.c_[frame.iloc[indices]["p_side"].to_numpy(float),
                       (frame.iloc[indices]["selected_side"].astype(str).str.upper() == "UP").to_numpy(float),
                       extra.iloc[indices].to_numpy(float)].astype("float32")
        return np.c_[np.tile(grid[gi], len(indices)), gc[indices][:, gi].reshape(-1),
                     np.repeat(static, len(gi), axis=0)].astype("float32")

    xtr = expand(gc_train, tr, tr_extra, correct_train, sample_grid)
    lows = tr.iloc[correct_train]["chosen_low"].to_numpy(float)
    ytr = (lows[:, None] <= grid[sample_grid][None, :]).astype("uint8").reshape(-1)
    gc_model = LGBMClassifier(
        n_estimators=500, learning_rate=0.025, num_leaves=15, max_depth=5,
        min_child_samples=250, subsample=0.8, colsample_bytree=0.25,
        reg_lambda=20, reg_alpha=4, random_state=20260704, verbosity=-1, n_jobs=-1,
        monotone_constraints=[1, 1] + [0] * (xtr.shape[1] - 2),
    )
    gc_model.fit(xtr, ytr)
    idx = np.arange(len(dv))
    gc_path = gc_model.predict_proba(expand(gc_dev, dv, dv_extra, idx, np.arange(len(grid))))[:, 1].reshape(len(dv), len(grid))
    gc_path = np.maximum.accumulate(np.clip(gc_path, 0, 1), axis=1)
    correct_dev = dv["correct"].astype(bool).to_numpy()
    target_gc = (dv.loc[correct_dev, "chosen_low"].to_numpy(float)[:, None] <= grid[None, :]).astype(float)
    calibration = {name: {"brier": float(brier_score_loss(dv["correct"], q)),
                          "log_loss": float(log_loss(dv["correct"], np.clip(q, 1e-6, 1-1e-6)))} for name, q in qs.items()}
    return {"prepared": prepared, "qs": qs, "gc": {"base": gc_dev, "path": gc_path}, "grid": grid,
            "calibration": calibration, "gc_brier": {"base": float(np.mean((gc_dev[correct_dev]-target_gc)**2)),
                                                       "path": float(np.mean((gc_path[correct_dev]-target_gc)**2))},
            "path_feature_count": len(tr_path.columns)}


def evaluate(payload: dict, q_name: str, gc_name: str, floor: float, min_ev: float,
             min_q: float = 0.0, max_bid: float = 0.85) -> dict:
    dev, accepted = payload["prepared"][0], payload["prepared"][1]
    allowed = payload["grid"] <= max_bid + 1e-12
    q = payload["qs"][q_name]
    bid, ev, fill = joint.choose_survival_expected_return_bids(
        q, payload["gc"][gc_name][:, allowed], payload["grid"][allowed], 0.01, min_ev,
        min_fill_probability=floor)
    submit = q >= min_q
    bid = np.where(submit, bid, 0.0)
    ev = np.where(submit, ev, 0.0)
    fill = np.where(submit, fill, 0.0)
    return joint.backtest_metrics(accepted, joint.backtest_with_bid(accepted, bid, ev, fill), len(dev))


def main() -> None:
    prepared = {f: prepare_fold(f) for f in FOLDS}
    rows = []
    for q_name in prepared["w1"]["qs"]:
        for gc_name in ("base", "path"):
            for floor in (0.70, 0.80, 0.85, 0.90):
                for min_ev in (0.0, 0.01, 0.02, 0.03, 0.05):
                    for min_q in (0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
                        for max_bid in (0.50, 0.60, 0.70, 0.85):
                            metrics = {f: evaluate(prepared[f], q_name, gc_name, floor, min_ev, min_q, max_bid) for f in FOLDS}
                            pnl = {f: metrics[f]["sum_pnl"] for f in FOLDS}
                            tune = np.array([pnl[f] for f in TUNE])
                            rows.append({"q_model": q_name, "gc_model": gc_name, "floor": floor,
                                         "min_ev": min_ev, "min_q": min_q, "max_bid": max_bid,
                                         **{f"{f}_pnl": pnl[f] for f in FOLDS}, "tune_sum": float(tune.sum()),
                                         "tune_worst": float(tune.min()), "tune_robust": float(tune.sum()-tune.std()),
                                         "holdout_sum": float(sum(pnl[f] for f in HOLDOUT)),
                                         "tune_loss_sum": float(sum(metrics[f]["loss_pnl_sum"] for f in TUNE)),
                                         "tune_win_sum": float(sum(metrics[f]["win_pnl_sum"] for f in TUNE))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0]
    table.to_csv(SESSION / "path_signature_search.csv", index=False)
    summary = {"experiment_count": len(table), "winner": winner.to_dict(), "selection_folds": TUNE,
               "untouched_folds": HOLDOUT, "btest_used": False,
               "path_feature_count": prepared["w1"]["path_feature_count"],
               "q_calibration": {f: p["calibration"] for f, p in prepared.items()},
               "gc_brier": {f: p["gc_brier"] for f, p in prepared.items()},
               "top20": table.head(20).to_dict("records")}
    (SESSION / "path_signature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": summary["winner"], "path_feature_count": summary["path_feature_count"],
                      "gc_brier": summary["gc_brier"]}, indent=2))


if __name__ == "__main__":
    main()
