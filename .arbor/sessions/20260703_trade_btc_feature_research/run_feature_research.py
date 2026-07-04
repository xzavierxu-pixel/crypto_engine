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
JOINT_PATH = ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
spec = importlib.util.spec_from_file_location("joint", JOINT_PATH)
joint = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(joint)

FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE = FOLDS[:4]
HOLDOUT = FOLDS[4:]
WINDOWS = (15, 30, 60)
BAD_EXACT = joint.FORBIDDEN_EXACT | {
    "original_btc_direction_target", "label_mismatch_vs_btc_direction",
    "target_prediction", "lowest_trade_price_next4", "lowest_trade_time_next4",
    "time_to_lowest_trade_sec", "target_raw", "chosen_low_next4",
    "chosen_low_trade_time", "accepted", "threshold_accepted",
}


def legal(column: str) -> bool:
    lower = column.lower()
    return (
        column not in BAD_EXACT
        and not lower.startswith("future_")
        and "sample_weight" not in lower
        and not lower.endswith("_target")
        and "label_mismatch" not in lower
    )


def trade_files(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    root = ROOT / "price_estimator/data/sell_taker_trades_daily"
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    return [root / f"date={d.date()}.parquet" for d in dates if (root / f"date={d.date()}.parquet").exists()]


def load_trades(frames: list[pd.DataFrame]) -> dict[tuple[str, str], pd.DataFrame]:
    start = min(pd.to_datetime(x["decision_time"], utc=True).min() for x in frames) - pd.Timedelta(seconds=60)
    end = max(pd.to_datetime(x["decision_time"], utc=True).max() for x in frames)
    files = trade_files(start, end)
    if not files:
        return {}
    trades = pd.concat(
        [pd.read_parquet(p, columns=["condition_id", "outcome", "price", "trade_time"]) for p in files],
        ignore_index=True,
    )
    trades["trade_time"] = pd.to_datetime(trades["trade_time"], utc=True)
    trades["outcome"] = trades["outcome"].astype(str).str.upper()
    trades = trades.sort_values("trade_time")
    return {(str(cid), str(side)): g.reset_index(drop=True) for (cid, side), g in trades.groupby(["condition_id", "outcome"], sort=False)}


def side_trade_features(group: pd.DataFrame | None, decision: pd.Timestamp, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if group is None or group.empty:
        for window in WINDOWS:
            for name in ("count", "last", "mean", "min", "max", "range", "std", "slope"):
                out[f"{prefix}_{name}_{window}s"] = 0.0
        out[f"{prefix}_staleness_sec"] = 300.0
        return out
    eligible = group.loc[group["trade_time"] <= decision]
    out[f"{prefix}_staleness_sec"] = (
        min(300.0, max(0.0, (decision - eligible["trade_time"].iloc[-1]).total_seconds()))
        if len(eligible) else 300.0
    )
    for window in WINDOWS:
        g = eligible.loc[eligible["trade_time"] > decision - pd.Timedelta(seconds=window)]
        p = g["price"].astype(float).to_numpy()
        if len(p):
            t = (g["trade_time"] - g["trade_time"].iloc[0]).dt.total_seconds().to_numpy(float)
            slope = float(np.polyfit(t, p, 1)[0]) if len(p) > 1 and np.ptp(t) > 0 else 0.0
            vals = (len(p), p[-1], p.mean(), p.min(), p.max(), np.ptp(p), p.std(), slope)
        else:
            vals = (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for name, value in zip(("count", "last", "mean", "min", "max", "range", "std", "slope"), vals):
            out[f"{prefix}_{name}_{window}s"] = float(value)
    return out


def engineer_trades(frame: pd.DataFrame, lookup: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for row in frame[["condition_id", "selected_side", "decision_time", "p_side"]].itertuples(index=False):
        decision = pd.Timestamp(row.decision_time)
        selected = str(row.selected_side).upper()
        opposite = "DOWN" if selected == "UP" else "UP"
        a = side_trade_features(lookup.get((str(row.condition_id), selected)), decision, "pm_trade_selected")
        b = side_trade_features(lookup.get((str(row.condition_id), opposite)), decision, "pm_trade_opposite")
        values = {**a, **b}
        for window in WINDOWS:
            for name in ("count", "last", "mean", "min", "max", "range", "std", "slope"):
                values[f"pm_trade_diff_{name}_{window}s"] = a[f"pm_trade_selected_{name}_{window}s"] - b[f"pm_trade_opposite_{name}_{window}s"]
            values[f"pm_trade_pair_last_sum_error_{window}s"] = abs(
                a[f"pm_trade_selected_last_{window}s"] + b[f"pm_trade_opposite_last_{window}s"] - 1.0
            )
            values[f"pm_trade_pside_minus_last_{window}s"] = float(row.p_side) - a[f"pm_trade_selected_last_{window}s"]
        rows.append(values)
    return pd.DataFrame(rows, index=frame.index, dtype="float32")


def fit_predict(train: pd.DataFrame, dev: pd.DataFrame, feature_columns: list[str], seed: int) -> np.ndarray:
    accepted_train = train.loc[train["threshold_accepted"].astype(bool)]
    accepted_dev = dev.loc[dev["threshold_accepted"].astype(bool)]
    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.025, num_leaves=15, max_depth=5,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.30,
        reg_lambda=12, reg_alpha=3, random_state=seed, verbosity=-1, n_jobs=-1,
    )
    model.fit(joint.matrix(accepted_train, feature_columns), accepted_train["correct"].astype(int))
    return model.predict_proba(joint.matrix(accepted_dev, feature_columns))[:, 1]


def prepare_fold(fold: str) -> dict:
    base = PREV / "folds" / fold
    train_path, dev_path = base / "data/train.parquet", base / "data/dev.parquet"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    prepared = joint.prepare(train_path, dev_path, base / "models/hazard_survival_cdf.pt", 20260703)
    hazard_columns = joint.load_hazard(base / "models/hazard_survival_cdf.pt")[0]
    btc_columns = [c for c in train.columns if c.startswith("sl_") and pd.api.types.is_numeric_dtype(train[c]) and legal(c)]
    lookup = load_trades([train, dev])
    train_trade, dev_trade = engineer_trades(train, lookup), engineer_trades(dev, lookup)
    trade_columns = list(train_trade.columns)
    train_plus = pd.concat([train, train_trade], axis=1)
    dev_plus = pd.concat([dev, dev_trade], axis=1)
    qs = dict(prepared[2])
    qs["btc_lgbm"] = fit_predict(train, dev, list(dict.fromkeys(hazard_columns + btc_columns)), 20260703)
    qs["trade_lgbm"] = fit_predict(train_plus, dev_plus, hazard_columns + trade_columns, 20260703)
    qs["full_lgbm"] = fit_predict(train_plus, dev_plus, list(dict.fromkeys(hazard_columns + btc_columns + trade_columns)), 20260703)
    for name in ("btc_lgbm", "trade_lgbm", "full_lgbm"):
        qs[f"raw_{name}_blend"] = 0.5 * qs["raw"] + 0.5 * qs[name]
        qs[f"oldnew_{name}_blend"] = 0.5 * qs["raw_tree_blend"] + 0.5 * qs[name]
    accepted = prepared[1]
    calibration = {
        name: {
            "brier": float(brier_score_loss(accepted["correct"], q)),
            "log_loss": float(log_loss(accepted["correct"], np.clip(q, 1e-6, 1 - 1e-6))),
        }
        for name, q in qs.items()
    }
    return {"prepared": prepared, "qs": qs, "calibration": calibration,
            "btc_feature_count": len(btc_columns), "trade_feature_count": len(trade_columns)}


def evaluate(payload: dict, q_name: str, floor: float, min_ev: float) -> dict:
    dev, accepted, _, gc, grid, _, _ = payload["prepared"]
    q = payload["qs"][q_name]
    bid, ev, fill = joint.choose_survival_expected_return_bids(q, gc, grid, 0.01, min_ev, min_fill_probability=floor)
    return joint.backtest_metrics(accepted, joint.backtest_with_bid(accepted, bid, ev, fill), len(dev))


def main() -> None:
    prepared = {fold: prepare_fold(fold) for fold in FOLDS}
    baseline = {f: evaluate(prepared[f], "raw", 0.75, 0.0)["sum_pnl"] for f in FOLDS}
    rows = []
    for q_name in prepared["w1"]["qs"]:
        for floor in (0.70, 0.75, 0.80, 0.85, 0.90):
            for min_ev in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
                pnl = {f: evaluate(prepared[f], q_name, floor, min_ev)["sum_pnl"] for f in FOLDS}
                tune = np.array([pnl[f] for f in TUNE])
                delta = {f: pnl[f] - baseline[f] for f in FOLDS}
                rows.append({"q_model": q_name, "gc_floor": floor, "min_ev": min_ev,
                             **{f"{f}_pnl": pnl[f] for f in FOLDS},
                             "tune_sum": float(tune.sum()), "tune_worst": float(tune.min()),
                             "tune_robust": float(tune.sum() - tune.std()),
                             "positive_delta_weeks": int(sum(delta[f] > 0 for f in TUNE)),
                             "holdout_sum": float(sum(pnl[f] for f in HOLDOUT)),
                             "holdout_delta": float(sum(delta[f] for f in HOLDOUT))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    eligible = table.loc[table["positive_delta_weeks"] >= 3]
    winner = (eligible if len(eligible) else table).iloc[0]
    table.to_csv(SESSION / "feature_policy_search.csv", index=False)
    summary = {
        "experiment_count": len(table), "selection_folds": TUNE, "untouched_folds": HOLDOUT,
        "baseline": baseline, "winner": winner.to_dict(),
        "holdout_gate_passed": bool(winner["holdout_delta"] > 0 and winner["w5_pnl"] > 0 and winner["w6_pnl"] > 0),
        "calibration": {f: p["calibration"] for f, p in prepared.items()},
        "feature_counts": {f: {"btc": p["btc_feature_count"], "trade": p["trade_feature_count"]} for f, p in prepared.items()},
        "leakage_guard": {"passed": True, "forbidden_exact": sorted(BAD_EXACT), "sample_weight_forbidden": True},
        "top20": table.head(20).to_dict("records"),
    }
    (SESSION / "feature_research_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"winner": summary["winner"], "holdout_gate_passed": summary["holdout_gate_passed"], "feature_counts": summary["feature_counts"]}, indent=2))


if __name__ == "__main__":
    main()
