#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
EXPERIMENTS = SESSION / "experiments"
CACHE = SESSION / "cache"
TRADE_DIR = ROOT / "price_estimator" / "data" / "sell_taker_trades_daily"
L2_DIR = ROOT / "artifacts" / "data_v2" / "polymarket_l2"
L2_FEATURE_DIR = L2_DIR / "first_minute_features"
ROLLING = ROOT / ".arbor" / "sessions" / "20260703_prefinal_rolling" / "folds"
BASELINE_MANIFEST = ROOT / "execution_engine" / "deploy" / "baseline" / "artifact_manifest.json"
TRAIN = ROOT / "price_estimator" / "expected_return" / "experiments" / "20260619_expected_return_trade_coverage_start" / "data" / "expected_return_train.parquet"
BTEST = ROOT / "price_estimator" / "expected_return" / "experiments" / "20260619_expected_return_trade_coverage_start" / "data" / "expected_return_validation.parquet"

FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE = FOLDS[:4]
HOLDOUT = FOLDS[4:]
CS = [0.00, 0.01, 0.02]
CAPS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00]
BAND_FLOORS = [0.00, 0.35, 0.45, 0.55]
BAND_CAPS = [0.50, 0.60, 0.70, 0.80, 1.00]
ACTION_NAMES = ["BUY_YES", "BUY_NO", "NO_TRADE"]
ACTION_YES = 0
ACTION_NO = 1
ACTION_NONE = 2
OUTCOME_MAP = {"YES": "UP", "NO": "DOWN"}
PRICE_BUCKETS = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
FORBIDDEN_EXACT = {
    "target",
    "abs_return",
    "signed_return",
    "stage1_target",
    "stage2_target",
    "stage1_sample_weight",
    "chosen_low",
    "chosen_low_trade_time",
    "time_to_chosen_low_sec",
    "correct",
    "winner",
    "pnl",
    "y",
    "trade_time",
    "endDate",
    "condition_id",
    "market_id",
    "slug",
    "outcome",
    "accepted",
    "threshold_accepted",
    "predicted_side",
    "m_yes",
    "m_no",
    "m_chosen",
}
FORBIDDEN_PREFIXES = ("future_", "chosen_low")


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def split_paths(name: str) -> tuple[Path, Path]:
    if name == "btest":
        return TRAIN, BTEST
    fold_dir = ROLLING / name / "data"
    return fold_dir / "train.parquet", fold_dir / "dev.parquet"


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_feature_columns() -> list[str]:
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    cols = list(manifest["feature_columns"])
    bad = forbidden_intersection(cols)
    if bad:
        raise RuntimeError(f"forbidden deploy features: {bad}")
    return cols


def forbidden_intersection(cols: list[str]) -> list[str]:
    return [
        c
        for c in cols
        if c in FORBIDDEN_EXACT or any(c.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    ]


def matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = df.reindex(columns=feature_columns).copy()
    for col in x.columns:
        if not pd.api.types.is_numeric_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).astype("float32")


def trade_files(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    dates = set(pd.date_range(start.normalize(), end.normalize(), freq="D").strftime("%Y-%m-%d"))
    return sorted(p for p in TRADE_DIR.glob("date=*.parquet") if p.stem.removeprefix("date=") in dates)


def load_trades(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    files = trade_files(start, end)
    if not files:
        return pd.DataFrame(columns=["condition_id", "outcome", "price", "trade_time"])
    trades = pd.concat(
        [
            pd.read_parquet(path, columns=["condition_id", "outcome", "price", "trade_time"])
            for path in files
        ],
        ignore_index=True,
    )
    trades["trade_time"] = pd.to_datetime(trades["trade_time"], utc=True)
    trades["outcome"] = trades["outcome"].astype(str).str.upper()
    trades["condition_id"] = trades["condition_id"].astype(str)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    return trades.dropna(subset=["price", "trade_time"])


def load_trade_lookup(start: pd.Timestamp, end: pd.Timestamp) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    trades = load_trades(start, end)
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, group in trades.groupby(["condition_id", "outcome"], sort=False):
        ordered = group.sort_values("trade_time")
        lookup[key] = (
            ordered["trade_time"].astype("int64").to_numpy(),
            ordered["price"].to_numpy(dtype=float),
        )
    return lookup


def last_price_before(
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    condition_id: str,
    outcome: str,
    cutoff_ns: int,
) -> tuple[float, pd.Timestamp | pd.NaT, bool]:
    payload = lookup.get((condition_id, outcome))
    if payload is None:
        return float("nan"), pd.NaT, False
    times, prices = payload
    idx = int(np.searchsorted(times, cutoff_ns, side="right") - 1)
    if idx < 0:
        return float("nan"), pd.NaT, False
    return float(prices[idx]), pd.to_datetime(times[idx], utc=True), True


def window_stats(times: np.ndarray, prices: np.ndarray, start_ns: int, end_ns: int) -> dict[str, float]:
    left = int(np.searchsorted(times, start_ns, side="left"))
    right = int(np.searchsorted(times, end_ns, side="right"))
    vals = prices[left:right]
    if len(vals) == 0:
        return {
            "count": 0.0,
            "last": np.nan,
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "std": np.nan,
            "slope": np.nan,
            "down_moves": np.nan,
            "low_share": np.nan,
        }
    x = np.arange(len(vals), dtype=float)
    slope = float(np.polyfit(x, vals, 1)[0]) if len(vals) >= 2 else 0.0
    diffs = np.diff(vals)
    return {
        "count": float(len(vals)),
        "last": float(vals[-1]),
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "range": float(np.max(vals) - np.min(vals)),
        "std": float(np.std(vals)) if len(vals) >= 2 else 0.0,
        "slope": slope,
        "down_moves": float((diffs < 0).sum()) if len(diffs) else 0.0,
        "low_share": float((vals <= 0.35).mean()),
    }


def build_market_frame(name: str, kind: str, df: pd.DataFrame) -> pd.DataFrame:
    out_path = CACHE / "market_frames" / f"{name}_{kind}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    market_t0 = pd.to_datetime(df["market_t0"], utc=True)
    decision_time = market_t0 + pd.Timedelta(seconds=68)
    lag_cutoff = decision_time - pd.Timedelta(seconds=60)
    lookup = load_trade_lookup(market_t0.min() - pd.Timedelta(minutes=3), decision_time.max())
    decision_ns = decision_time.astype("int64").to_numpy()
    lag_ns = lag_cutoff.astype("int64").to_numpy()
    targets = df["target"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(df.itertuples(index=False)):
        cid = str(row.condition_id)
        m_yes, m_yes_time, has_yes = last_price_before(lookup, cid, OUTCOME_MAP["YES"], int(decision_ns[idx]))
        m_no, m_no_time, has_no = last_price_before(lookup, cid, OUTCOME_MAP["NO"], int(decision_ns[idx]))
        m_yes_lag, _, has_yes_lag = last_price_before(lookup, cid, OUTCOME_MAP["YES"], int(lag_ns[idx]))
        m_no_lag, _, has_no_lag = last_price_before(lookup, cid, OUTCOME_MAP["NO"], int(lag_ns[idx]))
        record: dict[str, Any] = {
            "sample_id": f"{name}_{kind}_{idx}",
            "condition_id": cid,
            "market_t0": market_t0.iloc[idx],
            "decision_time": decision_time.iloc[idx],
            "y": int(targets[idx]),
            "m_yes": m_yes,
            "m_no": m_no,
            "m_yes_trade_time": m_yes_time,
            "m_no_trade_time": m_no_time,
            "m_yes_lag1m": m_yes_lag,
            "m_no_lag1m": m_no_lag,
            "d_yes_1m": m_yes - m_yes_lag if has_yes and has_yes_lag else np.nan,
            "d_no_1m": m_no - m_no_lag if has_no and has_no_lag else np.nan,
            "slope_yes_1m": (m_yes - m_yes_lag) / 60.0 if has_yes and has_yes_lag else np.nan,
            "slope_no_1m": (m_no - m_no_lag) / 60.0 if has_no and has_no_lag else np.nan,
            "yes_no_lag1m_spread": m_yes_lag - m_no_lag if has_yes_lag and has_no_lag else np.nan,
            "d_yes_minus_d_no_1m": (m_yes - m_yes_lag) - (m_no - m_no_lag)
            if has_yes and has_yes_lag and has_no and has_no_lag
            else np.nan,
            "has_yes": bool(has_yes),
            "has_no": bool(has_no),
            "has_any": bool(has_yes or has_no),
            "has_lag1m_yes": bool(has_yes_lag),
            "has_lag1m_no": bool(has_no_lag),
            "has_lag1m": bool(has_yes_lag or has_no_lag),
        }
        for outcome, prefix in [(OUTCOME_MAP["YES"], "yes"), (OUTCOME_MAP["NO"], "no")]:
            payload = lookup.get((cid, outcome))
            times = np.array([], dtype=np.int64) if payload is None else payload[0]
            prices = np.array([], dtype=float) if payload is None else payload[1]
            for seconds in [10, 30, 60, 120]:
                stats = window_stats(times, prices, int(decision_ns[idx] - seconds * 1_000_000_000), int(decision_ns[idx]))
                for key, value in stats.items():
                    record[f"trade_{prefix}_{seconds}s_{key}"] = value
            last_time = m_yes_time if prefix == "yes" else m_no_time
            record[f"trade_{prefix}_recency_sec"] = (
                (decision_time.iloc[idx] - last_time).total_seconds() if pd.notna(last_time) else np.nan
            )
        record["trade_last_yes_minus_no"] = m_yes - m_no if has_yes and has_no else np.nan
        record["trade_last_yes_no_ratio"] = m_yes / m_no if has_yes and has_no and m_no else np.nan
        record["trade_complement_deviation"] = abs(m_yes + m_no - 1.0) if has_yes and has_no else np.nan
        rows.append(record)
    out = pd.DataFrame(rows)
    late_yes = out["has_yes"] & (pd.to_datetime(out["m_yes_trade_time"], utc=True) > out["decision_time"])
    late_no = out["has_no"] & (pd.to_datetime(out["m_no_trade_time"], utc=True) > out["decision_time"])
    if bool(late_yes.any() or late_no.any()):
        raise RuntimeError(f"late join detected in {name}/{kind}")
    out.to_parquet(out_path, index=False)
    return out


def load_l2_feature_store() -> pd.DataFrame:
    out_path = CACHE / "l2_first_minute_features.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    files = sorted(L2_FEATURE_DIR.glob("date=*/part-*.parquet"))
    if not files:
        return pd.DataFrame(columns=["market_t0", "l2_covered"])
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["market_t0"] = pd.to_datetime(frame["market_t0"], utc=True)
    frame["l2_covered"] = frame.get("pm_l2_1m_available", 0).fillna(0).astype(float) > 0
    drop = {"market_slug", "feature_cutoff_time", "max_feature_event_time"}
    keep = ["market_t0", "l2_covered"] + [
        c for c in frame.columns if c.startswith("pm_l2_1m_") and c not in drop
    ]
    out = frame[keep].drop_duplicates("market_t0", keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def attach_research_features(splits: dict[str, dict[str, pd.DataFrame]], base_columns: list[str]) -> dict[str, list[str]]:
    l2_store = load_l2_feature_store()
    feature_sets: dict[str, list[str]] = {"F1_baseline_replay": base_columns}
    trade_cols: list[str] = []
    lag_cols = [
        "m_yes_lag1m",
        "m_no_lag1m",
        "d_yes_1m",
        "d_no_1m",
        "slope_yes_1m",
        "slope_no_1m",
        "yes_no_lag1m_spread",
        "d_yes_minus_d_no_1m",
        "has_lag1m_yes",
        "has_lag1m_no",
        "has_lag1m",
    ]
    l2_cols = [c for c in l2_store.columns if c not in {"market_t0", "l2_covered"}]
    for name, payload in splits.items():
        for kind in ["train", "dev"]:
            df_key = f"{kind}_df"
            market_key = f"{kind}_market"
            df = payload[df_key].copy()
            market = payload[market_key]
            additions = market.drop(columns=["sample_id", "condition_id", "market_t0", "decision_time", "y"])
            for col in additions.columns:
                if col.startswith("trade_") and col not in trade_cols:
                    trade_cols.append(col)
            df = pd.concat([df.reset_index(drop=True), additions.reset_index(drop=True)], axis=1)
            df["market_t0"] = pd.to_datetime(df["market_t0"], utc=True)
            df = df.merge(l2_store, on="market_t0", how="left")
            df["l2_covered"] = df["l2_covered"].where(df["l2_covered"].notna(), False).astype(bool)
            payload[df_key] = df
            payload[market_key] = market.merge(
                df[["market_t0", "l2_covered"]], on="market_t0", how="left"
            )
    feature_sets["F2_trade_path"] = base_columns + trade_cols
    feature_sets["F3_l2"] = base_columns + l2_cols + ["l2_covered"]
    feature_sets["F4_lag1m_price"] = base_columns + lag_cols
    feature_sets["F5_trade_l2_combined"] = base_columns + trade_cols + lag_cols + l2_cols + ["l2_covered"]
    return {k: [c for c in v if c not in FORBIDDEN_EXACT] for k, v in feature_sets.items()}


def compute_rewards(market: pd.DataFrame, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = market["y"].to_numpy(dtype=float)
    has_yes = market["has_yes"].to_numpy(dtype=bool)
    has_no = market["has_no"].to_numpy(dtype=bool)
    m_yes = market["m_yes"].to_numpy(dtype=float)
    m_no = market["m_no"].to_numpy(dtype=float)
    reward_yes = np.where(has_yes, y - m_yes - c, -np.inf)
    reward_no = np.where(has_no, (1.0 - y) - m_no - c, -np.inf)
    reward_none = np.zeros(len(market), dtype=float)
    stack = np.column_stack([reward_yes, reward_no, reward_none])
    oracle_action = np.argmax(stack, axis=1)
    top = stack[np.arange(len(stack)), oracle_action]
    second = np.partition(stack, -2, axis=1)[:, -2]
    margin = np.maximum(top - second, 0.0)
    weight = np.clip(np.log1p(margin / 0.02), 0.25, 5.0)
    oracle_pnl = np.max(
        np.column_stack(
            [
                np.where(has_yes, y - m_yes, -np.inf),
                np.where(has_no, (1.0 - y) - m_no, -np.inf),
                np.zeros(len(market), dtype=float),
            ]
        ),
        axis=1,
    )
    return oracle_action, weight, np.where(np.isfinite(oracle_pnl), oracle_pnl, 0.0), stack


def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    curve = np.cumsum(pnl)
    peak = np.maximum.accumulate(curve)
    return float((peak - curve).max())


def weekly_worst_pnl(decision_time: pd.Series, pnl: np.ndarray) -> float:
    frame = pd.DataFrame({"decision_time": pd.to_datetime(decision_time, utc=True), "pnl": pnl})
    weekly = frame.groupby(pd.Grouper(key="decision_time", freq="W-MON"))["pnl"].sum()
    return float(weekly.min()) if len(weekly) else 0.0


def evaluate_actions(
    market: pd.DataFrame,
    action_idx: np.ndarray,
    pi_yes: np.ndarray,
    pi_no: np.ndarray,
    pi_none: np.ndarray,
    cap_applied: str | float = "none",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = market["y"].to_numpy(dtype=int)
    m_yes = market["m_yes"].to_numpy(dtype=float)
    m_no = market["m_no"].to_numpy(dtype=float)
    has_yes = market["has_yes"].to_numpy(dtype=bool)
    has_no = market["has_no"].to_numpy(dtype=bool)
    oracle_action, _, oracle_pnl, _ = compute_rewards(market, 0.0)
    action_idx = action_idx.astype(int)
    action_idx = np.where((action_idx == ACTION_YES) & (~has_yes), ACTION_NONE, action_idx)
    action_idx = np.where((action_idx == ACTION_NO) & (~has_no), ACTION_NONE, action_idx)
    realized = np.zeros(len(market), dtype=float)
    yes_mask = action_idx == ACTION_YES
    no_mask = action_idx == ACTION_NO
    realized[yes_mask] = y[yes_mask] - m_yes[yes_mask]
    realized[no_mask] = (1 - y[no_mask]) - m_no[no_mask]
    trade_mask = action_idx != ACTION_NONE
    m_chosen = np.where(yes_mask, m_yes, np.where(no_mask, m_no, np.nan))
    correct = np.where(yes_mask, y == 1, np.where(no_mask, y == 0, False))
    wrong_buy_yes = yes_mask & (y == 0)
    wrong_buy_no = no_mask & (y == 1)
    wrong_side_mask = wrong_buy_yes | wrong_buy_no
    avg_entry = float(np.nanmean(m_chosen[trade_mask])) if trade_mask.any() else 0.0
    accuracy = float(correct[trade_mask].mean()) if trade_mask.any() else 0.0
    mean_pnl = float(realized[trade_mask].mean()) if trade_mask.any() else 0.0
    metrics = {
        "sample_count": int(len(market)),
        "trade_count": int(trade_mask.sum()),
        "trade_rate": float(trade_mask.mean()) if len(market) else 0.0,
        "accuracy": accuracy,
        "avg_entry_price": avg_entry,
        "edge": accuracy - avg_entry,
        "sum_pnl": float(realized.sum()),
        "mean_pnl_per_trade": mean_pnl,
        "pnl_identity_error": float((accuracy - avg_entry) - mean_pnl) if trade_mask.any() else 0.0,
        "YES_pnl": float(realized[yes_mask].sum()),
        "NO_pnl": float(realized[no_mask].sum()),
        "win_pnl_sum": float(realized[trade_mask & (realized > 0)].sum()),
        "loss_pnl_sum": float(realized[trade_mask & (realized < 0)].sum()),
        "wrong_side_loss": float(realized[wrong_side_mask].sum()),
        "wrong_side_loss_abs": float(abs(realized[wrong_side_mask].sum())),
        "wrong_buy_yes_count": int(wrong_buy_yes.sum()),
        "wrong_buy_no_count": int(wrong_buy_no.sum()),
        "wrong_buy_yes_loss": float(realized[wrong_buy_yes].sum()),
        "wrong_buy_no_loss": float(realized[wrong_buy_no].sum()),
        "no_trade_count": int((action_idx == ACTION_NONE).sum()),
        "oracle_pnl": float(oracle_pnl.sum()),
        "capture_ratio": float(realized.sum() / oracle_pnl.sum()) if float(oracle_pnl.sum()) else 0.0,
        "worst_week_pnl": weekly_worst_pnl(market["decision_time"], realized),
        "selection_drawdown": max_drawdown(realized),
        "m_yes_coverage": float(has_yes.mean()),
        "m_no_coverage": float(has_no.mean()),
        "m_any_coverage": float((has_yes | has_no).mean()),
        "lag1m_coverage": float(market.get("has_lag1m", pd.Series(False, index=market.index)).mean()),
        "l2_coverage": float(market.get("l2_covered", pd.Series(False, index=market.index)).fillna(False).mean()),
        "late_join_count": 0,
    }
    pred = pd.DataFrame(
        {
            "sample_id": market["sample_id"],
            "decision_time": market["decision_time"],
            "y": y,
            "m_yes": m_yes,
            "m_no": m_no,
            "m_chosen": m_chosen,
            "m_yes_lag1m": market.get("m_yes_lag1m", np.nan),
            "m_no_lag1m": market.get("m_no_lag1m", np.nan),
            "d_yes_1m": market.get("d_yes_1m", np.nan),
            "d_no_1m": market.get("d_no_1m", np.nan),
            "pi_yes": pi_yes,
            "pi_no": pi_no,
            "pi_none": pi_none,
            "action": [ACTION_NAMES[i] for i in action_idx],
            "cap_applied": cap_applied,
            "realized_pnl": realized,
            "oracle_action": [ACTION_NAMES[i] for i in oracle_action],
            "oracle_pnl": oracle_pnl,
            "gap_to_oracle": oracle_pnl - realized,
            "has_yes": has_yes,
            "has_no": has_no,
            "l2_covered": market.get("l2_covered", pd.Series(False, index=market.index)).fillna(False).astype(bool),
        }
    )
    confusion = pred.groupby(["action", "oracle_action"], dropna=False).agg(
        count=("sample_id", "size"),
        realized_pnl=("realized_pnl", "sum"),
        gap_to_oracle=("gap_to_oracle", "sum"),
    ).reset_index()
    wrong = pd.DataFrame(
        [
            {"bucket": "wrong_buy_yes", "count": int(wrong_buy_yes.sum()), "realized_pnl": float(realized[wrong_buy_yes].sum())},
            {"bucket": "wrong_buy_no", "count": int(wrong_buy_no.sum()), "realized_pnl": float(realized[wrong_buy_no].sum())},
            {"bucket": "no_trade", "count": int((action_idx == ACTION_NONE).sum()), "realized_pnl": 0.0},
        ]
    )
    edge = edge_by_price_bucket(pred)
    return metrics, pred, confusion, wrong, edge


def edge_by_price_bucket(pred: pd.DataFrame) -> pd.DataFrame:
    traded = pred[pred["action"] != "NO_TRADE"].copy()
    if traded.empty:
        return pd.DataFrame(columns=["bucket", "n", "mean_price", "win_rate", "edge", "bucket_pnl"])
    traded["correct"] = traded["realized_pnl"] > 0
    traded["bucket"] = pd.cut(traded["m_chosen"], PRICE_BUCKETS, include_lowest=True)
    out = traded.groupby("bucket", observed=False).agg(
        n=("sample_id", "size"),
        mean_price=("m_chosen", "mean"),
        win_rate=("correct", "mean"),
        bucket_pnl=("realized_pnl", "sum"),
    ).reset_index()
    out["bucket"] = out["bucket"].astype(str)
    out["edge"] = out["win_rate"] - out["mean_price"]
    return out


def fit_model(train_df: pd.DataFrame, train_market: pd.DataFrame, feature_columns: list[str], c: float) -> xgb.XGBClassifier:
    x = matrix(train_df, feature_columns)
    y, w, _, _ = compute_rewards(train_market, c)
    order = pd.to_datetime(train_market["decision_time"], utc=True).sort_values().index.to_numpy()
    split = min(max(int(len(order) * 0.85), 500), len(order) - 100) if len(order) > 600 else max(len(order) - 1, 1)
    fit_idx = order[:split]
    val_idx = order[split:] if split < len(order) else order[-1:]
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_lambda=5.0,
        n_estimators=350,
        eval_metric="mlogloss",
        early_stopping_rounds=40,
        random_state=20260709,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(
        x.iloc[fit_idx],
        y[fit_idx],
        sample_weight=w[fit_idx],
        eval_set=[(x.iloc[val_idx], y[val_idx])],
        sample_weight_eval_set=[w[val_idx]],
        verbose=False,
    )
    return model


def predict_model(model: xgb.XGBClassifier, df: pd.DataFrame, market: pd.DataFrame, feature_columns: list[str]):
    probs = model.predict_proba(matrix(df, feature_columns))
    masked = probs.copy()
    masked[~market["has_yes"].to_numpy(dtype=bool), ACTION_YES] = -np.inf
    masked[~market["has_no"].to_numpy(dtype=bool), ACTION_NO] = -np.inf
    action = masked.argmax(axis=1)
    return evaluate_actions(market, action, probs[:, 0], probs[:, 1], probs[:, 2])


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return pd.DataFrame(candidates).sort_values(
        ["tune_score", "tune_edge", "tune_wrong_side_loss_abs", "tune_drawdown"],
        ascending=[False, False, True, True],
    ).iloc[0].to_dict()


def write_node(
    node_name: str,
    config_text: str,
    metrics_bdev: dict[str, Any],
    metrics_btest: dict[str, Any],
    predictions_btest: pd.DataFrame,
    confusion_btest: pd.DataFrame,
    wrong_side_btest: pd.DataFrame,
    edge_btest: pd.DataFrame,
    feature_columns: list[str],
    research_additions: list[str],
    report_text: str,
    cap_sweep: pd.DataFrame | None = None,
) -> None:
    out = EXPERIMENTS / node_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(config_text, encoding="utf-8")
    dump_json(out / "metrics_bdev.json", metrics_bdev)
    dump_json(out / "metrics_btest.json", metrics_btest)
    predictions_btest.to_parquet(out / "predictions_btest.parquet", index=False)
    confusion_btest.to_csv(out / "confusion_btest.csv", index=False)
    wrong_side_btest.to_csv(out / "wrong_side_decomposition_btest.csv", index=False)
    edge_btest.to_csv(out / "edge_by_price_bucket_btest.csv", index=False)
    if cap_sweep is not None:
        cap_sweep.to_csv(out / "cap_sweep_btest.csv", index=False)
    bad = forbidden_intersection(feature_columns)
    dump_json(
        out / "feature_manifest.json",
        {
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "research_additions": research_additions,
            "deploy_manifest_path": str(BASELINE_MANIFEST.relative_to(ROOT)),
            "deploy_intersection_count": len([c for c in feature_columns if c not in research_additions]),
        },
    )
    dump_json(out / "leakage_check.json", {"forbidden_feature_intersection": bad, "passed": not bad})
    if bad:
        raise RuntimeError(f"leakage check failed for {node_name}: {bad}")
    (out / "REPORT.md").write_text(report_text, encoding="utf-8")


def run_f0(feature_sets: dict[str, list[str]], splits: dict[str, dict[str, pd.DataFrame]], base_columns: list[str]) -> dict[str, Any]:
    btest_market = splits["btest"]["dev_market"]
    qa = {
        "row_count": int(len(btest_market)),
        "m_yes_coverage": float(btest_market["has_yes"].mean()),
        "m_no_coverage": float(btest_market["has_no"].mean()),
        "any_side_coverage": float(btest_market["has_any"].mean()),
        "late_join_count": 0,
        "trade_path_coverage": float((btest_market["has_yes"] | btest_market["has_no"]).mean()),
        "l2_coverage": float(btest_market["l2_covered"].fillna(False).mean()),
        "lag1m_coverage": float(btest_market["has_lag1m"].mean()),
        "feature_sets": {k: {"count": len(v), "research_additions": [c for c in v if c not in base_columns]} for k, v in feature_sets.items()},
    }
    n = len(btest_market)
    metrics, pred, confusion, wrong, edge = evaluate_actions(
        btest_market,
        np.full(n, ACTION_NONE, dtype=int),
        np.zeros(n),
        np.zeros(n),
        np.ones(n),
    )
    write_node(
        "F0_frame_qa",
        "experiment_id: F0_frame_qa\ndecision_offset_seconds: 68\nlag_cutoff: decision_time_minus_60s\n",
        {"folds": {f: {"row_count": int(len(splits[f]["dev_market"]))} for f in FOLDS}},
        {"qa": qa, "sample_metrics": metrics, "btest_evaluation_count": 1},
        pred,
        confusion,
        wrong,
        edge,
        feature_sets["F1_baseline_replay"],
        [],
        "# F0 frame QA\n\nUniverse and research feature coverage were materialized without changing deploy artifacts.\n",
    )
    return {"experiment_id": "F0_frame_qa", "feature_family": "qa", "B_dev": None, "B_test": None, "delta_vs137": None, "accuracy": None, "avg_entry": None, "edge": None, "leakage_passed": True, "notes": "qa_once"}


def run_feature_node(node_name: str, feature_columns: list[str], base_columns: list[str], splits: dict[str, dict[str, pd.DataFrame]]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    fold_metrics_by_c: dict[str, dict[str, Any]] = {}
    fold_preds_by_c: dict[str, dict[str, pd.DataFrame]] = {}
    for c in CS:
        key = f"c={c:.2f}"
        fold_metrics_by_c[key] = {}
        fold_preds_by_c[key] = {}
        tune_pnl: list[np.ndarray] = []
        for fold in FOLDS:
            model = fit_model(splits[fold]["train_df"], splits[fold]["train_market"], feature_columns, c)
            metrics, pred, _, _, _ = predict_model(model, splits[fold]["dev_df"], splits[fold]["dev_market"], feature_columns)
            fold_metrics_by_c[key][fold] = metrics
            fold_preds_by_c[key][fold] = pred
            if fold in TUNE:
                tune_pnl.append(pred["realized_pnl"].to_numpy(dtype=float))
        tune_concat = np.concatenate(tune_pnl) if tune_pnl else np.array([], dtype=float)
        tune_metrics = [fold_metrics_by_c[key][fold] for fold in TUNE]
        holdout_sum = float(sum(fold_metrics_by_c[key][fold]["sum_pnl"] for fold in HOLDOUT))
        candidates.append(
            {
                "c": c,
                "tune_score": float(sum(m["sum_pnl"] for m in tune_metrics)),
                "tune_edge": float(np.mean([m["edge"] for m in tune_metrics])),
                "tune_wrong_side_loss_abs": float(sum(m["wrong_side_loss_abs"] for m in tune_metrics)),
                "tune_drawdown": max_drawdown(tune_concat),
                "holdout_sum": holdout_sum,
                "holdout_passed": bool(holdout_sum > 0 and all(fold_metrics_by_c[key][fold]["sum_pnl"] > 0 for fold in HOLDOUT)),
                **{f"{fold}_sum_pnl": fold_metrics_by_c[key][fold]["sum_pnl"] for fold in FOLDS},
            }
        )
    winner = select_candidate(candidates)
    key = f"c={winner['c']:.2f}"
    btest_model = fit_model(splits["btest"]["train_df"], splits["btest"]["train_market"], feature_columns, float(winner["c"]))
    btest_metrics, btest_pred, confusion, wrong, edge = predict_model(
        btest_model, splits["btest"]["dev_df"], splits["btest"]["dev_market"], feature_columns
    )
    btest_metrics["btest_evaluation_count"] = 1
    btest_metrics["holdout_passed"] = bool(winner["holdout_passed"])
    metrics_bdev = {
        "selection_folds": TUNE,
        "holdout_folds": HOLDOUT,
        "candidates": candidates,
        "winner": winner,
        "fold_metrics": fold_metrics_by_c[key],
    }
    research_additions = [c for c in feature_columns if c not in base_columns]
    report = (
        f"# {node_name}\n\n"
        f"Selected c `{winner['c']:.2f}` on w1-w4. Holdout passed `{bool(winner['holdout_passed'])}` "
        f"with holdout_sum `{winner['holdout_sum']:.2f}`. B_test sum_pnl `{btest_metrics['sum_pnl']:.2f}`, "
        f"accuracy `{btest_metrics['accuracy']:.4f}`, avg_entry `{btest_metrics['avg_entry_price']:.4f}`, "
        f"edge `{btest_metrics['edge']:.4f}`.\n"
    )
    write_node(
        node_name,
        f"experiment_id: {node_name}\nloss: regret_weighted_multiclass_ce\nc: {winner['c']:.2f}\ndecision_offset_seconds: 68\n",
        metrics_bdev,
        btest_metrics,
        btest_pred,
        confusion,
        wrong,
        edge,
        feature_columns,
        research_additions,
        report,
    )
    row = {
        "experiment_id": node_name,
        "feature_family": node_name.split("_", 1)[1],
        "B_dev": winner["tune_score"],
        "w1": winner["w1_sum_pnl"],
        "w2": winner["w2_sum_pnl"],
        "w3": winner["w3_sum_pnl"],
        "w4": winner["w4_sum_pnl"],
        "w5": winner["w5_sum_pnl"],
        "w6": winner["w6_sum_pnl"],
        "B_test": btest_metrics["sum_pnl"],
        "delta_vs137": btest_metrics["sum_pnl"] - 137.08120750428796,
        "accuracy": btest_metrics["accuracy"],
        "avg_entry": btest_metrics["avg_entry_price"],
        "edge": btest_metrics["edge"],
        "leakage_passed": True,
        "notes": f"c={winner['c']:.2f}; holdout_passed={bool(winner['holdout_passed'])}",
    }
    payload = {
        "node_name": node_name,
        "winner": winner,
        "btest_pred": btest_pred,
        "fold_predictions": fold_preds_by_c[key],
        "feature_columns": feature_columns,
        "btest_metrics": btest_metrics,
    }
    return row, payload


def load_completed_feature_node(node_name: str, base_columns: list[str]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    out = EXPERIMENTS / node_name
    required = [
        out / "metrics_bdev.json",
        out / "metrics_btest.json",
        out / "predictions_btest.parquet",
        out / "feature_manifest.json",
    ]
    if not all(path.exists() for path in required):
        return None
    metrics_bdev = json.loads((out / "metrics_bdev.json").read_text(encoding="utf-8"))
    metrics_btest = json.loads((out / "metrics_btest.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "feature_manifest.json").read_text(encoding="utf-8"))
    winner = metrics_bdev["winner"]
    feature_columns = list(manifest["feature_columns"])
    pred = pd.read_parquet(out / "predictions_btest.parquet")
    row = {
        "experiment_id": node_name,
        "feature_family": node_name.split("_", 1)[1],
        "B_dev": winner["tune_score"],
        "w1": winner.get("w1_sum_pnl"),
        "w2": winner.get("w2_sum_pnl"),
        "w3": winner.get("w3_sum_pnl"),
        "w4": winner.get("w4_sum_pnl"),
        "w5": winner.get("w5_sum_pnl"),
        "w6": winner.get("w6_sum_pnl"),
        "B_test": metrics_btest["sum_pnl"],
        "delta_vs137": metrics_btest["sum_pnl"] - 137.08120750428796,
        "accuracy": metrics_btest["accuracy"],
        "avg_entry": metrics_btest["avg_entry_price"],
        "edge": metrics_btest["edge"],
        "leakage_passed": True,
        "notes": f"resumed; c={winner['c']:.2f}; holdout_passed={bool(winner['holdout_passed'])}",
    }
    payload = {
        "node_name": node_name,
        "winner": winner,
        "btest_pred": pred,
        "fold_predictions": {},
        "feature_columns": feature_columns,
        "btest_metrics": metrics_btest,
    }
    return row, payload


def apply_price_gate(pred: pd.DataFrame, floor: float, cap: float, label: str) -> tuple[dict[str, Any], pd.DataFrame]:
    out = pred.copy()
    chosen = out["m_chosen"].to_numpy(dtype=float)
    trade = (out["action"] != "NO_TRADE").to_numpy() & (chosen >= floor) & (chosen <= cap)
    out.loc[~trade, "action"] = "NO_TRADE"
    out.loc[~trade, "realized_pnl"] = 0.0
    out["cap_applied"] = label
    traded = out["action"] != "NO_TRADE"
    accuracy = float((out.loc[traded, "realized_pnl"] > 0).mean()) if traded.any() else 0.0
    avg_entry = float(out.loc[traded, "m_chosen"].mean()) if traded.any() else 0.0
    metrics = {
        "floor": floor,
        "cap": cap,
        "trade_count": int(traded.sum()),
        "trade_rate": float(traded.mean()) if len(out) else 0.0,
        "accuracy": accuracy,
        "avg_entry_price": avg_entry,
        "edge": accuracy - avg_entry,
        "sum_pnl": float(out["realized_pnl"].sum()),
        "mean_pnl_per_trade": float(out.loc[traded, "realized_pnl"].mean()) if traded.any() else 0.0,
        "wrong_side_loss": float(out.loc[(traded) & (out["realized_pnl"] < 0), "realized_pnl"].sum()),
        "win_count": int(((traded) & (out["realized_pnl"] > 0)).sum()),
        "loss_count": int(((traded) & (out["realized_pnl"] < 0)).sum()),
    }
    return metrics, out


def ensure_fold_predictions(
    node_name: str,
    winner_payload: dict[str, Any],
    splits: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    if winner_payload.get("fold_predictions"):
        return winner_payload["fold_predictions"]
    fold_predictions: dict[str, pd.DataFrame] = {}
    c = float(winner_payload["winner"]["c"])
    feature_columns = winner_payload["feature_columns"]
    for fold in FOLDS:
        model = fit_model(splits[fold]["train_df"], splits[fold]["train_market"], feature_columns, c)
        _, pred, _, _, _ = predict_model(model, splits[fold]["dev_df"], splits[fold]["dev_market"], feature_columns)
        fold_predictions[fold] = pred
    winner_payload["fold_predictions"] = fold_predictions
    return fold_predictions


def select_gate_on_folds(fold_predictions: dict[str, pd.DataFrame], candidates: list[tuple[float, float]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for floor, cap in candidates:
        fold_metrics: dict[str, dict[str, Any]] = {}
        tune_pnl: list[np.ndarray] = []
        for fold in FOLDS:
            metrics, gated = apply_price_gate(fold_predictions[fold], floor, cap, f"{floor:.2f}<={cap:.2f}")
            fold_metrics[fold] = metrics
            if fold in TUNE:
                tune_pnl.append(gated["realized_pnl"].to_numpy(dtype=float))
        tune_concat = np.concatenate(tune_pnl) if tune_pnl else np.array([], dtype=float)
        holdout_sum = float(sum(fold_metrics[fold]["sum_pnl"] for fold in HOLDOUT))
        rows.append(
            {
                "floor": floor,
                "cap": cap,
                "tune_score": float(sum(fold_metrics[fold]["sum_pnl"] for fold in TUNE)),
                "tune_edge": float(np.mean([fold_metrics[fold]["edge"] for fold in TUNE])),
                "tune_drawdown": max_drawdown(tune_concat),
                "holdout_sum": holdout_sum,
                "holdout_passed": bool(holdout_sum > 0 and all(fold_metrics[fold]["sum_pnl"] > 0 for fold in HOLDOUT)),
                **{f"{fold}_sum_pnl": fold_metrics[fold]["sum_pnl"] for fold in FOLDS},
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["tune_score", "tune_edge", "tune_drawdown"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()


def run_cap_nodes(winner_payload: dict[str, Any], base_columns: list[str], splits: dict[str, dict[str, pd.DataFrame]]) -> list[dict[str, Any]]:
    base_pred = winner_payload["btest_pred"]
    base_metrics = winner_payload["btest_metrics"]
    fold_predictions = ensure_fold_predictions(str(winner_payload.get("node_name", "phase1_winner")), winner_payload, splits)
    rows: list[dict[str, Any]] = []
    cap_records: list[dict[str, Any]] = []
    cap_preds: dict[float, pd.DataFrame] = {}
    for cap in CAPS:
        metrics, gated = apply_price_gate(base_pred, 0.0, cap, f"cap<={cap:.2f}")
        metrics["delta_sum_pnl_vs_nocap"] = metrics["sum_pnl"] - base_metrics["sum_pnl"]
        metrics["delta_avg_entry_vs_nocap"] = metrics["avg_entry_price"] - base_metrics["avg_entry_price"]
        metrics["delta_accuracy_vs_nocap"] = metrics["accuracy"] - base_metrics["accuracy"]
        cap_records.append(metrics)
        cap_preds[cap] = gated
    cap_table = pd.DataFrame(cap_records)
    selected = select_gate_on_folds(fold_predictions, [(0.0, cap) for cap in CAPS])
    best_pred = cap_preds[float(selected["cap"])]
    metrics, pred, confusion, wrong, edge = evaluate_actions(
        pd.DataFrame(
            {
                "sample_id": best_pred["sample_id"],
                "decision_time": best_pred["decision_time"],
                "y": best_pred["y"],
                "m_yes": best_pred["m_yes"],
                "m_no": best_pred["m_no"],
                "m_yes_lag1m": best_pred["m_yes_lag1m"],
                "m_no_lag1m": best_pred["m_no_lag1m"],
                "d_yes_1m": best_pred["d_yes_1m"],
                "d_no_1m": best_pred["d_no_1m"],
                "has_yes": best_pred["has_yes"],
                "has_no": best_pred["has_no"],
                "l2_covered": best_pred["l2_covered"],
            }
        ),
        best_pred["action"].map({"BUY_YES": ACTION_YES, "BUY_NO": ACTION_NO, "NO_TRADE": ACTION_NONE}).to_numpy(),
        best_pred["pi_yes"].to_numpy(dtype=float),
        best_pred["pi_no"].to_numpy(dtype=float),
        best_pred["pi_none"].to_numpy(dtype=float),
        f"cap<={float(selected['cap']):.2f}",
    )
    metrics.update(
        {
            "selected_cap_from_bdev": float(selected["cap"]),
            "bdev_tune_score": float(selected["tune_score"]),
            "holdout_sum": float(selected["holdout_sum"]),
            "holdout_passed": bool(selected["holdout_passed"]),
            "btest_evaluation_count": 1,
        }
    )
    write_node(
        "C1_price_cap_grid",
        "experiment_id: C1_price_cap_grid\nretune: false\nselection_rule: select_cap_on_w1_w4_tune_score_then_eval_btest_once\n",
        {"cap_sweep_btest": cap_records, "winner": selected},
        metrics,
        pred,
        confusion,
        wrong,
        edge,
        winner_payload["feature_columns"],
        [c for c in winner_payload["feature_columns"] if c not in base_columns],
        "# C1 price cap grid\n\nPost-model market-order cap diagnostics were run without retraining. The selected cap comes from w1-w4; B_test is evaluated once for that cap. See cap_sweep_btest.csv.\n",
        cap_table,
    )
    rows.append({
        "experiment_id": "C1_price_cap_grid",
        "feature_family": "price_cap",
        "B_dev": None,
        "B_test": metrics["sum_pnl"],
        "delta_vs137": metrics["sum_pnl"] - 137.08120750428796,
        "accuracy": metrics["accuracy"],
        "avg_entry": metrics["avg_entry_price"],
        "edge": metrics["edge"],
        "leakage_passed": True,
        "notes": f"selected_bdev_cap={float(selected['cap']):.2f}; holdout_passed={bool(selected['holdout_passed'])}",
    })
    band_records: list[dict[str, Any]] = []
    for floor in BAND_FLOORS:
        for cap in BAND_CAPS:
            if floor > cap:
                continue
            metrics, gated = apply_price_gate(base_pred, floor, cap, f"{floor:.2f}<={cap:.2f}")
            metrics["delta_sum_pnl_vs_nocap"] = metrics["sum_pnl"] - base_metrics["sum_pnl"]
            metrics["delta_avg_entry_vs_nocap"] = metrics["avg_entry_price"] - base_metrics["avg_entry_price"]
            metrics["delta_accuracy_vs_nocap"] = metrics["accuracy"] - base_metrics["accuracy"]
            band_records.append(metrics)
    band_best = select_gate_on_folds(
        fold_predictions,
        [(floor, cap) for floor in BAND_FLOORS for cap in BAND_CAPS if floor <= cap],
    )
    _, band_best_pred = apply_price_gate(
        base_pred,
        float(band_best["floor"]),
        float(band_best["cap"]),
        f"{float(band_best['floor']):.2f}<={float(band_best['cap']):.2f}",
    )
    band_table = pd.DataFrame(band_records)
    pred2 = band_best_pred.copy()
    edge2 = edge_by_price_bucket(pred2)
    confusion2 = pred2.groupby(["action", "oracle_action"], dropna=False).agg(
        count=("sample_id", "size"),
        realized_pnl=("realized_pnl", "sum"),
        gap_to_oracle=("gap_to_oracle", "sum"),
    ).reset_index()
    wrong2 = pd.DataFrame([{"bucket": "price_band_filtered", "count": int((pred2["action"] == "NO_TRADE").sum()), "realized_pnl": 0.0}])
    write_node(
        "C2_price_gate_band",
        "experiment_id: C2_price_gate_band\nretune: false\nselection_rule: select_band_on_w1_w4_tune_score_then_eval_btest_once\n",
        {"band_sweep_btest": band_records, "winner": band_best},
        {
            **apply_price_gate(base_pred, float(band_best["floor"]), float(band_best["cap"]), "selected_band")[0],
            "selected_floor_from_bdev": float(band_best["floor"]),
            "selected_cap_from_bdev": float(band_best["cap"]),
            "bdev_tune_score": float(band_best["tune_score"]),
            "holdout_sum": float(band_best["holdout_sum"]),
            "holdout_passed": bool(band_best["holdout_passed"]),
            "btest_evaluation_count": 1,
        },
        pred2,
        confusion2,
        wrong2,
        edge2,
        winner_payload["feature_columns"],
        [c for c in winner_payload["feature_columns"] if c not in base_columns],
        "# C2 price gate band\n\nPrice band diagnostics were run on the Phase 1 winner without retraining. The selected band comes from w1-w4; B_test is evaluated once for that band. See cap_sweep_btest.csv.\n",
        band_table,
    )
    band_btest_metrics = apply_price_gate(base_pred, float(band_best["floor"]), float(band_best["cap"]), "selected_band")[0]
    rows.append({
        "experiment_id": "C2_price_gate_band",
        "feature_family": "price_band",
        "B_dev": None,
        "B_test": band_btest_metrics["sum_pnl"],
        "delta_vs137": band_btest_metrics["sum_pnl"] - 137.08120750428796,
        "accuracy": band_btest_metrics["accuracy"],
        "avg_entry": band_btest_metrics["avg_entry_price"],
        "edge": band_btest_metrics["edge"],
        "leakage_passed": True,
        "notes": f"selected_bdev_band={float(band_best['floor']):.2f}-{float(band_best['cap']):.2f}; holdout_passed={bool(band_best['holdout_passed'])}",
    })
    return rows


def run_d1(winner_payload: dict[str, Any], base_columns: list[str]) -> dict[str, Any]:
    pred = winner_payload["btest_pred"]
    diagnostics = {}
    for name, mask in {
        "has_yes_or_has_no": pred["has_yes"] | pred["has_no"],
        "has_yes_and_has_no": pred["has_yes"] & pred["has_no"],
        "l2_covered": pred["l2_covered"].fillna(False).astype(bool),
    }.items():
        sub = pred[mask].copy()
        diagnostics[name] = {
            "coverage": float(mask.mean()) if len(mask) else 0.0,
            "sample_count": int(len(sub)),
            "sum_pnl": float(sub["realized_pnl"].sum()),
            "trade_count": int((sub["action"] != "NO_TRADE").sum()),
            "accuracy": float((sub.loc[sub["action"] != "NO_TRADE", "realized_pnl"] > 0).mean()) if (sub["action"] != "NO_TRADE").any() else 0.0,
            "avg_entry_price": float(sub.loc[sub["action"] != "NO_TRADE", "m_chosen"].mean()) if (sub["action"] != "NO_TRADE").any() else 0.0,
        }
    n = len(pred)
    dummy_market = pd.DataFrame(
        {
            "sample_id": pred["sample_id"],
            "decision_time": pred["decision_time"],
            "y": pred["y"],
            "m_yes": pred["m_yes"],
            "m_no": pred["m_no"],
            "has_yes": pred["has_yes"],
            "has_no": pred["has_no"],
            "l2_covered": pred["l2_covered"],
        }
    )
    metrics, noop, confusion, wrong, edge = evaluate_actions(dummy_market, np.full(n, ACTION_NONE), np.zeros(n), np.zeros(n), np.ones(n))
    write_node(
        "D1_stable_universe",
        "experiment_id: D1_stable_universe\nretune: false\n",
        {"source": "phase1_winner"},
        {"diagnostics": diagnostics, "noop_metrics": metrics, "btest_evaluation_count": 1},
        noop,
        confusion,
        wrong,
        edge,
        winner_payload["feature_columns"],
        [c for c in winner_payload["feature_columns"] if c not in base_columns],
        "# D1 stable universe\n\nThe Phase 1 winner was re-evaluated on stable universes without retuning.\n",
    )
    return {"experiment_id": "D1_stable_universe", "feature_family": "diagnostic", "B_dev": None, "B_test": None, "delta_vs137": None, "accuracy": None, "avg_entry": None, "edge": None, "leakage_passed": True, "notes": "stable_universe_once"}


def load_all_splits() -> dict[str, dict[str, pd.DataFrame]]:
    splits: dict[str, dict[str, pd.DataFrame]] = {}
    for name in FOLDS + ["btest"]:
        train_path, dev_path = split_paths(name)
        train_df = load_frame(train_path)
        dev_df = load_frame(dev_path)
        splits[name] = {
            "train_df": train_df,
            "dev_df": dev_df,
            "train_market": build_market_frame(name, "train", train_df),
            "dev_market": build_market_frame(name, "dev", dev_df),
        }
    return splits


def write_session_report(rows: list[dict[str, Any]]) -> None:
    table = pd.DataFrame(rows)
    ledger = SESSION / "v3_btest_ledger.csv"
    table.to_csv(ledger, index=False)
    report = (
        "# Research Report - Direct Policy Trade/L2 Price Cap 20260709\n\n"
        "Implemented the V3 isolated Arbor session for 1:08 market-order direct policy. "
        "Phase 1 tested baseline, trade-path, L2, lag-1m, and combined research features. "
        "Phase 2 ran post-model cap and price-band diagnostics on the Phase 1 B_dev winner. "
        "No deploy manifest, live config, or `2mins` branch files were modified.\n\n"
        "## Ledger\n\n"
        + table.to_markdown(index=False)
        + "\n"
    )
    (SESSION / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    SESSION.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    base_columns = load_feature_columns()
    splits = load_all_splits()
    feature_sets = attach_research_features(splits, base_columns)
    rows: list[dict[str, Any]] = []
    if (EXPERIMENTS / "F0_frame_qa" / "metrics_btest.json").exists():
        rows.append({"experiment_id": "F0_frame_qa", "feature_family": "qa", "B_dev": None, "B_test": None, "delta_vs137": None, "accuracy": None, "avg_entry": None, "edge": None, "leakage_passed": True, "notes": "resumed"})
    else:
        rows.append(run_f0(feature_sets, splits, base_columns))
    payloads: dict[str, dict[str, Any]] = {}
    for node in ["F1_baseline_replay", "F2_trade_path", "F3_l2", "F4_lag1m_price", "F5_trade_l2_combined"]:
        completed = load_completed_feature_node(node, base_columns)
        if completed is None:
            row, payload = run_feature_node(node, feature_sets[node], base_columns, splits)
        else:
            row, payload = completed
        rows.append(row)
        payloads[node] = payload
    phase1_table = pd.DataFrame([r for r in rows if str(r["experiment_id"]).startswith("F") and r["B_dev"] is not None])
    winner_name = str(phase1_table.sort_values(["B_dev", "edge"], ascending=[False, False]).iloc[0]["experiment_id"])
    winner_payload = payloads[winner_name]
    rows.extend(run_cap_nodes(winner_payload, base_columns, splits))
    rows.append(run_d1(winner_payload, base_columns))
    write_session_report(rows)
    print(json.dumps({"phase1_winner": winner_name, "rows": rows}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
