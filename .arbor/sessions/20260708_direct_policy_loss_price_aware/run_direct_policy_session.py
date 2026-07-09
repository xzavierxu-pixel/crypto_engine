#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
EXPERIMENTS = SESSION / "experiments"
CACHE = SESSION / "cache"
TRADE_DIR = ROOT / "price_estimator" / "data" / "sell_taker_trades_daily"
ROLLING = ROOT / ".arbor" / "sessions" / "20260703_prefinal_rolling" / "folds"
BASELINE_MANIFEST = ROOT / "execution_engine" / "deploy" / "baseline" / "artifact_manifest.json"
TRAIN = ROOT / "price_estimator" / "expected_return" / "experiments" / "20260619_expected_return_trade_coverage_start" / "data" / "expected_return_train.parquet"
BTEST = ROOT / "price_estimator" / "expected_return" / "experiments" / "20260619_expected_return_trade_coverage_start" / "data" / "expected_return_validation.parquet"

FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE = FOLDS[:4]
HOLDOUT = FOLDS[4:]
TAUS = [0.00, 0.01, 0.02]
CS = [0.00, 0.01, 0.02]
DP2_TEMPS = [0.03, 0.05, 0.08, 0.10]
DP3_REGRET_SCALES = [0.03, 0.05, 0.08]
DP4_TAUS = [0.00, 0.01, 0.02, 0.03]
DP5_CS = [0.00, 0.01, 0.02, 0.03]
DP5_TEMPS = [0.05, 0.08, 0.10, 0.15]
DP5_MAX_DEPTHS = [2, 3]
DP5_ETAS = [0.03, 0.05]
ACTION_NAMES = ["BUY_YES", "BUY_NO", "NO_TRADE"]
ACTION_YES = 0
ACTION_NO = 1
ACTION_NONE = 2
PRICE_MODES = ["actual-price", "proxy-price"]
OUTCOME_MAP = {"YES": "UP", "NO": "DOWN"}
FORBIDDEN_EXACT = {
    "y",
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
    "trade_time",
    "condition_id",
    "market_id",
    "slug",
    "outcome",
    "endDate",
}
FORBIDDEN_PREFIXES = ("future_",)
P_BASE_COLUMN = "p_up"
DEFAULT_TRAINING_PARAMS = {
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "lambda": 5.0,
    "tree_method": "hist",
    "nthread": -1,
    "seed": 20260708,
}


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.nanmax(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return exp / denom


def finite_reward_matrix(reward_matrix: np.ndarray) -> np.ndarray:
    safe = reward_matrix.copy()
    finite_mask = np.isfinite(safe)
    row_min = np.where(
        finite_mask.any(axis=1, keepdims=True),
        np.nanmin(np.where(finite_mask, safe, np.nan), axis=1, keepdims=True),
        0.0,
    )
    safe = np.where(finite_mask, safe, row_min - 1.0)
    return safe


def load_feature_columns() -> list[str]:
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    cols = list(manifest["feature_columns"])
    bad = [
        c
        for c in cols
        if c in FORBIDDEN_EXACT
        or any(c.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    ]
    if bad:
        raise RuntimeError(f"forbidden deploy features: {bad}")
    return cols


def split_paths(name: str) -> tuple[Path, Path]:
    if name == "btest":
        return TRAIN, BTEST
    fold_dir = ROLLING / name / "data"
    return fold_dir / "train.parquet", fold_dir / "dev.parquet"


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def trade_files(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    dates = set(
        pd.date_range(start.normalize(), end.normalize(), freq="D").strftime("%Y-%m-%d")
    )
    return sorted(
        p for p in TRADE_DIR.glob("date=*.parquet") if p.stem.removeprefix("date=") in dates
    )


def load_trade_lookup(start: pd.Timestamp, end: pd.Timestamp) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    files = trade_files(start, end)
    if not files:
        return {}
    trades = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["condition_id", "outcome", "price", "trade_time"],
            )
            for path in files
        ],
        ignore_index=True,
    )
    trades["trade_time"] = pd.to_datetime(trades["trade_time"], utc=True)
    trades["outcome"] = trades["outcome"].astype(str).str.upper()
    trades["condition_id"] = trades["condition_id"].astype(str)
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


def build_market_frame(name: str, kind: str, df: pd.DataFrame) -> pd.DataFrame:
    out_path = CACHE / "market_frames" / f"{name}_{kind}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    market_t0 = pd.to_datetime(df["market_t0"], utc=True)
    cutoff = market_t0 + pd.Timedelta(seconds=68)
    lookup = load_trade_lookup(market_t0.min(), cutoff.max())
    rows: list[dict[str, Any]] = []
    cutoff_ns = cutoff.view("int64").to_numpy()
    targets = df["target"].astype(int).to_numpy()
    for idx, row in enumerate(df.itertuples(index=False)):
        cid = str(row.condition_id)
        m_yes, m_yes_time, has_yes = last_price_before(
            lookup, cid, OUTCOME_MAP["YES"], int(cutoff_ns[idx])
        )
        m_no, m_no_time, has_no = last_price_before(
            lookup, cid, OUTCOME_MAP["NO"], int(cutoff_ns[idx])
        )
        rows.append(
            {
                "sample_id": f"{name}_{kind}_{idx}",
                "condition_id": cid,
                "market_t0": market_t0.iloc[idx],
                "decision_time": cutoff.iloc[idx],
                "y": int(targets[idx]),
                "m_yes": m_yes,
                "m_no": m_no,
                "m_yes_trade_time": m_yes_time,
                "m_no_trade_time": m_no_time,
                "has_yes": bool(has_yes),
                "has_no": bool(has_no),
                "has_any": bool(has_yes or has_no),
            }
        )
    out = pd.DataFrame(rows)
    late_yes = out["has_yes"] & (
        pd.to_datetime(out["m_yes_trade_time"], utc=True)
        > pd.to_datetime(out["decision_time"], utc=True)
    )
    late_no = out["has_no"] & (
        pd.to_datetime(out["m_no_trade_time"], utc=True)
        > pd.to_datetime(out["decision_time"], utc=True)
    )
    if bool(late_yes.any() or late_no.any()):
        raise RuntimeError(f"late join detected in {name}/{kind}")
    out.to_parquet(out_path, index=False)
    return out


def matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = df.reindex(columns=feature_columns).copy()
    for col in x.columns:
        if not pd.api.types.is_numeric_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).astype("float32")


def safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def build_dp5_frame(
    df: pd.DataFrame,
    market: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    out = matrix(df, feature_columns)
    m_yes = market["m_yes"].to_numpy(dtype=float)
    m_no = market["m_no"].to_numpy(dtype=float)
    p_base = df[P_BASE_COLUMN].to_numpy(dtype=float)
    out["m_yes"] = m_yes
    out["m_no"] = m_no
    out["m_yes_minus_m_no"] = m_yes - m_no
    out["abs_m_yes_minus_half"] = np.abs(m_yes - 0.5)
    out["abs_m_no_minus_half"] = np.abs(m_no - 0.5)
    out["logit_m_yes"] = safe_logit(m_yes)
    out["logit_m_no"] = safe_logit(m_no)
    out["p_base"] = p_base
    out["p_base_minus_m_yes"] = p_base - m_yes
    out["one_minus_p_base_minus_m_no"] = (1.0 - p_base) - m_no
    out["abs_p_base_minus_m_yes"] = np.abs(p_base - m_yes)
    out["abs_one_minus_p_base_minus_m_no"] = np.abs((1.0 - p_base) - m_no)
    out["abs_p_base_minus_half"] = np.abs(p_base - 0.5)
    return out.astype("float32"), list(out.columns)


def build_dp6_frame(
    df: pd.DataFrame,
    market: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    out = matrix(df, feature_columns)
    m_yes = market["m_yes"].to_numpy(dtype=float)
    m_no = market["m_no"].to_numpy(dtype=float)
    out["m_yes"] = m_yes
    out["m_no"] = m_no
    out["m_yes_minus_m_no"] = m_yes - m_no
    out["min_price_side"] = np.fmin(m_yes, m_no)
    out["max_price_side"] = np.fmax(m_yes, m_no)
    return out.astype("float32"), list(out.columns)


def compute_rewards(
    market: pd.DataFrame,
    c: float,
    price_mode: str = "actual-price",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = market["y"].to_numpy(dtype=float)
    has_yes = market["has_yes"].to_numpy(dtype=bool)
    has_no = market["has_no"].to_numpy(dtype=bool)
    m_yes = market["m_yes"].to_numpy(dtype=float)
    if price_mode == "actual-price":
        m_no = market["m_no"].to_numpy(dtype=float)
        reward_yes = np.where(has_yes, y - m_yes - c, -np.inf)
        reward_no = np.where(has_no, (1.0 - y) - m_no - c, -np.inf)
    elif price_mode == "proxy-price":
        reward_yes = np.where(has_yes, y - m_yes - c, -np.inf)
        reward_no = np.where(has_yes, m_yes - y - c, -np.inf)
    else:
        raise ValueError(f"unknown price_mode: {price_mode}")
    reward_none = np.zeros(len(market), dtype=float)
    reward_matrix = np.column_stack([reward_yes, reward_no, reward_none])
    oracle = np.argmax(reward_matrix, axis=1)
    top = reward_matrix[np.arange(len(reward_matrix)), oracle]
    second = np.partition(reward_matrix, -2, axis=1)[:, -2]
    margin = np.maximum(top - second, 0.0)
    oracle_pnl = np.max(
        np.column_stack(
            [
                np.where(has_yes, y - m_yes, -np.inf),
                np.where(has_no, (1.0 - y) - market["m_no"].to_numpy(dtype=float), -np.inf),
                np.zeros(len(market), dtype=float),
            ]
        ),
        axis=1,
    )
    oracle_pnl = np.where(np.isfinite(oracle_pnl), oracle_pnl, 0.0)
    return reward_matrix, oracle, margin, oracle_pnl


def cross_side_regret_weights(
    reward_matrix: np.ndarray,
    oracle_action: np.ndarray,
    regret_scale: float,
) -> np.ndarray:
    reward_yes = reward_matrix[:, ACTION_YES]
    reward_no = reward_matrix[:, ACTION_NO]
    best_reward = reward_matrix[np.arange(len(reward_matrix)), oracle_action]
    wrong_side_reward = np.where(
        oracle_action == ACTION_YES,
        reward_no,
        np.where(
            oracle_action == ACTION_NO,
            reward_yes,
            np.maximum(reward_yes, reward_no),
        ),
    )
    cross_side_regret = best_reward - wrong_side_reward
    cross_side_regret = np.where(np.isfinite(cross_side_regret), cross_side_regret, best_reward)
    return np.clip(np.log1p(np.maximum(cross_side_regret, 0.0) / regret_scale), 0.5, 10.0)


def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    curve = np.cumsum(pnl)
    peak = np.maximum.accumulate(curve)
    return float((peak - curve).max())


def weekly_worst_pnl(decision_time: pd.Series, pnl: np.ndarray) -> float:
    frame = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(decision_time, utc=True),
            "pnl": pnl,
        }
    )
    weekly = frame.groupby(pd.Grouper(key="decision_time", freq="W-MON"))["pnl"].sum()
    return float(weekly.min()) if len(weekly) else 0.0


def make_prediction_frame(
    market: pd.DataFrame,
    action_idx: np.ndarray,
    pi_yes: np.ndarray,
    pi_no: np.ndarray,
    pi_none: np.ndarray,
    utility_yes: np.ndarray,
    utility_no: np.ndarray,
    utility_none: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y = market["y"].to_numpy(dtype=int)
    m_yes = market["m_yes"].to_numpy(dtype=float)
    m_no = market["m_no"].to_numpy(dtype=float)
    has_yes = market["has_yes"].to_numpy(dtype=bool)
    has_no = market["has_no"].to_numpy(dtype=bool)
    reward_matrix, oracle_action, _, oracle_pnl = compute_rewards(market, 0.0, "actual-price")

    action_idx = action_idx.astype(int)
    action_idx = np.where((action_idx == ACTION_YES) & (~has_yes), ACTION_NONE, action_idx)
    action_idx = np.where((action_idx == ACTION_NO) & (~has_no), ACTION_NONE, action_idx)

    realized = np.zeros(len(market), dtype=float)
    yes_mask = action_idx == ACTION_YES
    no_mask = action_idx == ACTION_NO
    realized[yes_mask] = y[yes_mask] - m_yes[yes_mask]
    realized[no_mask] = (1 - y[no_mask]) - m_no[no_mask]

    trade_mask = action_idx != ACTION_NONE
    wrong_buy_yes_mask = yes_mask & (y == 0)
    wrong_buy_no_mask = no_mask & (y == 1)
    wrong_side_mask = wrong_buy_yes_mask | wrong_buy_no_mask
    gap_to_oracle = oracle_pnl - realized
    oracle_gap_for_none = np.where(action_idx == ACTION_NONE, oracle_pnl, 0.0)

    metrics = {
        "sample_count": int(len(market)),
        "trade_count": int(trade_mask.sum()),
        "trade_rate": float(trade_mask.mean()) if len(market) else 0.0,
        "trade_accuracy": float((realized[trade_mask] > 0).mean()) if trade_mask.any() else 0.0,
        "sum_pnl": float(realized.sum()),
        "mean_pnl_per_trade": float(realized[trade_mask].mean()) if trade_mask.any() else 0.0,
        "YES_pnl": float(realized[yes_mask].sum()),
        "NO_pnl": float(realized[no_mask].sum()),
        "win_pnl_sum": float(realized[trade_mask & (realized > 0)].sum()),
        "loss_pnl_sum": float(realized[trade_mask & (realized < 0)].sum()),
        "wrong_side_loss": float(realized[wrong_side_mask].sum()),
        "wrong_side_loss_abs": float(np.abs(realized[wrong_side_mask].sum())),
        "wrong_side_count": int(wrong_side_mask.sum()),
        "wrong_buy_yes_count": int(wrong_buy_yes_mask.sum()),
        "wrong_buy_no_count": int(wrong_buy_no_mask.sum()),
        "wrong_buy_yes_loss": float(realized[wrong_buy_yes_mask].sum()),
        "wrong_buy_no_loss": float(realized[wrong_buy_no_mask].sum()),
        "no_trade_count": int((action_idx == ACTION_NONE).sum()),
        "no_trade_missed_oracle_pnl": float(oracle_gap_for_none.sum()),
        "oracle_pnl": float(oracle_pnl.sum()),
        "capture_ratio": float(realized.sum() / oracle_pnl.sum()) if float(oracle_pnl.sum()) else 0.0,
        "cross_side_gap_to_oracle": float(gap_to_oracle.sum()),
        "worst_week_pnl": weekly_worst_pnl(market["decision_time"], realized),
        "selection_drawdown": max_drawdown(realized),
        "m_yes_coverage": float(has_yes.mean()),
        "m_no_coverage": float(has_no.mean()),
        "m_any_coverage": float((has_yes | has_no).mean()),
        "late_join_count": 0,
        "holdout_passed": None,
    }

    pred = pd.DataFrame(
        {
            "sample_id": market["sample_id"],
            "decision_time": market["decision_time"],
            "y": y,
            "m_yes": m_yes,
            "m_no": m_no,
            "pi_yes": pi_yes,
            "pi_no": pi_no,
            "pi_none": pi_none,
            "utility_yes": utility_yes,
            "utility_no": utility_no,
            "utility_none": utility_none,
            "action": [ACTION_NAMES[i] for i in action_idx],
            "realized_pnl": realized,
            "oracle_action": [ACTION_NAMES[i] for i in oracle_action],
            "oracle_pnl": oracle_pnl,
            "gap_to_oracle": gap_to_oracle,
            "m_yes_trade_time": market["m_yes_trade_time"],
            "m_no_trade_time": market["m_no_trade_time"],
            "has_yes": has_yes,
            "has_no": has_no,
        }
    )

    confusion = (
        pred.assign(
            oracle_gap=pred["gap_to_oracle"],
        )
        .groupby(["action", "oracle_action"], dropna=False)
        .agg(
            count=("sample_id", "size"),
            realized_pnl=("realized_pnl", "sum"),
            oracle_gap=("oracle_gap", "sum"),
        )
        .reset_index()
    )

    wrong_side = pd.DataFrame(
        [
            {
                "bucket": "wrong_buy_yes",
                "count": int(wrong_buy_yes_mask.sum()),
                "realized_pnl": float(realized[wrong_buy_yes_mask].sum()),
                "oracle_gap": float(gap_to_oracle[wrong_buy_yes_mask].sum()),
            },
            {
                "bucket": "wrong_buy_no",
                "count": int(wrong_buy_no_mask.sum()),
                "realized_pnl": float(realized[wrong_buy_no_mask].sum()),
                "oracle_gap": float(gap_to_oracle[wrong_buy_no_mask].sum()),
            },
            {
                "bucket": "no_trade_missed_oracle",
                "count": int((action_idx == ACTION_NONE).sum()),
                "realized_pnl": 0.0,
                "oracle_gap": float(oracle_gap_for_none.sum()),
            },
        ]
    )
    return pred, confusion, wrong_side, metrics


def evaluate_actions(
    market: pd.DataFrame,
    action_idx: np.ndarray,
    pi_yes: np.ndarray,
    pi_no: np.ndarray,
    pi_none: np.ndarray,
    utility_yes: np.ndarray,
    utility_no: np.ndarray,
    utility_none: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred, confusion, wrong_side, metrics = make_prediction_frame(
        market,
        action_idx,
        pi_yes,
        pi_no,
        pi_none,
        utility_yes,
        utility_no,
        utility_none,
    )
    return metrics, pred, confusion, wrong_side


def no_trade_and_oracle_metrics(market: pd.DataFrame) -> dict[str, Any]:
    n = len(market)
    zero = np.zeros(n, dtype=float)
    none = np.full(n, 1.0, dtype=float)
    no_trade_metrics, _, _, _ = evaluate_actions(
        market,
        np.full(n, ACTION_NONE, dtype=int),
        zero,
        zero,
        none,
        zero,
        zero,
        zero,
    )
    reward_matrix, oracle_action, _, _ = compute_rewards(market, 0.0, "actual-price")
    pi = softmax(np.where(np.isfinite(reward_matrix), reward_matrix, -1e9))
    oracle_metrics, _, _, _ = evaluate_actions(
        market,
        oracle_action,
        pi[:, 0],
        pi[:, 1],
        pi[:, 2],
        reward_matrix[:, 0],
        reward_matrix[:, 1],
        reward_matrix[:, 2],
    )
    return {"no_trade": no_trade_metrics, "oracle": oracle_metrics}


def write_node(
    node_name: str,
    config_text: str,
    metrics_bdev: dict[str, Any],
    metrics_btest: dict[str, Any],
    predictions_btest: pd.DataFrame,
    confusion_btest: pd.DataFrame,
    wrong_side_btest: pd.DataFrame,
    feature_columns: list[str],
    report_text: str,
    extra_manifest: dict[str, Any] | None = None,
) -> None:
    out = EXPERIMENTS / node_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(config_text, encoding="utf-8")
    dump_json(out / "metrics_bdev.json", metrics_bdev)
    dump_json(out / "metrics_btest.json", metrics_btest)
    predictions_btest.to_parquet(out / "predictions_btest.parquet", index=False)
    confusion_btest.to_csv(out / "confusion_btest.csv", index=False)
    wrong_side_btest.to_csv(out / "wrong_side_decomposition_btest.csv", index=False)
    bad = [
        c
        for c in feature_columns
        if c in FORBIDDEN_EXACT or any(c.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    ]
    manifest = {
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    dump_json(out / "feature_manifest.json", manifest)
    dump_json(
        out / "leakage_check.json",
        {
            "forbidden_feature_intersection": bad,
            "passed": not bad,
        },
    )
    if bad:
        raise RuntimeError(f"leakage check failed for {node_name}: {bad}")
    (out / "REPORT.md").write_text(report_text, encoding="utf-8")


def select_candidate(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    table = pd.DataFrame(candidate_rows)
    order = table.sort_values(
        ["tune_score", "tune_wrong_side_loss_abs", "tune_drawdown", "tune_trade_rate"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return order.iloc[0].to_dict()


def split_train_valid(
    x: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order = pd.to_datetime(market["decision_time"], utc=True).sort_values().index.to_numpy()
    split = max(int(len(order) * 0.85), 500)
    split = min(split, len(order) - 100) if len(order) > 600 else max(len(order) - 1, 1)
    fit_idx = order[:split]
    val_idx = order[split:]
    if len(val_idx) == 0:
        fit_idx = order[:-1]
        val_idx = order[-1:]
    return (
        x.iloc[fit_idx],
        x.iloc[val_idx],
        market.iloc[fit_idx].reset_index(drop=True),
        market.iloc[val_idx].reset_index(drop=True),
    )


def make_sum_pnl_metric(
    market: pd.DataFrame,
    train_market: pd.DataFrame | None = None,
    temperature: float = 1.0,
) -> Callable[[np.ndarray, xgb.DMatrix], tuple[str, float]]:
    def unpack(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            frame["has_yes"].to_numpy(dtype=bool),
            frame["has_no"].to_numpy(dtype=bool),
            frame["y"].to_numpy(dtype=float),
            frame["m_yes"].to_numpy(dtype=float),
            frame["m_no"].to_numpy(dtype=float),
        )

    valid_payload = unpack(market)
    train_payload = unpack(train_market) if train_market is not None else None

    def metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
        if predt.ndim == 1:
            predt_2d = predt.reshape(-1, 3)
        else:
            predt_2d = predt
        if train_payload is not None and predt_2d.shape[0] == len(train_market):
            has_yes, has_no, y, m_yes, m_no = train_payload
        else:
            has_yes, has_no, y, m_yes, m_no = valid_payload
        scores = predt_2d / temperature
        action = scores.copy()
        action[~has_yes, ACTION_YES] = -np.inf
        action[~has_no, ACTION_NO] = -np.inf
        choice = np.argmax(action, axis=1)
        realized = np.zeros(len(choice), dtype=float)
        yes_mask = choice == ACTION_YES
        no_mask = choice == ACTION_NO
        realized[yes_mask] = y[yes_mask] - m_yes[yes_mask]
        realized[no_mask] = (1.0 - y[no_mask]) - m_no[no_mask]
        return "sum_pnl", float(realized.sum())

    return metric


def fit_multiclass_model(
    train_df: pd.DataFrame,
    train_market: pd.DataFrame,
    feature_columns: list[str],
    c: float,
) -> xgb.XGBClassifier:
    x = matrix(train_df, feature_columns)
    reward_matrix, oracle_action, margin, _ = compute_rewards(train_market, c, "actual-price")
    weight = np.clip(np.log1p(margin / 0.02), 0.25, 5.0)
    order = pd.to_datetime(train_market["decision_time"], utc=True).sort_values().index.to_numpy()
    split = max(int(len(order) * 0.85), 500)
    split = min(split, len(order) - 100) if len(order) > 600 else max(len(order) - 1, 1)
    fit_idx = order[:split]
    val_idx = order[split:]
    if len(val_idx) == 0:
        fit_idx = order[:-1]
        val_idx = order[-1:]
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_lambda=5.0,
        n_estimators=500,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=20260708,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(
        x.iloc[fit_idx],
        oracle_action[fit_idx],
        sample_weight=weight[fit_idx],
        eval_set=[(x.iloc[val_idx], oracle_action[val_idx])],
        sample_weight_eval_set=[weight[val_idx]],
        verbose=False,
    )
    return model


def predict_multiclass(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    market: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probs = model.predict_proba(matrix(df, feature_columns))
    has_yes = market["has_yes"].to_numpy(dtype=bool)
    has_no = market["has_no"].to_numpy(dtype=bool)
    masked = probs.copy()
    masked[~has_yes, ACTION_YES] = -np.inf
    masked[~has_no, ACTION_NO] = -np.inf
    action = masked.argmax(axis=1)
    return evaluate_actions(
        market,
        action,
        probs[:, 0],
        probs[:, 1],
        probs[:, 2],
        probs[:, 0],
        probs[:, 1],
        probs[:, 2],
    )


def fit_soft_reward_model(
    train_df: pd.DataFrame,
    train_market: pd.DataFrame,
    feature_columns: list[str],
    c: float,
    temperature: float,
) -> tuple[xgb.Booster, dict[str, Any]]:
    x = matrix(train_df, feature_columns)
    reward_matrix, _, _, _ = compute_rewards(train_market, c, "actual-price")
    target_scores = finite_reward_matrix(reward_matrix) / temperature
    soft_targets = softmax(target_scores)
    x_fit, x_val, market_fit, market_val = split_train_valid(x, train_market)
    fit_rewards, _, _, _ = compute_rewards(market_fit, c, "actual-price")
    val_rewards, _, _, _ = compute_rewards(market_val, c, "actual-price")
    fit_targets = softmax(finite_reward_matrix(fit_rewards) / temperature)
    val_targets = softmax(finite_reward_matrix(val_rewards) / temperature)

    dtrain = xgb.DMatrix(x_fit, label=np.argmax(fit_targets, axis=1))
    dvalid = xgb.DMatrix(x_val, label=np.argmax(val_targets, axis=1))

    def obj(predt: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        probs = softmax(predt)
        grad = probs - fit_targets
        hess = np.maximum(probs * (1.0 - probs), 1e-4)
        return grad, hess

    def metric(predt: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[str, float]:
        rows = dmatrix.num_row()
        if rows == len(market_fit):
            probs = softmax(predt)
            ce = -np.mean(np.sum(fit_targets * np.log(np.clip(probs, 1e-9, 1.0)), axis=1))
            return "soft_ce", float(-ce)
        probs = softmax(predt)
        ce = -np.mean(np.sum(val_targets * np.log(np.clip(probs, 1e-9, 1.0)), axis=1))
        return "soft_ce", float(-ce)

    evals_result: dict[str, Any] = {}
    booster = xgb.train(
        {
            **DEFAULT_TRAINING_PARAMS,
            "num_class": 3,
            "max_depth": 3,
            "eta": 0.05,
            "objective": "multi:softprob",
            "disable_default_eval_metric": 1,
        },
        dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        obj=obj,
        custom_metric=make_sum_pnl_metric(market_val, train_market=market_fit),
        maximize=True,
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False,
    )
    return booster, {"soft_targets_shape": list(soft_targets.shape), "evals_result": evals_result}


def fit_soft_reward_model_matrix(
    x: pd.DataFrame,
    train_market: pd.DataFrame,
    c: float,
    temperature: float,
) -> tuple[xgb.Booster, dict[str, Any]]:
    reward_matrix, _, _, _ = compute_rewards(train_market, c, "actual-price")
    target_scores = finite_reward_matrix(reward_matrix) / temperature
    soft_targets = softmax(target_scores)
    x_fit, x_val, market_fit, market_val = split_train_valid(x, train_market)
    fit_rewards, _, _, _ = compute_rewards(market_fit, c, "actual-price")
    val_rewards, _, _, _ = compute_rewards(market_val, c, "actual-price")
    fit_targets = softmax(finite_reward_matrix(fit_rewards) / temperature)
    val_targets = softmax(finite_reward_matrix(val_rewards) / temperature)

    dtrain = xgb.DMatrix(x_fit, label=np.argmax(fit_targets, axis=1))
    dvalid = xgb.DMatrix(x_val, label=np.argmax(val_targets, axis=1))

    def obj(predt: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        probs = softmax(predt)
        grad = probs - fit_targets
        hess = np.maximum(probs * (1.0 - probs), 1e-4)
        return grad, hess

    evals_result: dict[str, Any] = {}
    booster = xgb.train(
        {
            **DEFAULT_TRAINING_PARAMS,
            "num_class": 3,
            "max_depth": 3,
            "eta": 0.05,
            "objective": "multi:softprob",
            "disable_default_eval_metric": 1,
        },
        dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        obj=obj,
        custom_metric=make_sum_pnl_metric(market_val, train_market=market_fit),
        maximize=True,
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False,
    )
    return booster, {"soft_targets_shape": list(soft_targets.shape), "evals_result": evals_result}


def predict_soft_booster(
    booster: xgb.Booster,
    df: pd.DataFrame,
    market: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dmatrix = xgb.DMatrix(matrix(df, feature_columns))
    raw = booster.predict(dmatrix, output_margin=True)
    probs = softmax(raw)
    masked = raw.copy()
    masked[~market["has_yes"].to_numpy(dtype=bool), ACTION_YES] = -np.inf
    masked[~market["has_no"].to_numpy(dtype=bool), ACTION_NO] = -np.inf
    action = np.argmax(masked, axis=1)
    metrics, pred, confusion, wrong = evaluate_actions(
        market,
        action,
        probs[:, 0],
        probs[:, 1],
        probs[:, 2],
        probs[:, 0],
        probs[:, 1],
        probs[:, 2],
    )
    return metrics, pred, confusion, wrong, {}


def predict_soft_booster_matrix(
    booster: xgb.Booster,
    x: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dmatrix = xgb.DMatrix(x)
    raw = booster.predict(dmatrix, output_margin=True)
    probs = softmax(raw)
    masked = raw.copy()
    masked[~market["has_yes"].to_numpy(dtype=bool), ACTION_YES] = -np.inf
    masked[~market["has_no"].to_numpy(dtype=bool), ACTION_NO] = -np.inf
    action = np.argmax(masked, axis=1)
    return evaluate_actions(
        market,
        action,
        probs[:, 0],
        probs[:, 1],
        probs[:, 2],
        probs[:, 0],
        probs[:, 1],
        probs[:, 2],
    )


def fit_regret_weighted_model(
    train_df: pd.DataFrame,
    train_market: pd.DataFrame,
    feature_columns: list[str],
    c: float,
    regret_scale: float,
) -> xgb.XGBClassifier:
    x = matrix(train_df, feature_columns)
    reward_matrix, oracle_action, _, _ = compute_rewards(train_market, c, "actual-price")
    weight = cross_side_regret_weights(reward_matrix, oracle_action, regret_scale)
    order = pd.to_datetime(train_market["decision_time"], utc=True).sort_values().index.to_numpy()
    split = max(int(len(order) * 0.85), 500)
    split = min(split, len(order) - 100) if len(order) > 600 else max(len(order) - 1, 1)
    fit_idx = order[:split]
    val_idx = order[split:]
    if len(val_idx) == 0:
        fit_idx = order[:-1]
        val_idx = order[-1:]
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_lambda=5.0,
        n_estimators=500,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=20260708,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(
        x.iloc[fit_idx],
        oracle_action[fit_idx],
        sample_weight=weight[fit_idx],
        eval_set=[(x.iloc[val_idx], oracle_action[val_idx])],
        sample_weight_eval_set=[weight[val_idx]],
        verbose=False,
    )
    return model


def fit_direct_pnl_model(
    train_df: pd.DataFrame,
    train_market: pd.DataFrame,
    feature_columns: list[str],
    c: float,
    temperature: float,
    max_depth: int,
    eta: float,
    price_mode: str,
) -> tuple[xgb.Booster, dict[str, Any], list[str]]:
    x, all_columns = build_dp5_frame(train_df, train_market, feature_columns)
    x_fit, x_val, market_fit, market_val = split_train_valid(x, train_market)
    fit_rewards, _, _, _ = compute_rewards(market_fit, c, price_mode)
    val_rewards, _, _, _ = compute_rewards(market_val, c, price_mode)
    fit_rewards_safe = finite_reward_matrix(fit_rewards)
    val_rewards_safe = finite_reward_matrix(val_rewards)

    dtrain = xgb.DMatrix(x_fit, label=np.zeros(len(x_fit)))
    dvalid = xgb.DMatrix(x_val, label=np.zeros(len(x_val)))

    def obj(predt: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        probs = softmax(predt / temperature)
        expected_reward = np.sum(probs * fit_rewards_safe, axis=1, keepdims=True)
        grad = probs * (expected_reward - fit_rewards_safe) / temperature
        reward_scale = np.maximum(np.max(np.abs(fit_rewards_safe[:, :2]), axis=1, keepdims=True), 0.05)
        hess = np.maximum(
            probs * (1.0 - probs) * reward_scale / (temperature**2),
            1e-4,
        )
        return grad, hess

    evals_result: dict[str, Any] = {}
    booster = xgb.train(
        {
            **DEFAULT_TRAINING_PARAMS,
            "num_class": 3,
            "max_depth": max_depth,
            "eta": eta,
            "disable_default_eval_metric": 1,
        },
        dtrain,
        num_boost_round=800,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        obj=obj,
        custom_metric=make_sum_pnl_metric(market_val, train_market=market_fit, temperature=1.0),
        maximize=True,
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False,
    )
    metadata = {
        "price_mode": price_mode,
        "temperature": temperature,
        "objective_train_curve": evals_result.get("train", {}),
        "validation_sum_pnl_by_round": evals_result.get("valid", {}).get("sum_pnl", []),
        "validation_expected_reward_proxy": float(val_rewards_safe.max(axis=1).sum()),
        "custom_hessian_mode": "surrogate_diagonal",
        "p_base_leakage_check": {
            "column": P_BASE_COLUMN,
            "present_in_source_frame": P_BASE_COLUMN in train_df.columns,
            "note": "Inherited from precomputed rolling/base frame; not regenerated in this session.",
        },
    }
    return booster, metadata, all_columns


def predict_direct_pnl_booster(
    booster: xgb.Booster,
    df: pd.DataFrame,
    market: pd.DataFrame,
    feature_columns: list[str],
    temperature: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    x, all_columns = build_dp5_frame(df, market, feature_columns)
    dmatrix = xgb.DMatrix(x)
    raw = booster.predict(dmatrix, output_margin=True)
    probs = softmax(raw / temperature)
    masked = raw.copy()
    masked[~market["has_yes"].to_numpy(dtype=bool), ACTION_YES] = -np.inf
    masked[~market["has_no"].to_numpy(dtype=bool), ACTION_NO] = -np.inf
    action = np.argmax(masked, axis=1)
    metrics, pred, confusion, wrong = evaluate_actions(
        market,
        action,
        probs[:, 0],
        probs[:, 1],
        probs[:, 2],
        raw[:, 0],
        raw[:, 1],
        raw[:, 2],
    )
    return metrics, pred, confusion, wrong, {}, all_columns


def run_dp0(feature_columns: list[str], splits: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
    fold_qa = {}
    for name in FOLDS:
        market = splits[name]["dev_market"]
        fold_qa[name] = {
            "row_count": int(len(market)),
            "m_yes_coverage": float(market["has_yes"].mean()),
            "m_no_coverage": float(market["has_no"].mean()),
            "m_any_coverage": float(market["has_any"].mean()),
            "late_join_count": 0,
            "forbidden_feature_intersection": [],
            "feature_count": len(feature_columns),
        }
    btest_market = splits["btest"]["dev_market"]
    btest_q = {
        "row_count": int(len(btest_market)),
        "m_yes_coverage": float(btest_market["has_yes"].mean()),
        "m_no_coverage": float(btest_market["has_no"].mean()),
        "m_any_coverage": float(btest_market["has_any"].mean()),
        "late_join_count": 0,
        "forbidden_feature_intersection": [],
        "feature_count": len(feature_columns),
    }
    n = len(btest_market)
    metrics_btest, pred, confusion, wrong = evaluate_actions(
        btest_market,
        np.full(n, ACTION_NONE, dtype=int),
        np.zeros(n, dtype=float),
        np.zeros(n, dtype=float),
        np.ones(n, dtype=float),
        np.zeros(n, dtype=float),
        np.zeros(n, dtype=float),
        np.zeros(n, dtype=float),
    )
    metrics_btest = {
        "qa": btest_q,
        "btest_evaluation_count": 1,
        "sample_metrics": metrics_btest,
    }
    report = (
        "# DP0 frame QA v2\n\n"
        f"B_test rows `{btest_q['row_count']}`, m_yes coverage `{btest_q['m_yes_coverage']:.4f}`, "
        f"m_no coverage `{btest_q['m_no_coverage']:.4f}`, any-side coverage `{btest_q['m_any_coverage']:.4f}`, "
        "late joins `0`, deploy feature count fixed.\n"
    )
    write_node(
        "DP0_frame_qa_v2",
        "experiment_id: DP0_frame_qa_v2\ndecision_offset_seconds: 68\nfeature_manifest_source: execution_engine/deploy/baseline/artifact_manifest.json\n",
        {"folds": fold_qa},
        metrics_btest,
        pred,
        confusion,
        wrong,
        feature_columns,
        report,
    )
    return {
        "experiment_id": "DP0_frame_qa_v2",
        "btest_status": "qa_once",
        "btest_sum_pnl": None,
        "holdout_passed": None,
    }


def run_dp1(feature_columns: list[str], splits: dict[str, dict[str, pd.DataFrame]]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = []
    per_fold_metrics: dict[str, dict[str, Any]] = {}
    for tau in TAUS:
        key = f"tau={tau:.2f}"
        per_fold_metrics[key] = {}
        tune_pnls: list[np.ndarray] = []
        trade_counts = 0
        sample_counts = 0
        tune_wrong_side_loss_abs = 0.0
        for fold in FOLDS:
            df = splits[fold]["dev_df"]
            market = splits[fold]["dev_market"]
            score = df["p_up"].to_numpy(dtype=float)
            edge_yes = np.where(market["has_yes"].to_numpy(dtype=bool), score - market["m_yes"].to_numpy(dtype=float) - tau, -np.inf)
            edge_no = np.where(market["has_no"].to_numpy(dtype=bool), (1.0 - score) - market["m_no"].to_numpy(dtype=float) - tau, -np.inf)
            best = np.column_stack([edge_yes, edge_no, np.zeros(len(df), dtype=float)])
            action = best.argmax(axis=1)
            metrics, pred, _, _ = evaluate_actions(
                market,
                action,
                score,
                1.0 - score,
                np.zeros(len(df), dtype=float),
                edge_yes,
                edge_no,
                np.zeros(len(df), dtype=float),
            )
            per_fold_metrics[key][fold] = metrics
            if fold in TUNE:
                tune_pnls.append(pred["realized_pnl"].to_numpy(dtype=float))
                trade_counts += metrics["trade_count"]
                sample_counts += metrics["sample_count"]
                tune_wrong_side_loss_abs += metrics["wrong_side_loss_abs"]
        tune_concat = np.concatenate(tune_pnls) if tune_pnls else np.array([], dtype=float)
        holdout_sum = float(sum(per_fold_metrics[key][fold]["sum_pnl"] for fold in HOLDOUT))
        holdout_both_positive = bool(all(per_fold_metrics[key][fold]["sum_pnl"] > 0 for fold in HOLDOUT))
        candidates.append(
            {
                "tau": tau,
                "tune_score": float(sum(per_fold_metrics[key][fold]["sum_pnl"] for fold in TUNE)),
                "tune_wrong_side_loss_abs": tune_wrong_side_loss_abs,
                "tune_drawdown": max_drawdown(tune_concat),
                "tune_trade_rate": float(trade_counts / sample_counts) if sample_counts else 0.0,
                "holdout_sum": holdout_sum,
                "holdout_passed": bool(holdout_sum > 0 and holdout_both_positive),
                **{f"{fold}_sum_pnl": per_fold_metrics[key][fold]["sum_pnl"] for fold in FOLDS},
            }
        )
    winner = select_candidate(candidates)
    btest_df = splits["btest"]["dev_df"]
    btest_market = splits["btest"]["dev_market"]
    score = btest_df["p_up"].to_numpy(dtype=float)
    edge_yes = np.where(btest_market["has_yes"].to_numpy(dtype=bool), score - btest_market["m_yes"].to_numpy(dtype=float) - float(winner["tau"]), -np.inf)
    edge_no = np.where(btest_market["has_no"].to_numpy(dtype=bool), (1.0 - score) - btest_market["m_no"].to_numpy(dtype=float) - float(winner["tau"]), -np.inf)
    action = np.column_stack([edge_yes, edge_no, np.zeros(len(btest_df), dtype=float)]).argmax(axis=1)
    btest_metrics, btest_pred, confusion, wrong = evaluate_actions(
        btest_market,
        action,
        score,
        1.0 - score,
        np.zeros(len(btest_df), dtype=float),
        edge_yes,
        edge_no,
        np.zeros(len(btest_df), dtype=float),
    )
    btest_metrics["holdout_passed"] = bool(winner["holdout_passed"])
    btest_metrics["btest_evaluation_count"] = 1
    btest_metrics["baseline_family"] = "static_market_ev"
    btest_metrics["no_trade_and_oracle"] = no_trade_and_oracle_metrics(btest_market)
    metrics_bdev = {
        "selection_folds": TUNE,
        "holdout_folds": HOLDOUT,
        "candidates": candidates,
        "winner": winner,
        "fold_metrics": per_fold_metrics[f"tau={winner['tau']:.2f}"],
    }
    report = (
        "# DP1 replay V1 reference\n\n"
        f"Selected tau `{winner['tau']:.2f}` on w1-w4. Holdout passed `{bool(winner['holdout_passed'])}` "
        f"with holdout_sum `{winner['holdout_sum']:.2f}`. "
        f"B_test sum_pnl `{btest_metrics['sum_pnl']:.2f}`, wrong_side_loss_abs `{btest_metrics['wrong_side_loss_abs']:.2f}`.\n"
    )
    write_node(
        "DP1_replay_v1_reference",
        f"experiment_id: DP1_replay_v1_reference\nscore_column: p_up\ntau: {winner['tau']:.2f}\ndecision_rule: argmax(score-m_yes-tau, 1-score-m_no-tau, 0)\n",
        metrics_bdev,
        btest_metrics,
        btest_pred,
        confusion,
        wrong,
        feature_columns,
        report,
    )
    row = {
        "experiment_id": "DP1_replay_v1_reference",
        "btest_status": "evaluated_once",
        "btest_sum_pnl": btest_metrics["sum_pnl"],
        "holdout_passed": bool(winner["holdout_passed"]),
    }
    return row, {
        "node_name": "DP1_replay_v1_reference",
        "btest_pred": btest_pred,
        "winner": winner,
    }


def run_dp2(feature_columns: list[str], splits: dict[str, dict[str, pd.DataFrame]]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = []
    fold_metrics_by_key: dict[str, dict[str, Any]] = {}
    fold_predictions_by_key: dict[str, dict[str, pd.DataFrame]] = {}
    training_meta_by_key: dict[str, dict[str, Any]] = {}
    for c in CS:
        for temperature in DP2_TEMPS:
            key = f"c={c:.2f}|temperature={temperature:.2f}"
            fold_metrics_by_key[key] = {}
            fold_predictions_by_key[key] = {}
            training_meta_by_key[key] = {}
            tune_pnls: list[np.ndarray] = []
            trade_counts = 0
            sample_counts = 0
            tune_wrong_side_loss_abs = 0.0
            for fold in FOLDS:
                booster, fit_meta = fit_soft_reward_model(
                    splits[fold]["train_df"],
                    splits[fold]["train_market"],
                    feature_columns,
                    c,
                    temperature,
                )
                metrics, pred, _, _, pred_meta = predict_soft_booster(
                    booster,
                    splits[fold]["dev_df"],
                    splits[fold]["dev_market"],
                    feature_columns,
                )
                fold_metrics_by_key[key][fold] = metrics
                fold_predictions_by_key[key][fold] = pred
                training_meta_by_key[key][fold] = fit_meta | pred_meta
                if fold in TUNE:
                    tune_pnls.append(pred["realized_pnl"].to_numpy(dtype=float))
                    trade_counts += metrics["trade_count"]
                    sample_counts += metrics["sample_count"]
                    tune_wrong_side_loss_abs += metrics["wrong_side_loss_abs"]
            tune_concat = np.concatenate(tune_pnls) if tune_pnls else np.array([], dtype=float)
            holdout_sum = float(sum(fold_metrics_by_key[key][fold]["sum_pnl"] for fold in HOLDOUT))
            holdout_both_positive = bool(all(fold_metrics_by_key[key][fold]["sum_pnl"] > 0 for fold in HOLDOUT))
            candidates.append(
                {
                    "c": c,
                    "temperature": temperature,
                    "tune_score": float(sum(fold_metrics_by_key[key][fold]["sum_pnl"] for fold in TUNE)),
                    "tune_wrong_side_loss_abs": tune_wrong_side_loss_abs,
                    "tune_drawdown": max_drawdown(tune_concat),
                    "tune_trade_rate": float(trade_counts / sample_counts) if sample_counts else 0.0,
                    "holdout_sum": holdout_sum,
                    "holdout_passed": bool(holdout_sum > 0 and holdout_both_positive),
                }
            )
    winner = select_candidate(candidates)
    winner_key = f"c={winner['c']:.2f}|temperature={winner['temperature']:.2f}"
    booster, fit_meta = fit_soft_reward_model(
        splits["btest"]["train_df"],
        splits["btest"]["train_market"],
        feature_columns,
        float(winner["c"]),
        float(winner["temperature"]),
    )
    btest_metrics, btest_pred, confusion, wrong, pred_meta = predict_soft_booster(
        booster,
        splits["btest"]["dev_df"],
        splits["btest"]["dev_market"],
        feature_columns,
    )
    btest_metrics["holdout_passed"] = bool(winner["holdout_passed"])
    btest_metrics["btest_evaluation_count"] = 1
    btest_metrics["baseline_family"] = "soft_reward_ce"
    btest_metrics["no_trade_and_oracle"] = no_trade_and_oracle_metrics(splits["btest"]["dev_market"])
    metrics_bdev = {
        "selection_folds": TUNE,
        "holdout_folds": HOLDOUT,
        "candidates": candidates,
        "winner": winner,
        "fold_metrics": fold_metrics_by_key[winner_key],
        "training_meta": training_meta_by_key[winner_key],
        "btest_training_meta": fit_meta | pred_meta,
    }
    report = (
        "# DP2 soft reward CE\n\n"
        f"Selected c `{winner['c']:.2f}` and temperature `{winner['temperature']:.2f}`. "
        f"Holdout passed `{bool(winner['holdout_passed'])}` with holdout_sum `{winner['holdout_sum']:.2f}`. "
        f"B_test sum_pnl `{btest_metrics['sum_pnl']:.2f}`, wrong_side_loss_abs `{btest_metrics['wrong_side_loss_abs']:.2f}`.\n"
    )
    write_node(
        "DP2_soft_reward_ce",
        (
            "experiment_id: DP2_soft_reward_ce\n"
            f"c: {winner['c']:.2f}\n"
            f"temperature: {winner['temperature']:.2f}\n"
            "objective: custom_soft_target_ce\n"
            "num_class: 3\n"
            "max_depth: 3\n"
            "eta: 0.05\n"
            "subsample: 0.9\n"
            "colsample_bytree: 0.8\n"
            "min_child_weight: 50\n"
            "lambda: 5\n"
            "num_boost_round: 400\n"
            "early_stopping_rounds: 50\n"
        ),
        metrics_bdev,
        btest_metrics,
        btest_pred,
        confusion,
        wrong,
        feature_columns,
        report,
    )
    row = {
        "experiment_id": "DP2_soft_reward_ce",
        "btest_status": "evaluated_once",
        "btest_sum_pnl": btest_metrics["sum_pnl"],
        "holdout_passed": bool(winner["holdout_passed"]),
    }
    return row, {
        "node_name": "DP2_soft_reward_ce",
        "btest_pred": btest_pred,
        "winner": winner,
    }


def run_dp3(feature_columns: list[str], splits: dict[str, dict[str, pd.DataFrame]]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = []
    fold_metrics_by_key: dict[str, dict[str, Any]] = {}
    for c in CS:
        for regret_scale in DP3_REGRET_SCALES:
            key = f"c={c:.2f}|regret_scale={regret_scale:.2f}"
            fold_metrics_by_key[key] = {}
            tune_pnls: list[np.ndarray] = []
            trade_counts = 0
            sample_counts = 0
            tune_wrong_side_loss_abs = 0.0
            for fold in FOLDS:
                model = fit_regret_weighted_model(
                    splits[fold]["train_df"],
                    splits[fold]["train_market"],
                    feature_columns,
                    c,
                    regret_scale,
                )
                metrics, pred, _, _ = predict_multiclass(
                    model,
                    splits[fold]["dev_df"],
                    splits[fold]["dev_market"],
                    feature_columns,
                )
                fold_metrics_by_key[key][fold] = metrics
                if fold in TUNE:
                    tune_pnls.append(pred["realized_pnl"].to_numpy(dtype=float))
                    trade_counts += metrics["trade_count"]
                    sample_counts += metrics["sample_count"]
                    tune_wrong_side_loss_abs += metrics["wrong_side_loss_abs"]
            tune_concat = np.concatenate(tune_pnls) if tune_pnls else np.array([], dtype=float)
            holdout_sum = float(sum(fold_metrics_by_key[key][fold]["sum_pnl"] for fold in HOLDOUT))
            holdout_both_positive = bool(all(fold_metrics_by_key[key][fold]["sum_pnl"] > 0 for fold in HOLDOUT))
            candidates.append(
                {
                    "c": c,
                    "regret_scale": regret_scale,
                    "tune_score": float(sum(fold_metrics_by_key[key][fold]["sum_pnl"] for fold in TUNE)),
                    "tune_wrong_side_loss_abs": tune_wrong_side_loss_abs,
                    "tune_drawdown": max_drawdown(tune_concat),
                    "tune_trade_rate": float(trade_counts / sample_counts) if sample_counts else 0.0,
                    "holdout_sum": holdout_sum,
                    "holdout_passed": bool(holdout_sum > 0 and holdout_both_positive),
                }
            )
    winner = select_candidate(candidates)
    winner_key = f"c={winner['c']:.2f}|regret_scale={winner['regret_scale']:.2f}"
    model = fit_regret_weighted_model(
        splits["btest"]["train_df"],
        splits["btest"]["train_market"],
        feature_columns,
        float(winner["c"]),
        float(winner["regret_scale"]),
    )
    btest_metrics, btest_pred, confusion, wrong = predict_multiclass(
        model,
        splits["btest"]["dev_df"],
        splits["btest"]["dev_market"],
        feature_columns,
    )
    btest_metrics["holdout_passed"] = bool(winner["holdout_passed"])
    btest_metrics["btest_evaluation_count"] = 1
    btest_metrics["baseline_family"] = "cross_side_regret_weighted_ce"
    btest_metrics["no_trade_and_oracle"] = no_trade_and_oracle_metrics(splits["btest"]["dev_market"])
    metrics_bdev = {
        "selection_folds": TUNE,
        "holdout_folds": HOLDOUT,
        "candidates": candidates,
        "winner": winner,
        "fold_metrics": fold_metrics_by_key[winner_key],
    }
    report = (
        "# DP3 cross-side regret weighted CE\n\n"
        f"Selected c `{winner['c']:.2f}` and regret_scale `{winner['regret_scale']:.2f}`. "
        f"Holdout passed `{bool(winner['holdout_passed'])}` with holdout_sum `{winner['holdout_sum']:.2f}`. "
        f"B_test sum_pnl `{btest_metrics['sum_pnl']:.2f}`, wrong_side_loss_abs `{btest_metrics['wrong_side_loss_abs']:.2f}`.\n"
    )
    write_node(
        "DP3_cross_side_regret_weighted_ce",
        (
            "experiment_id: DP3_cross_side_regret_weighted_ce\n"
            f"c: {winner['c']:.2f}\n"
            f"regret_scale: {winner['regret_scale']:.2f}\n"
            "objective: multi:softprob\n"
            "num_class: 3\n"
            "max_depth: 3\n"
            "eta: 0.05\n"
            "subsample: 0.9\n"
            "colsample_bytree: 0.8\n"
            "min_child_weight: 50\n"
            "lambda: 5\n"
            "num_boost_round: 500\n"
            "early_stopping_rounds: 50\n"
        ),
        metrics_bdev,
        btest_metrics,
        btest_pred,
        confusion,
        wrong,
        feature_columns,
        report,
    )
    row = {
        "experiment_id": "DP3_cross_side_regret_weighted_ce",
        "btest_status": "evaluated_once",
        "btest_sum_pnl": btest_metrics["sum_pnl"],
        "holdout_passed": bool(winner["holdout_passed"]),
    }
    return row, {
        "node_name": "DP3_cross_side_regret_weighted_ce",
        "btest_pred": btest_pred,
        "winner": winner,
    }


def apply_price_aware_utility(pred: pd.DataFrame, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    denom = pred["pi_yes"].to_numpy(dtype=float) + pred["pi_no"].to_numpy(dtype=float)
    denom = np.where(denom <= 1e-9, 1.0, denom)
    s_yes = pred["pi_yes"].to_numpy(dtype=float) / denom
    s_no = pred["pi_no"].to_numpy(dtype=float) / denom
    utility_yes = s_yes - pred["m_yes"].to_numpy(dtype=float) - tau
    utility_no = s_no - pred["m_no"].to_numpy(dtype=float) - tau
    utility_none = np.zeros(len(pred), dtype=float)
    action = np.column_stack([utility_yes, utility_no, utility_none]).argmax(axis=1)
    return action, utility_yes, utility_no, utility_none


def run_dp4(
    feature_columns: list[str],
    splits: dict[str, dict[str, pd.DataFrame]],
    source_nodes: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_names = [
        "DP2_soft_reward_ce",
        "DP3_cross_side_regret_weighted_ce",
        "DP5_xgb_direct_pnl_objective",
    ]
    candidates = []
    source_fold_predictions = source_nodes["fold_predictions"]
    candidate_fold_predictions: dict[str, dict[str, pd.DataFrame]] = {}
    for source_name in source_names:
        for tau in DP4_TAUS:
            key = f"{source_name}|tau={tau:.2f}"
            candidate_fold_predictions[key] = {}
            tune_pnls: list[np.ndarray] = []
            trade_counts = 0
            sample_counts = 0
            tune_wrong_side_loss_abs = 0.0
            fold_metrics = {}
            for fold in FOLDS:
                pred = source_fold_predictions[source_name][fold]
                market = splits[fold]["dev_market"]
                action, utility_yes, utility_no, utility_none = apply_price_aware_utility(pred, tau)
                metrics, utility_pred, _, _ = evaluate_actions(
                    market,
                    action,
                    pred["pi_yes"].to_numpy(dtype=float),
                    pred["pi_no"].to_numpy(dtype=float),
                    pred["pi_none"].to_numpy(dtype=float),
                    utility_yes,
                    utility_no,
                    utility_none,
                )
                fold_metrics[fold] = metrics
                candidate_fold_predictions[key][fold] = utility_pred
                if fold in TUNE:
                    tune_pnls.append(utility_pred["realized_pnl"].to_numpy(dtype=float))
                    trade_counts += metrics["trade_count"]
                    sample_counts += metrics["sample_count"]
                    tune_wrong_side_loss_abs += metrics["wrong_side_loss_abs"]
            tune_concat = np.concatenate(tune_pnls) if tune_pnls else np.array([], dtype=float)
            holdout_sum = float(sum(fold_metrics[fold]["sum_pnl"] for fold in HOLDOUT))
            holdout_both_positive = bool(all(fold_metrics[fold]["sum_pnl"] > 0 for fold in HOLDOUT))
            candidates.append(
                {
                    "source_model": source_name,
                    "tau": tau,
                    "tune_score": float(sum(fold_metrics[fold]["sum_pnl"] for fold in TUNE)),
                    "tune_wrong_side_loss_abs": tune_wrong_side_loss_abs,
                    "tune_drawdown": max_drawdown(tune_concat),
                    "tune_trade_rate": float(trade_counts / sample_counts) if sample_counts else 0.0,
                    "holdout_sum": holdout_sum,
                    "holdout_passed": bool(holdout_sum > 0 and holdout_both_positive),
                    "fold_metrics": fold_metrics,
                }
            )
    winner = select_candidate(candidates)
    btest_market = splits["btest"]["dev_market"]
    source_pred = source_nodes["btest_predictions"][winner["source_model"]]
    action, utility_yes, utility_no, utility_none = apply_price_aware_utility(source_pred, float(winner["tau"]))
    btest_metrics, btest_pred, confusion, wrong = evaluate_actions(
        btest_market,
        action,
        source_pred["pi_yes"].to_numpy(dtype=float),
        source_pred["pi_no"].to_numpy(dtype=float),
        source_pred["pi_none"].to_numpy(dtype=float),
        utility_yes,
        utility_no,
        utility_none,
    )
    btest_metrics["holdout_passed"] = bool(winner["holdout_passed"])
    btest_metrics["btest_evaluation_count"] = 1
    btest_metrics["baseline_family"] = "price_aware_utility"
    metrics_bdev = {
        "selection_folds": TUNE,
        "holdout_folds": HOLDOUT,
        "candidates": [
            {k: v for k, v in row.items() if k != "fold_metrics"} for row in candidates
        ],
        "winner": {k: v for k, v in winner.items() if k != "fold_metrics"},
        "fold_metrics": winner["fold_metrics"],
    }
    report = (
        "# DP4 price-aware utility layer\n\n"
        f"Selected source_model `{winner['source_model']}` and tau `{winner['tau']:.2f}`. "
        f"Holdout passed `{bool(winner['holdout_passed'])}` with holdout_sum `{winner['holdout_sum']:.2f}`. "
        f"B_test sum_pnl `{btest_metrics['sum_pnl']:.2f}`, wrong_side_loss_abs `{btest_metrics['wrong_side_loss_abs']:.2f}`.\n"
    )
    write_node(
        "DP4_price_aware_utility_layer",
        (
            "experiment_id: DP4_price_aware_utility_layer\n"
            f"source_model: {winner['source_model']}\n"
            f"tau: {winner['tau']:.2f}\n"
            "decision_rule: argmax(s_yes-m_yes-tau, s_no-m_no-tau, 0)\n"
        ),
        metrics_bdev,
        btest_metrics,
        btest_pred,
        confusion,
        wrong,
        feature_columns,
        report,
    )
    row = {
        "experiment_id": "DP4_price_aware_utility_layer",
        "btest_status": "evaluated_once",
        "btest_sum_pnl": btest_metrics["sum_pnl"],
        "holdout_passed": bool(winner["holdout_passed"]),
    }
    return row, {
        "node_name": "DP4_price_aware_utility_layer",
        "btest_pred": btest_pred,
        "winner": winner,
        "fold_predictions": candidate_fold_predictions[
            f"{winner['source_model']}|tau={winner['tau']:.2f}"
        ],
    }


def run_dp5(feature_columns: list[str], splits: dict[str, dict[str, pd.DataFrame]]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = []
    fold_predictions_by_price_mode: dict[str, dict[str, pd.DataFrame]] = {
        "actual-price": {},
        "proxy-price": {},
    }
    actual_best_payload: dict[str, Any] | None = None
    proxy_best_payload: dict[str, Any] | None = None
    for price_mode in PRICE_MODES:
        local_candidates = []
        local_fold_predictions: dict[str, dict[str, pd.DataFrame]] = {}
        for c in DP5_CS:
            for temperature in DP5_TEMPS:
                for max_depth in DP5_MAX_DEPTHS:
                    for eta in DP5_ETAS:
                        key = f"{price_mode}|c={c:.2f}|temperature={temperature:.2f}|max_depth={max_depth}|eta={eta:.2f}"
                        local_fold_predictions[key] = {}
                        tune_pnls: list[np.ndarray] = []
                        trade_counts = 0
                        sample_counts = 0
                        tune_wrong_side_loss_abs = 0.0
                        fold_metrics = {}
                        training_meta = {}
                        for fold in FOLDS:
                            booster, meta, _ = fit_direct_pnl_model(
                                splits[fold]["train_df"],
                                splits[fold]["train_market"],
                                feature_columns,
                                c,
                                temperature,
                                max_depth,
                                eta,
                                price_mode,
                            )
                            metrics, pred, _, _, pred_meta, _ = predict_direct_pnl_booster(
                                booster,
                                splits[fold]["dev_df"],
                                splits[fold]["dev_market"],
                                feature_columns,
                                temperature,
                            )
                            fold_metrics[fold] = metrics
                            local_fold_predictions[key][fold] = pred
                            training_meta[fold] = meta | pred_meta
                            if fold in TUNE:
                                tune_pnls.append(pred["realized_pnl"].to_numpy(dtype=float))
                                trade_counts += metrics["trade_count"]
                                sample_counts += metrics["sample_count"]
                                tune_wrong_side_loss_abs += metrics["wrong_side_loss_abs"]
                        tune_concat = np.concatenate(tune_pnls) if tune_pnls else np.array([], dtype=float)
                        holdout_sum = float(sum(fold_metrics[fold]["sum_pnl"] for fold in HOLDOUT))
                        holdout_both_positive = bool(all(fold_metrics[fold]["sum_pnl"] > 0 for fold in HOLDOUT))
                        local_candidates.append(
                            {
                                "price_mode": price_mode,
                                "c": c,
                                "temperature": temperature,
                                "max_depth": max_depth,
                                "eta": eta,
                                "tune_score": float(sum(fold_metrics[fold]["sum_pnl"] for fold in TUNE)),
                                "tune_wrong_side_loss_abs": tune_wrong_side_loss_abs,
                                "tune_drawdown": max_drawdown(tune_concat),
                                "tune_trade_rate": float(trade_counts / sample_counts) if sample_counts else 0.0,
                                "holdout_sum": holdout_sum,
                                "holdout_passed": bool(holdout_sum > 0 and holdout_both_positive),
                                "fold_metrics": fold_metrics,
                                "training_meta": training_meta,
                            }
                        )
        local_winner = select_candidate(local_candidates)
        local_key = (
            f"{price_mode}|c={local_winner['c']:.2f}|temperature={local_winner['temperature']:.2f}"
            f"|max_depth={int(local_winner['max_depth'])}|eta={local_winner['eta']:.2f}"
        )
        booster, meta, all_columns = fit_direct_pnl_model(
            splits["btest"]["train_df"],
            splits["btest"]["train_market"],
            feature_columns,
            float(local_winner["c"]),
            float(local_winner["temperature"]),
            int(local_winner["max_depth"]),
            float(local_winner["eta"]),
            price_mode,
        )
        btest_metrics, btest_pred, confusion, wrong, pred_meta, all_columns = predict_direct_pnl_booster(
            booster,
            splits["btest"]["dev_df"],
            splits["btest"]["dev_market"],
            feature_columns,
            float(local_winner["temperature"]),
        )
        payload = {
            "winner": {k: v for k, v in local_winner.items() if k not in {"fold_metrics", "training_meta"}},
            "fold_metrics": local_winner["fold_metrics"],
            "training_meta": local_winner["training_meta"],
            "btest_metrics": btest_metrics,
            "btest_pred": btest_pred,
            "confusion": confusion,
            "wrong": wrong,
            "btest_meta": meta | pred_meta,
            "feature_columns": all_columns,
        }
        fold_predictions_by_price_mode[price_mode] = local_fold_predictions[local_key]
        candidates.extend(local_candidates)
        if price_mode == "actual-price":
            actual_best_payload = payload
        else:
            proxy_best_payload = payload

    if actual_best_payload is None or proxy_best_payload is None:
        raise RuntimeError("DP5 failed to produce both price modes")

    winner = actual_best_payload["winner"]
    btest_metrics = actual_best_payload["btest_metrics"]
    btest_metrics["holdout_passed"] = bool(winner["holdout_passed"])
    btest_metrics["btest_evaluation_count"] = 1
    btest_metrics["baseline_family"] = "xgb_direct_pnl_objective"
    btest_metrics["price_mode"] = "actual-price"
    metrics_bdev = {
        "selection_folds": TUNE,
        "holdout_folds": HOLDOUT,
        "actual_price_winner": winner,
        "actual_price_fold_metrics": actual_best_payload["fold_metrics"],
        "actual_price_training_meta": actual_best_payload["training_meta"],
        "actual_price_btest_meta": actual_best_payload["btest_meta"],
        "proxy_price_winner": proxy_best_payload["winner"],
        "proxy_price_fold_metrics": proxy_best_payload["fold_metrics"],
        "proxy_price_training_meta": proxy_best_payload["training_meta"],
        "proxy_price_btest_metrics": proxy_best_payload["btest_metrics"],
    }
    report = (
        "# DP5 XGBoost direct PnL objective\n\n"
        f"Actual-price winner: c `{winner['c']:.2f}`, temperature `{winner['temperature']:.2f}`, "
        f"max_depth `{int(winner['max_depth'])}`, eta `{winner['eta']:.2f}`. "
        f"Holdout passed `{bool(winner['holdout_passed'])}` with holdout_sum `{winner['holdout_sum']:.2f}`. "
        f"B_test sum_pnl `{btest_metrics['sum_pnl']:.2f}`, wrong_side_loss_abs `{btest_metrics['wrong_side_loss_abs']:.2f}`. "
        f"Proxy-price B_test sum_pnl `{proxy_best_payload['btest_metrics']['sum_pnl']:.2f}`.\n"
    )
    write_node(
        "DP5_xgb_direct_pnl_objective",
        (
            "experiment_id: DP5_xgb_direct_pnl_objective\n"
            "price_mode: actual-price\n"
            f"c: {winner['c']:.2f}\n"
            f"temperature: {winner['temperature']:.2f}\n"
            f"max_depth: {int(winner['max_depth'])}\n"
            f"eta: {winner['eta']:.2f}\n"
            "objective: custom_expected_reward_surrogate_hessian\n"
            "custom_hessian_mode: surrogate_diagonal\n"
            "subsample: 0.9\n"
            "colsample_bytree: 0.8\n"
            "num_boost_round: 800\n"
            "early_stopping_rounds: 50\n"
        ),
        metrics_bdev,
        btest_metrics,
        actual_best_payload["btest_pred"],
        actual_best_payload["confusion"],
        actual_best_payload["wrong"],
        actual_best_payload["feature_columns"],
        report,
        extra_manifest={
            "research_only_price_and_base_features": True,
            "p_base_source": "precomputed_p_up_from_source_frame",
        },
    )
    row = {
        "experiment_id": "DP5_xgb_direct_pnl_objective",
        "btest_status": "evaluated_once",
        "btest_sum_pnl": btest_metrics["sum_pnl"],
        "holdout_passed": bool(winner["holdout_passed"]),
    }
    return row, {
        "node_name": "DP5_xgb_direct_pnl_objective",
        "btest_pred": actual_best_payload["btest_pred"],
        "winner": winner,
        "fold_predictions": fold_predictions_by_price_mode["actual-price"],
    }


def run_dp6(feature_columns: list[str], splits: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
    candidates = []
    fold_metrics_by_key: dict[str, dict[str, Any]] = {}
    feature_columns_dp6: list[str] | None = None
    for c in CS:
        for temperature in DP2_TEMPS:
            key = f"c={c:.2f}|temperature={temperature:.2f}"
            fold_metrics_by_key[key] = {}
            tune_pnls: list[np.ndarray] = []
            trade_counts = 0
            sample_counts = 0
            tune_wrong_side_loss_abs = 0.0
            for fold in FOLDS:
                train_x, feature_columns_dp6 = build_dp6_frame(
                    splits[fold]["train_df"],
                    splits[fold]["train_market"],
                    feature_columns,
                )
                dev_x, _ = build_dp6_frame(
                    splits[fold]["dev_df"],
                    splits[fold]["dev_market"],
                    feature_columns,
                )
                booster, _ = fit_soft_reward_model_matrix(
                    train_x,
                    splits[fold]["train_market"],
                    c,
                    temperature,
                )
                metrics, pred, _, _ = predict_soft_booster_matrix(
                    booster,
                    dev_x,
                    splits[fold]["dev_market"],
                )
                fold_metrics_by_key[key][fold] = metrics
                if fold in TUNE:
                    tune_pnls.append(pred["realized_pnl"].to_numpy(dtype=float))
                    trade_counts += metrics["trade_count"]
                    sample_counts += metrics["sample_count"]
                    tune_wrong_side_loss_abs += metrics["wrong_side_loss_abs"]
            tune_concat = np.concatenate(tune_pnls) if tune_pnls else np.array([], dtype=float)
            holdout_sum = float(sum(fold_metrics_by_key[key][fold]["sum_pnl"] for fold in HOLDOUT))
            holdout_both_positive = bool(all(fold_metrics_by_key[key][fold]["sum_pnl"] > 0 for fold in HOLDOUT))
            candidates.append(
                {
                    "c": c,
                    "temperature": temperature,
                    "tune_score": float(sum(fold_metrics_by_key[key][fold]["sum_pnl"] for fold in TUNE)),
                    "tune_wrong_side_loss_abs": tune_wrong_side_loss_abs,
                    "tune_drawdown": max_drawdown(tune_concat),
                    "tune_trade_rate": float(trade_counts / sample_counts) if sample_counts else 0.0,
                    "holdout_sum": holdout_sum,
                    "holdout_passed": bool(holdout_sum > 0 and holdout_both_positive),
                }
            )
    winner = select_candidate(candidates)
    btest_x, feature_columns_dp6 = build_dp6_frame(
        splits["btest"]["train_df"],
        splits["btest"]["train_market"],
        feature_columns,
    )
    booster, _ = fit_soft_reward_model_matrix(
        btest_x,
        splits["btest"]["train_market"],
        float(winner["c"]),
        float(winner["temperature"]),
    )
    btest_dev_x, _ = build_dp6_frame(
        splits["btest"]["dev_df"],
        splits["btest"]["dev_market"],
        feature_columns,
    )
    btest_metrics, btest_pred, confusion, wrong = predict_soft_booster_matrix(
        booster,
        btest_dev_x,
        splits["btest"]["dev_market"],
    )
    btest_metrics["holdout_passed"] = bool(winner["holdout_passed"])
    btest_metrics["btest_evaluation_count"] = 1
    btest_metrics["baseline_family"] = "optional_price_context_soft_reward_ce"
    metrics_bdev = {
        "selection_folds": TUNE,
        "holdout_folds": HOLDOUT,
        "candidates": candidates,
        "winner": winner,
        "fold_metrics": fold_metrics_by_key[f"c={winner['c']:.2f}|temperature={winner['temperature']:.2f}"],
    }
    report = (
        "# DP6 optional price context model\n\n"
        f"Selected c `{winner['c']:.2f}` and temperature `{winner['temperature']:.2f}` with research-only "
        "price context features. "
        f"Holdout passed `{bool(winner['holdout_passed'])}` with holdout_sum `{winner['holdout_sum']:.2f}`. "
        f"B_test sum_pnl `{btest_metrics['sum_pnl']:.2f}`, wrong_side_loss_abs `{btest_metrics['wrong_side_loss_abs']:.2f}`.\n"
    )
    write_node(
        "DP6_optional_price_context_model",
        (
            "experiment_id: DP6_optional_price_context_model\n"
            f"c: {winner['c']:.2f}\n"
            f"temperature: {winner['temperature']:.2f}\n"
            "objective: custom_soft_target_ce\n"
            "research_only_price_context: true\n"
        ),
        metrics_bdev,
        btest_metrics,
        btest_pred,
        confusion,
        wrong,
        feature_columns_dp6 or feature_columns,
        report,
        extra_manifest={
            "research_only_price_context": True,
        },
    )
    return {
        "experiment_id": "DP6_optional_price_context_model",
        "btest_status": "evaluated_once",
        "btest_sum_pnl": btest_metrics["sum_pnl"],
        "holdout_passed": bool(winner["holdout_passed"]),
    }


def reeval_universe(pred: pd.DataFrame, universe_mask: np.ndarray) -> dict[str, Any]:
    scoped = pred.loc[universe_mask].reset_index(drop=True)
    trade_mask = scoped["action"] != "NO_TRADE"
    wrong_side_mask = (
        ((scoped["action"] == "BUY_YES") & (scoped["y"] == 0))
        | ((scoped["action"] == "BUY_NO") & (scoped["y"] == 1))
    )
    return {
        "sample_count": int(len(scoped)),
        "trade_count": int(trade_mask.sum()),
        "trade_rate": float(trade_mask.mean()) if len(scoped) else 0.0,
        "sum_pnl": float(scoped["realized_pnl"].sum()),
        "wrong_side_loss": float(scoped.loc[wrong_side_mask, "realized_pnl"].sum()),
    }


def run_dp7(
    feature_columns: list[str],
    splits: dict[str, dict[str, pd.DataFrame]],
    winners: dict[str, dict[str, Any]],
    source_fold_predictions: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for node_name, payload in winners.items():
        node_diag = {"w1_w6": {}, "btest": {}}
        for fold in FOLDS:
            pred = source_fold_predictions[node_name][fold]
            universe_a = pred["m_yes"].notna() | pred["m_no"].notna()
            universe_b = pred["m_yes"].notna() & pred["m_no"].notna()
            node_diag["w1_w6"][fold] = {
                "coverage_a": float(universe_a.mean()),
                "coverage_b": float(universe_b.mean()),
                "universe_a": reeval_universe(pred, universe_a.to_numpy()),
                "universe_b": reeval_universe(pred, universe_b.to_numpy()),
            }
        pred = payload["btest_pred"]
        universe_a = pred["m_yes"].notna() | pred["m_no"].notna()
        universe_b = pred["m_yes"].notna() & pred["m_no"].notna()
        node_diag["btest"] = {
            "coverage_a": float(universe_a.mean()),
            "coverage_b": float(universe_b.mean()),
            "universe_a": reeval_universe(pred, universe_a.to_numpy()),
            "universe_b": reeval_universe(pred, universe_b.to_numpy()),
        }
        diagnostics[node_name] = node_diag

    metrics_btest = {
        "diagnostics": diagnostics,
        "btest_evaluation_count": 1,
    }
    n = len(splits["btest"]["dev_market"])
    noop_metrics, pred, confusion, wrong = evaluate_actions(
        splits["btest"]["dev_market"],
        np.full(n, ACTION_NONE, dtype=int),
        np.zeros(n, dtype=float),
        np.zeros(n, dtype=float),
        np.ones(n, dtype=float),
        np.zeros(n, dtype=float),
        np.zeros(n, dtype=float),
        np.zeros(n, dtype=float),
    )
    report = (
        "# DP7 stable universe diagnostic\n\n"
        "Re-evaluated DP1/DP2/DP3/DP4/DP5 winners under `has_yes or has_no` and `has_yes and has_no` universes "
        "without retuning. See metrics_btest.json for per-fold coverage and PnL decompositions.\n"
    )
    write_node(
        "DP7_stable_universe_diagnostic",
        "experiment_id: DP7_stable_universe_diagnostic\nretune: false\n",
        {"source_nodes": list(winners)},
        metrics_btest | {"noop_metrics": noop_metrics},
        pred,
        confusion,
        wrong,
        feature_columns,
        report,
    )
    return {
        "experiment_id": "DP7_stable_universe_diagnostic",
        "btest_status": "diagnostic_once",
        "btest_sum_pnl": None,
        "holdout_passed": None,
    }


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
    report = (
        "# Research Report - Direct Policy Loss Price Aware 0708\n\n"
        "Implemented `arbor_research_0708_v2.md` as an isolated Arbor session on the 1:08 market-order universe. "
        "The deploy feature manifest stayed fixed for DP0-DP4; DP5 added research-only price/base features; "
        "B_test was recorded once per named node.\n\n"
        "## Final B_test\n\n"
        + table.to_markdown(index=False)
        + "\n"
    )
    (SESSION / "REPORT.md").write_text(report, encoding="utf-8")
    table.to_csv(SESSION / "direct_policy_btest_ledger.csv", index=False)


def main() -> None:
    SESSION.mkdir(parents=True, exist_ok=True)
    feature_columns = load_feature_columns()
    splits = load_all_splits()
    rows: list[dict[str, Any]] = []

    dp0 = run_dp0(feature_columns, splits)
    rows.append(dp0)

    dp1_row, dp1_payload = run_dp1(feature_columns, splits)
    rows.append(dp1_row)

    dp2_row, dp2_payload = run_dp2(feature_columns, splits)
    rows.append(dp2_row)

    dp3_row, dp3_payload = run_dp3(feature_columns, splits)
    rows.append(dp3_row)

    dp5_row, dp5_payload = run_dp5(feature_columns, splits)
    rows.append(dp5_row)

    dp6_row = run_dp6(feature_columns, splits)
    rows.append(dp6_row)

    # Recompute source fold predictions needed by DP4/DP7 from stored winners.
    # DP2/DP3 fold predictions are regenerated deterministically for session-local diagnostics.
    fold_predictions: dict[str, dict[str, pd.DataFrame]] = {
        "DP1_replay_v1_reference": {},
        "DP2_soft_reward_ce": {},
        "DP3_cross_side_regret_weighted_ce": {},
        "DP5_xgb_direct_pnl_objective": dp5_payload["fold_predictions"],
    }
    for fold in FOLDS:
        df = splits[fold]["dev_df"]
        market = splits[fold]["dev_market"]
        tau = float(dp1_payload["winner"]["tau"])
        score = df["p_up"].to_numpy(dtype=float)
        edge_yes = np.where(market["has_yes"].to_numpy(dtype=bool), score - market["m_yes"].to_numpy(dtype=float) - tau, -np.inf)
        edge_no = np.where(market["has_no"].to_numpy(dtype=bool), (1.0 - score) - market["m_no"].to_numpy(dtype=float) - tau, -np.inf)
        action = np.column_stack([edge_yes, edge_no, np.zeros(len(df), dtype=float)]).argmax(axis=1)
        _, pred, _, _ = evaluate_actions(
            market,
            action,
            score,
            1.0 - score,
            np.zeros(len(df), dtype=float),
            edge_yes,
            edge_no,
            np.zeros(len(df), dtype=float),
        )
        fold_predictions["DP1_replay_v1_reference"][fold] = pred

        booster2, _ = fit_soft_reward_model(
            splits[fold]["train_df"],
            splits[fold]["train_market"],
            feature_columns,
            float(dp2_payload["winner"]["c"]),
            float(dp2_payload["winner"]["temperature"]),
        )
        _, pred2, _, _, _ = predict_soft_booster(
            booster2,
            splits[fold]["dev_df"],
            splits[fold]["dev_market"],
            feature_columns,
        )
        fold_predictions["DP2_soft_reward_ce"][fold] = pred2

        model3 = fit_regret_weighted_model(
            splits[fold]["train_df"],
            splits[fold]["train_market"],
            feature_columns,
            float(dp3_payload["winner"]["c"]),
            float(dp3_payload["winner"]["regret_scale"]),
        )
        _, pred3, _, _ = predict_multiclass(
            model3,
            splits[fold]["dev_df"],
            splits[fold]["dev_market"],
            feature_columns,
        )
        fold_predictions["DP3_cross_side_regret_weighted_ce"][fold] = pred3

    dp4_row, dp4_payload = run_dp4(
        feature_columns,
        splits,
        {
            "fold_predictions": fold_predictions,
            "btest_predictions": {
                "DP2_soft_reward_ce": dp2_payload["btest_pred"],
                "DP3_cross_side_regret_weighted_ce": dp3_payload["btest_pred"],
                "DP5_xgb_direct_pnl_objective": dp5_payload["btest_pred"],
            },
        },
    )
    rows.append(dp4_row)

    dp7 = run_dp7(
        feature_columns,
        splits,
        {
            "DP1_replay_v1_reference": dp1_payload,
            "DP2_soft_reward_ce": dp2_payload,
            "DP3_cross_side_regret_weighted_ce": dp3_payload,
            "DP4_price_aware_utility_layer": dp4_payload,
            "DP5_xgb_direct_pnl_objective": dp5_payload,
        },
        {
            **fold_predictions,
            "DP4_price_aware_utility_layer": dp4_payload["fold_predictions"],
        },
    )
    rows.append(dp7)

    write_session_report(rows)
    print(json.dumps(rows, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
