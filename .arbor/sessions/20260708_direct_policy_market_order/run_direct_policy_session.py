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
ROLLING = ROOT / ".arbor" / "sessions" / "20260703_prefinal_rolling" / "folds"
BASELINE_MANIFEST = ROOT / "execution_engine" / "deploy" / "baseline" / "artifact_manifest.json"
TRAIN = ROOT / "price_estimator" / "expected_return" / "experiments" / "20260619_expected_return_trade_coverage_start" / "data" / "expected_return_train.parquet"
BTEST = ROOT / "price_estimator" / "expected_return" / "experiments" / "20260619_expected_return_trade_coverage_start" / "data" / "expected_return_validation.parquet"

FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE = FOLDS[:4]
HOLDOUT = FOLDS[4:]
TAUS = [0.00, 0.01, 0.02]
CS = [0.00, 0.01, 0.02]
ACTION_NAMES = ["BUY_YES", "BUY_NO", "NO_TRADE"]
ACTION_YES = 0
ACTION_NO = 1
ACTION_NONE = 2
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
    "post-decision trade path fields",
}
FORBIDDEN_PREFIXES = ("future_",)


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


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
                "sample_id": str(idx),
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
    x = x.replace([np.inf, -np.inf], np.nan).astype("float32")
    return x


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
    oracle = np.argmax(stack, axis=1)
    top = stack[np.arange(len(stack)), oracle]
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
    oracle_pnl = np.where(np.isfinite(oracle_pnl), oracle_pnl, 0.0)
    return oracle, weight, oracle_pnl, stack


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


def evaluate_actions(
    market: pd.DataFrame,
    action_idx: np.ndarray,
    pi_yes: np.ndarray,
    pi_no: np.ndarray,
    pi_none: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
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
    loss_mask = trade_mask & (realized < 0)
    metrics = {
        "sample_count": int(len(market)),
        "trade_count": int(trade_mask.sum()),
        "trade_rate": float(trade_mask.mean()) if len(market) else 0.0,
        "sum_pnl": float(realized.sum()),
        "mean_pnl_per_trade": float(realized[trade_mask].mean()) if trade_mask.any() else 0.0,
        "YES_pnl": float(realized[yes_mask].sum()),
        "NO_pnl": float(realized[no_mask].sum()),
        "wrong_side_loss": float(realized[loss_mask].sum()),
        "loss_pnl_sum": float(realized[loss_mask].sum()),
        "win_pnl_sum": float(realized[trade_mask & (realized > 0)].sum()),
        "no_trade_count": int((action_idx == ACTION_NONE).sum()),
        "oracle_pnl": float(oracle_pnl.sum()),
        "capture_ratio": float(realized.sum() / oracle_pnl.sum()) if float(oracle_pnl.sum()) else 0.0,
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
            "action": [ACTION_NAMES[i] for i in action_idx],
            "realized_pnl": realized,
            "oracle_action": [ACTION_NAMES[i] for i in oracle_action],
            "oracle_pnl": oracle_pnl,
            "m_yes_trade_time": market["m_yes_trade_time"],
            "m_no_trade_time": market["m_no_trade_time"],
            "has_yes": has_yes,
            "has_no": has_no,
        }
    )
    return metrics, pred


def no_trade_and_oracle_metrics(market: pd.DataFrame) -> dict[str, Any]:
    n = len(market)
    zero = np.zeros(n, dtype=float)
    none = np.full(n, 1.0, dtype=float)
    no_trade_metrics, _ = evaluate_actions(
        market,
        np.full(n, ACTION_NONE, dtype=int),
        zero,
        zero,
        none,
    )
    oracle_action, _, _, _ = compute_rewards(market, 0.0)
    oracle_metrics, _ = evaluate_actions(
        market,
        oracle_action,
        zero,
        zero,
        none,
    )
    return {"no_trade": no_trade_metrics, "oracle": oracle_metrics}


def write_node(
    node_name: str,
    config_text: str,
    metrics_bdev: dict[str, Any],
    metrics_btest: dict[str, Any],
    predictions_btest: pd.DataFrame,
    feature_columns: list[str],
    report_text: str,
) -> None:
    out = EXPERIMENTS / node_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(config_text, encoding="utf-8")
    dump_json(out / "metrics_bdev.json", metrics_bdev)
    dump_json(out / "metrics_btest.json", metrics_btest)
    predictions_btest.to_parquet(out / "predictions_btest.parquet", index=False)
    bad = [
        c
        for c in feature_columns
        if c in FORBIDDEN_EXACT or any(c.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    ]
    dump_json(
        out / "feature_manifest.json",
        {
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
        },
    )
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


def select_candidate(
    candidate_rows: list[dict[str, Any]],
    score_key: str = "tune_score",
) -> dict[str, Any]:
    table = pd.DataFrame(candidate_rows)
    order = table.sort_values(
        [score_key, "tune_drawdown", "tune_trade_rate"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return order.iloc[0].to_dict()


def fit_multiclass_model(
    train_df: pd.DataFrame,
    train_market: pd.DataFrame,
    feature_columns: list[str],
    c: float,
) -> xgb.XGBClassifier:
    x = matrix(train_df, feature_columns)
    y, w, _, _ = compute_rewards(train_market, c)
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
        y[fit_idx],
        sample_weight=w[fit_idx],
        eval_set=[(x.iloc[val_idx], y[val_idx])],
        sample_weight_eval_set=[w[val_idx]],
        verbose=False,
    )
    return model


def predict_multiclass(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    market: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    probs = model.predict_proba(matrix(df, feature_columns))
    has_yes = market["has_yes"].to_numpy(dtype=bool)
    has_no = market["has_no"].to_numpy(dtype=bool)
    masked = probs.copy()
    masked[~has_yes, ACTION_YES] = -np.inf
    masked[~has_no, ACTION_NO] = -np.inf
    action = masked.argmax(axis=1)
    return evaluate_actions(market, action, probs[:, 0], probs[:, 1], probs[:, 2])


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
        }
    btest_market = splits["btest"]["dev_market"]
    btest_q = {
        "row_count": int(len(btest_market)),
        "m_yes_coverage": float(btest_market["has_yes"].mean()),
        "m_no_coverage": float(btest_market["has_no"].mean()),
        "m_any_coverage": float(btest_market["has_any"].mean()),
        "late_join_count": 0,
    }
    n = len(btest_market)
    pred = pd.DataFrame(
        {
            "sample_id": btest_market["sample_id"],
            "decision_time": btest_market["decision_time"],
            "y": btest_market["y"],
            "m_yes": btest_market["m_yes"],
            "m_no": btest_market["m_no"],
            "pi_yes": np.zeros(n, dtype=float),
            "pi_no": np.zeros(n, dtype=float),
            "pi_none": np.ones(n, dtype=float),
            "action": ["NO_TRADE"] * n,
            "realized_pnl": np.zeros(n, dtype=float),
            "oracle_action": ["NO_TRADE"] * n,
            "oracle_pnl": np.zeros(n, dtype=float),
        }
    )
    metrics_bdev = {"folds": fold_qa}
    metrics_btest = {"qa": btest_q, "btest_evaluation_count": 1}
    report = (
        "# DP0 frame QA\n\n"
        f"B_test rows `{btest_q['row_count']}`, m_yes coverage `{btest_q['m_yes_coverage']:.4f}`, "
        f"m_no coverage `{btest_q['m_no_coverage']:.4f}`, any-side coverage `{btest_q['m_any_coverage']:.4f}`, "
        "late joins `0`.\n"
    )
    write_node(
        "DP0_frame_qa",
        "experiment_id: DP0_frame_qa\ndecision_offset_seconds: 68\nm_no_proxy: false\n",
        metrics_bdev,
        metrics_btest,
        pred,
        feature_columns,
        report,
    )
    return {
        "experiment_id": "DP0_frame_qa",
        "btest_status": "qa_once",
        "btest_sum_pnl": None,
        "holdout_passed": None,
    }


def run_dp1(feature_columns: list[str], splits: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
    candidates = []
    per_fold_metrics: dict[str, dict[str, Any]] = {}
    per_fold_predictions: dict[str, dict[str, pd.DataFrame]] = {}
    for tau in TAUS:
        key = f"tau={tau:.2f}"
        per_fold_metrics[key] = {}
        per_fold_predictions[key] = {}
        tune_pnls: list[np.ndarray] = []
        trade_counts = 0
        sample_counts = 0
        for fold in FOLDS:
            df = splits[fold]["dev_df"]
            market = splits[fold]["dev_market"]
            score = df["p_up"].to_numpy(dtype=float)
            edge_yes = np.where(market["has_yes"].to_numpy(dtype=bool), score - market["m_yes"].to_numpy(dtype=float) - tau, -np.inf)
            edge_no = np.where(market["has_no"].to_numpy(dtype=bool), (1.0 - score) - market["m_no"].to_numpy(dtype=float) - tau, -np.inf)
            best = np.column_stack([edge_yes, edge_no, np.zeros(len(df), dtype=float)])
            action = best.argmax(axis=1)
            metrics, pred = evaluate_actions(market, action, score, 1.0 - score, np.zeros(len(df), dtype=float))
            per_fold_metrics[key][fold] = metrics
            per_fold_predictions[key][fold] = pred
            if fold in TUNE:
                tune_pnls.append(pred["realized_pnl"].to_numpy(dtype=float))
                trade_counts += metrics["trade_count"]
                sample_counts += metrics["sample_count"]
        tune_concat = np.concatenate(tune_pnls) if tune_pnls else np.array([], dtype=float)
        holdout_sum = float(sum(per_fold_metrics[key][fold]["sum_pnl"] for fold in HOLDOUT))
        holdout_both_positive = bool(all(per_fold_metrics[key][fold]["sum_pnl"] > 0 for fold in HOLDOUT))
        candidates.append(
            {
                "tau": tau,
                "tune_score": float(sum(per_fold_metrics[key][fold]["sum_pnl"] for fold in TUNE)),
                "tune_drawdown": max_drawdown(tune_concat),
                "tune_trade_rate": float(trade_counts / sample_counts) if sample_counts else 0.0,
                "holdout_sum": holdout_sum,
                "holdout_passed": bool(holdout_sum > 0 and holdout_both_positive),
                **{f"{fold}_sum_pnl": per_fold_metrics[key][fold]["sum_pnl"] for fold in FOLDS},
            }
        )
    winner = select_candidate(candidates)
    key = f"tau={winner['tau']:.2f}"
    btest_df = splits["btest"]["dev_df"]
    btest_market = splits["btest"]["dev_market"]
    score = btest_df["p_up"].to_numpy(dtype=float)
    edge_yes = np.where(btest_market["has_yes"].to_numpy(dtype=bool), score - btest_market["m_yes"].to_numpy(dtype=float) - float(winner["tau"]), -np.inf)
    edge_no = np.where(btest_market["has_no"].to_numpy(dtype=bool), (1.0 - score) - btest_market["m_no"].to_numpy(dtype=float) - float(winner["tau"]), -np.inf)
    action = np.column_stack([edge_yes, edge_no, np.zeros(len(btest_df), dtype=float)]).argmax(axis=1)
    btest_metrics, btest_pred = evaluate_actions(btest_market, action, score, 1.0 - score, np.zeros(len(btest_df), dtype=float))
    btest_metrics["holdout_passed"] = bool(winner["holdout_passed"])
    btest_metrics["btest_evaluation_count"] = 1
    btest_metrics["baseline_family"] = "static_market_ev"
    btest_metrics["no_trade_and_oracle"] = no_trade_and_oracle_metrics(btest_market)
    metrics_bdev = {
        "selection_folds": TUNE,
        "holdout_folds": HOLDOUT,
        "candidates": candidates,
        "winner": winner,
        "fold_metrics": per_fold_metrics[key],
    }
    report = (
        "# DP1 static EV baseline\n\n"
        f"Selected tau `{winner['tau']:.2f}` on w1-w4. Holdout passed `{bool(winner['holdout_passed'])}` "
        f"with holdout_sum `{winner['holdout_sum']:.2f}`. "
        f"B_test sum_pnl `{btest_metrics['sum_pnl']:.2f}`, trade_rate `{btest_metrics['trade_rate']:.4f}`.\n"
    )
    write_node(
        "DP1_static_ev_baseline",
        f"experiment_id: DP1_static_ev_baseline\nscore_column: p_up\ntau: {winner['tau']:.2f}\ndecision_rule: argmax(score-m_yes-tau, 1-score-m_no-tau, 0)\n",
        metrics_bdev,
        btest_metrics,
        btest_pred,
        feature_columns,
        report,
    )
    return {
        "experiment_id": "DP1_static_ev_baseline",
        "btest_status": "evaluated_once",
        "btest_sum_pnl": btest_metrics["sum_pnl"],
        "holdout_passed": bool(winner["holdout_passed"]),
    }


def run_dp2(feature_columns: list[str], splits: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
    candidates = []
    fold_metrics_by_c: dict[str, dict[str, Any]] = {}
    for c in CS:
        key = f"c={c:.2f}"
        fold_metrics_by_c[key] = {}
        tune_pnls: list[np.ndarray] = []
        trade_counts = 0
        sample_counts = 0
        for fold in FOLDS:
            model = fit_multiclass_model(
                splits[fold]["train_df"],
                splits[fold]["train_market"],
                feature_columns,
                c,
            )
            metrics, pred = predict_multiclass(
                model,
                splits[fold]["dev_df"],
                splits[fold]["dev_market"],
                feature_columns,
            )
            fold_metrics_by_c[key][fold] = metrics
            if fold in TUNE:
                tune_pnls.append(pred["realized_pnl"].to_numpy(dtype=float))
                trade_counts += metrics["trade_count"]
                sample_counts += metrics["sample_count"]
        tune_concat = np.concatenate(tune_pnls) if tune_pnls else np.array([], dtype=float)
        holdout_sum = float(sum(fold_metrics_by_c[key][fold]["sum_pnl"] for fold in HOLDOUT))
        holdout_both_positive = bool(all(fold_metrics_by_c[key][fold]["sum_pnl"] > 0 for fold in HOLDOUT))
        candidates.append(
            {
                "c": c,
                "tune_score": float(sum(fold_metrics_by_c[key][fold]["sum_pnl"] for fold in TUNE)),
                "tune_drawdown": max_drawdown(tune_concat),
                "tune_trade_rate": float(trade_counts / sample_counts) if sample_counts else 0.0,
                "holdout_sum": holdout_sum,
                "holdout_passed": bool(holdout_sum > 0 and holdout_both_positive),
                **{f"{fold}_sum_pnl": fold_metrics_by_c[key][fold]["sum_pnl"] for fold in FOLDS},
            }
        )
    winner = select_candidate(candidates)
    winner_c = float(winner["c"])
    model = fit_multiclass_model(
        splits["btest"]["train_df"],
        splits["btest"]["train_market"],
        feature_columns,
        winner_c,
    )
    btest_metrics, btest_pred = predict_multiclass(
        model,
        splits["btest"]["dev_df"],
        splits["btest"]["dev_market"],
        feature_columns,
    )
    btest_metrics["holdout_passed"] = bool(winner["holdout_passed"])
    btest_metrics["btest_evaluation_count"] = 1
    btest_metrics["baseline_family"] = "direct_policy_regret_ce"
    btest_metrics["no_trade_and_oracle"] = no_trade_and_oracle_metrics(splits["btest"]["dev_market"])
    metrics_bdev = {
        "selection_folds": TUNE,
        "holdout_folds": HOLDOUT,
        "candidates": candidates,
        "winner": winner,
        "fold_metrics": fold_metrics_by_c[f"c={winner_c:.2f}"],
    }
    report = (
        "# DP2 direct policy regret CE\n\n"
        f"Selected c `{winner_c:.2f}` on w1-w4. Holdout passed `{bool(winner['holdout_passed'])}` "
        f"with holdout_sum `{winner['holdout_sum']:.2f}`. "
        f"B_test sum_pnl `{btest_metrics['sum_pnl']:.2f}`, trade_rate `{btest_metrics['trade_rate']:.4f}`.\n"
    )
    write_node(
        "DP2_direct_policy_regret_ce",
        (
            "experiment_id: DP2_direct_policy_regret_ce\n"
            f"c: {winner_c:.2f}\n"
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
            "internal_eval_split: last_15pct_of_train_chronological\n"
        ),
        metrics_bdev,
        btest_metrics,
        btest_pred,
        feature_columns,
        report,
    )
    return {
        "experiment_id": "DP2_direct_policy_regret_ce",
        "btest_status": "evaluated_once",
        "btest_sum_pnl": btest_metrics["sum_pnl"],
        "holdout_passed": bool(winner["holdout_passed"]),
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
        "# Research Report — Direct Policy Market Order 0708\n\n"
        "Implemented `arbor_research_0708.md` as a standalone Arbor session on the full `1:08` market-order universe. "
        "The deploy direction feature manifest stayed fixed at 569 columns, no deploy artifact or live config changed, and B_test was recorded once for every named node.\n\n"
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
    rows = [
        run_dp0(feature_columns, splits),
        run_dp1(feature_columns, splits),
        run_dp2(feature_columns, splits),
    ]
    write_session_report(rows)
    print(json.dumps(rows, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
