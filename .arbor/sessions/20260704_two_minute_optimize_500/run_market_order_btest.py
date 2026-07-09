#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
OUT = SESSION / "experiments" / "8_market_order_btest"
FULL_OUT = SESSION / "experiments" / "9_full_market_order_btest"
SOURCE = SESSION.parent / "20260704_two_minute_shift"
BTEST = ROOT / "price_estimator" / "expected_return" / "experiments" / "20260619_expected_return_trade_coverage_start" / "data" / "expected_return_validation.parquet"
TRADE_DIR = ROOT / "price_estimator" / "data" / "sell_taker_trades_daily"

WINNER_PARAMS = {
    "q_source": "lgbm",
    "q_weight": 0.25,
    "direct_weight": 0.75,
    "gc_power": 2.0,
    "h_scale": 1.1,
    "min_ev": 0.02,
}
OUTCOME_MAP = {"UP": "UP", "DOWN": "DOWN"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("two_min_base", SESSION / "run_joint_fill_value.py")
factorized = load_module("two_min_factorized", base.JOINT_PATH)
ensemble = load_module("two_min_ensemble", SESSION / "run_joint_factorized_ensemble.py")


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def trade_files(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    dates = set(pd.date_range(start.normalize(), end.normalize(), freq="D").strftime("%Y-%m-%d"))
    return sorted(p for p in TRADE_DIR.glob("date=*.parquet") if p.stem.removeprefix("date=") in dates)


def load_trade_lookup(start: pd.Timestamp, end: pd.Timestamp) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    files = trade_files(start, end)
    if not files:
        raise RuntimeError(f"no trade files for {start}..{end}")
    trades = pd.concat(
        [pd.read_parquet(p, columns=["condition_id", "outcome", "price", "trade_time"]) for p in files],
        ignore_index=True,
    )
    trades["condition_id"] = trades["condition_id"].astype(str)
    trades["outcome"] = trades["outcome"].astype(str).str.upper()
    trades["trade_time"] = pd.to_datetime(trades["trade_time"], utc=True)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades = trades.dropna(subset=["price", "trade_time"]).sort_values("trade_time")
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, group in trades.groupby(["condition_id", "outcome"], sort=False):
        lookup[key] = (group["trade_time"].astype("int64").to_numpy(), group["price"].to_numpy(float))
    return lookup


def last_trade_price(
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    condition_id: str,
    outcome: str,
    cutoff_ns: int,
) -> tuple[float, pd.Timestamp | pd.NaT]:
    payload = lookup.get((condition_id, outcome))
    if payload is None:
        return float("nan"), pd.NaT
    times, prices = payload
    idx = int(np.searchsorted(times, cutoff_ns, side="right") - 1)
    if idx < 0:
        return float("nan"), pd.NaT
    return float(prices[idx]), pd.to_datetime(times[idx], utc=True)


def two_min_btest_frame() -> Path:
    out = OUT / "data" / "btest_two_minute.parquet"
    if out.exists():
        return out
    df = pd.read_parquet(BTEST)
    df["market_t0"] = pd.to_datetime(df["market_t0"], utc=True)
    df["timestamp"] = df["market_t0"]
    df["decision_time"] = df["market_t0"] + pd.Timedelta(minutes=2)
    df["feature_timestamp"] = df["decision_time"]
    df["decision_alignment_mode"] = "market_t0_plus_2m_btest"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def btest_item() -> dict[str, Any]:
    dev_path = two_min_btest_frame()
    train_path = SOURCE / "folds" / "w6" / "data" / "train.parquet"
    hazard_path = SOURCE / "folds" / "w6" / "models" / "hazard_survival_cdf.pt"
    # base.fit_fold is fold-path based, so build the direct component explicitly for B_test.
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    cols, _, _, _ = factorized.load_hazard(hazard_path)
    xtr = factorized.matrix(train.loc[train["threshold_accepted"].astype(bool)], cols)
    xdv = factorized.matrix(dev.loc[dev["threshold_accepted"].astype(bool)], cols)
    accepted_train = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    accepted_dev = dev.loc[dev["threshold_accepted"].astype(bool)].reset_index(drop=True)

    from lightgbm import LGBMClassifier

    q_model = LGBMClassifier(
        n_estimators=450,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=7,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.45,
        reg_alpha=2.0,
        reg_lambda=10.0,
        random_state=20260704,
        verbosity=-1,
        n_jobs=-1,
    )
    q_model.fit(xtr, accepted_train["correct"].astype(int).to_numpy())
    top_index = np.argsort(np.asarray(q_model.feature_importances_))[-120:]
    q = q_model.predict_proba(xdv)[:, 1]
    xtr_top = xtr.iloc[:, top_index].to_numpy(np.float32)
    xdv_top = xdv.iloc[:, top_index].to_numpy(np.float32)
    repeated = np.repeat(xtr_top, len(base.BIDS), axis=0)
    bid_column = np.tile(base.BIDS.astype(np.float32), len(accepted_train))[:, None]
    expanded_x = np.concatenate([repeated, bid_column], axis=1)
    correct = accepted_train["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(accepted_train["chosen_low"], errors="coerce").to_numpy(float)
    expanded_y = (correct[:, None] & np.isfinite(low[:, None]) & (low[:, None] <= base.BIDS[None, :] + 1e-12)).astype(np.uint8).reshape(-1)
    model = LGBMClassifier(
        n_estimators=550,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=7,
        min_child_samples=250,
        subsample=0.85,
        colsample_bytree=0.60,
        reg_alpha=2.0,
        reg_lambda=12.0,
        monotone_constraints=[0] * xtr_top.shape[1] + [1],
        random_state=20260704,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(expanded_x, expanded_y)
    dev_expanded = np.concatenate(
        [np.repeat(xdv_top, len(base.BIDS), axis=0), np.tile(base.BIDS.astype(np.float32), len(accepted_dev))[:, None]],
        axis=1,
    )
    h = np.maximum.accumulate(model.predict_proba(dev_expanded)[:, 1].reshape(len(accepted_dev), len(base.BIDS)), axis=1)
    direct = {
        "q": q,
        "h": h,
        "correct": accepted_dev["correct"].astype(bool).to_numpy(),
        "low": pd.to_numeric(accepted_dev["chosen_low"], errors="coerce").to_numpy(float),
    }

    prepared = factorized.prepare(train_path, dev_path, hazard_path, 20260704)
    grid = np.asarray(prepared[4], dtype=float)
    indices = [int(np.argmin(np.abs(grid - bid))) for bid in base.BIDS]
    return {
        "direct": direct,
        "accepted": prepared[1],
        "qs": prepared[2],
        "gc": prepared[3][:, indices],
        "sample_count": len(prepared[0]),
    }


def market_order_metrics(item: dict[str, Any], params: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    bid, expected_ev, fill_prob = ensemble.choose(item, params)
    accepted = item["accepted"].reset_index(drop=True).copy()
    submitted = bid > 0
    decision_time = pd.to_datetime(accepted["market_t0"], utc=True) + pd.Timedelta(minutes=2)
    lookup = load_trade_lookup(decision_time.min() - pd.Timedelta(days=1), decision_time.max())
    cutoff_ns = decision_time.astype("int64").to_numpy()
    market_price = np.full(len(accepted), np.nan)
    market_trade_time = [pd.NaT] * len(accepted)
    selected_side = accepted["selected_side"].astype(str).to_numpy()
    for i, row in enumerate(accepted.itertuples(index=False)):
        price, trade_time = last_trade_price(lookup, str(row.condition_id), OUTCOME_MAP[selected_side[i]], int(cutoff_ns[i]))
        market_price[i] = price
        market_trade_time[i] = trade_time
    has_market = np.isfinite(market_price)
    order = submitted & has_market
    correct = accepted["correct"].astype(bool).to_numpy()
    pnl = np.zeros(len(accepted), dtype=float)
    pnl[order & correct] = 1.0 - market_price[order & correct]
    pnl[order & ~correct] = -market_price[order & ~correct]
    late_join = sum(
        1
        for t, dt in zip(market_trade_time, decision_time, strict=True)
        if pd.notna(t) and pd.Timestamp(t) > pd.Timestamp(dt)
    )
    market_bid = np.where(order, market_price, 0.0)
    from types import SimpleNamespace

    metrics = factorized.backtest_metrics(
        accepted,
        SimpleNamespace(
            bid=market_bid,
            expected_ev=expected_ev,
            fill_prob=fill_prob,
            pnl=pnl,
            filled=order,
            printed_filled=order,
        ),
        len(pd.read_parquet(two_min_btest_frame())),
    )
    metrics.update(
        {
            "execution_mode": "market_order_last_sell_taker_before_t0_plus_2m",
            "selected_limit_policy_order_count": float(submitted.sum()),
            "market_price_coverage": float(has_market.mean()),
            "submitted_market_price_coverage": float(has_market[submitted].mean()) if submitted.any() else float("nan"),
            "late_join_count": float(late_join),
            "mean_market_entry": float(np.nanmean(market_price[order])) if order.any() else float("nan"),
            "median_market_entry": float(np.nanmedian(market_price[order])) if order.any() else float("nan"),
        }
    )
    pred = accepted[
        [
            c
            for c in [
                "market_t0",
                "condition_id",
                "polymarket_slug",
                "selected_side",
                "p_up",
                "p_side",
                "target",
                "correct",
                "chosen_low",
            ]
            if c in accepted.columns
        ]
    ].copy()
    pred["decision_time_2m"] = decision_time
    pred["selected_limit_bid"] = bid
    pred["expected_ev"] = expected_ev
    pred["model_fill_prob"] = fill_prob
    pred["submitted_by_limit_policy"] = submitted
    pred["market_entry_price"] = market_price
    pred["market_entry_trade_time"] = market_trade_time
    pred["market_order_submitted"] = order
    pred["realized_pnl_market"] = pnl
    return metrics, pred


def full_market_order_metrics() -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_parquet(two_min_btest_frame()).reset_index(drop=True)
    frame["selected_side"] = frame["selected_side"].astype(str)
    decision_time = pd.to_datetime(frame["market_t0"], utc=True) + pd.Timedelta(minutes=2)
    lookup = load_trade_lookup(decision_time.min() - pd.Timedelta(days=1), decision_time.max())
    cutoff_ns = decision_time.astype("int64").to_numpy()
    market_price = np.full(len(frame), np.nan)
    market_trade_time = [pd.NaT] * len(frame)
    selected_side = frame["selected_side"].to_numpy()
    for i, row in enumerate(frame.itertuples(index=False)):
        price, trade_time = last_trade_price(lookup, str(row.condition_id), OUTCOME_MAP[selected_side[i]], int(cutoff_ns[i]))
        market_price[i] = price
        market_trade_time[i] = trade_time
    has_market = np.isfinite(market_price)
    order = has_market
    correct = frame["correct"].astype(bool).to_numpy()
    pnl = np.zeros(len(frame), dtype=float)
    pnl[order & correct] = 1.0 - market_price[order & correct]
    pnl[order & ~correct] = -market_price[order & ~correct]
    late_join = sum(
        1
        for t, dt in zip(market_trade_time, decision_time, strict=True)
        if pd.notna(t) and pd.Timestamp(t) > pd.Timestamp(dt)
    )
    market_bid = np.where(order, market_price, 0.0)
    from types import SimpleNamespace

    metrics = factorized.backtest_metrics(
        frame,
        SimpleNamespace(
            bid=market_bid,
            expected_ev=np.zeros(len(frame), dtype=float),
            fill_prob=np.ones(len(frame), dtype=float),
            pnl=pnl,
            filled=order,
            printed_filled=order,
        ),
        len(frame),
    )
    metrics.update(
        {
            "execution_mode": "full_btest_market_order_last_sell_taker_before_t0_plus_2m",
            "policy_gate": "none",
            "intended_order_count": float(len(frame)),
            "market_price_coverage": float(has_market.mean()),
            "missing_market_price_count": float((~has_market).sum()),
            "late_join_count": float(late_join),
            "mean_market_entry": float(np.nanmean(market_price[order])) if order.any() else float("nan"),
            "median_market_entry": float(np.nanmedian(market_price[order])) if order.any() else float("nan"),
        }
    )
    pred = frame[
        [
            c
            for c in [
                "market_t0",
                "condition_id",
                "polymarket_slug",
                "selected_side",
                "p_up",
                "p_side",
                "selected_t_up",
                "selected_t_down",
                "target",
                "correct",
                "threshold_accepted",
            ]
            if c in frame.columns
        ]
    ].copy()
    pred["decision_time_2m"] = decision_time
    pred["market_entry_price"] = market_price
    pred["market_entry_trade_time"] = market_trade_time
    pred["market_order_submitted"] = order
    pred["realized_pnl_market"] = pnl
    return metrics, pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    item = btest_item()
    limit_reference = ensemble.metrics(item, WINNER_PARAMS)
    metrics, pred = market_order_metrics(item, WINNER_PARAMS)
    pred.to_parquet(OUT / "predictions_btest.parquet", index=False)
    dump_json(
        OUT / "metrics_btest.json",
        {
            "source_session": str(SESSION),
            "source_winner": "cycle7_joint_factorized_ensemble",
            "source_winner_params": WINNER_PARAMS,
            "btest_evaluation_count": 1,
            "metrics": metrics,
            "same_policy_limit_order_reference": limit_reference,
            "protocol": {
                "selection_basis": "fixed cycle7 winner from 20260704_two_minute_optimize_500",
                "hazard_checkpoint": str(SOURCE / "folds" / "w6" / "models" / "hazard_survival_cdf.pt"),
                "btest_frame": str(BTEST),
                "decision_time": "market_t0 + 2 minutes",
                "market_entry": "last observed sell-taker trade price for selected side at or before decision_time",
            },
        },
    )
    (OUT / "REPORT.md").write_text(
        "# Two-minute market-order B_test\n\n"
        "Fixed the prior `cycle7_joint_factorized_ensemble` winner and replaced limit-order realized PnL with market-order entry prices.\n\n"
        f"B_test sum_pnl: `{metrics['sum_pnl']}`\n\n"
        f"Same-policy limit-order reference sum_pnl: `{limit_reference['sum_pnl']}`\n\n"
        f"Order count: `{metrics['order_count']}`; market entry coverage on submitted orders: `{metrics['submitted_market_price_coverage']}`; "
        f"late joins: `{metrics['late_join_count']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, allow_nan=True))
    FULL_OUT.mkdir(parents=True, exist_ok=True)
    full_metrics, full_pred = full_market_order_metrics()
    full_pred.to_parquet(FULL_OUT / "predictions_btest.parquet", index=False)
    dump_json(
        FULL_OUT / "metrics_btest.json",
        {
            "source_session": str(SESSION),
            "source_model_basis": "20260704_two_minute_shift B_test frame selected_side",
            "btest_evaluation_count": 1,
            "metrics": full_metrics,
            "protocol": {
                "selection_basis": "no order gate; every frozen B_test row attempts one market order",
                "btest_frame": str(BTEST),
                "decision_time": "market_t0 + 2 minutes",
                "market_entry": "last observed sell-taker trade price for selected side at or before decision_time",
                "missing_market_price": "counted in denominator; no synthetic price is invented",
            },
        },
    )
    (FULL_OUT / "REPORT.md").write_text(
        "# Full B_test market-order no-gate\n\n"
        "No order gate was applied. Every frozen B_test row attempts one market order on `selected_side`; rows without a valid pre-decision market price are counted in the denominator and receive no synthetic fill.\n\n"
        f"B_test sum_pnl: `{full_metrics['sum_pnl']}`\n\n"
        f"Intended order count: `{full_metrics['intended_order_count']}`; executed order count: `{full_metrics['order_count']}`; "
        f"market price coverage: `{full_metrics['market_price_coverage']}`; late joins: `{full_metrics['late_join_count']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"full_no_gate_metrics": full_metrics}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
