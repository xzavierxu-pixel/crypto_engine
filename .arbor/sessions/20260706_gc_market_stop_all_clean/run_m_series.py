#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
EXPERIMENTS = SESSION / "experiments"
TRADE_DIR = ROOT / "price_estimator/data/sell_taker_trades_daily"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


gmod = load_module("g_shared_m", SESSION / "run_g_series.py")
joint = gmod.joint
from train_low_cdf_and_backtest import BacktestResult
FOLDS, TUNE, HOLDOUT = gmod.FOLDS, gmod.TUNE, gmod.HOLDOUT
TAUS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
Q_NAMES = ["raw_tree_blend", "isotonic", "raw"]


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def trade_files(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    dates = set(pd.date_range(start.normalize(), end.normalize(), freq="D").strftime("%Y-%m-%d"))
    return sorted(p for p in TRADE_DIR.glob("date=*.parquet") if p.stem.removeprefix("date=") in dates)


def build_market_prices(name: str, prepared) -> pd.DataFrame:
    cache = SESSION / "cache" / "market_prices" / f"{name}.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    accepted = prepared[1].reset_index(drop=True)
    t0 = pd.to_datetime(accepted["market_t0"], utc=True)
    cutoff = t0 + pd.Timedelta(seconds=68)
    files = trade_files(t0.min(), cutoff.max())
    trades = pd.concat(
        [pd.read_parquet(p, columns=["condition_id", "outcome", "price", "trade_time"]) for p in files],
        ignore_index=True,
    ) if files else pd.DataFrame(columns=["condition_id", "outcome", "price", "trade_time"])
    trades["trade_time"] = pd.to_datetime(trades["trade_time"], utc=True)
    trades["outcome"] = trades["outcome"].astype(str).str.upper()
    lookup = {
        (str(cid), str(side)): grp.sort_values("trade_time").reset_index(drop=True)
        for (cid, side), grp in trades.groupby(["condition_id", "outcome"], sort=False)
    }
    rows = []
    for i, row in enumerate(accepted.itertuples(index=False)):
        limit = cutoff.iloc[i]
        grp = lookup.get((str(row.condition_id), str(row.selected_side).upper()))
        eligible = grp.loc[grp["trade_time"] <= limit] if grp is not None else pd.DataFrame()
        if len(eligible):
            last = eligible.iloc[-1]
            rows.append({"market_price": float(last["price"]), "m_trade_time": last["trade_time"], "has_pre_108_trade": True, "cutoff": limit})
        else:
            rows.append({"market_price": float("nan"), "m_trade_time": pd.NaT, "has_pre_108_trade": False, "cutoff": limit})
    frame = pd.DataFrame(rows)
    bad_time = frame["has_pre_108_trade"] & (pd.to_datetime(frame["m_trade_time"], utc=True) > pd.to_datetime(frame["cutoff"], utc=True))
    if bad_time.any():
        raise RuntimeError("M0 timestamp leakage: m_trade_time after market_t0+68s")
    with cache.open("wb") as fh:
        pickle.dump(frame, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return frame


def anchor(prepared):
    return gmod.run_policy(prepared, 0.0, 0.85, 0.02, prepared[3])


def market_policy(prepared, mframe: pd.DataFrame, q_name: str, tau: float):
    accepted = prepared[1].reset_index(drop=True)
    q = prepared[2][q_name]
    m = mframe["market_price"].to_numpy(float)
    legal = np.isfinite(m)
    ev = q - m
    submitted = legal & (ev > tau)
    bid = np.where(submitted, m, 0.0)
    correct = accepted["correct"].astype(bool).to_numpy()
    filled = submitted.copy()
    pnl = np.where(submitted, np.where(correct, 1.0 - m, -m), 0.0)
    result = BacktestResult(bid=bid, expected_ev=np.where(submitted, ev, 0.0), fill_prob=np.where(submitted, 1.0, np.nan), pnl=pnl, filled=filled, printed_filled=filled)
    metrics = joint.backtest_metrics(accepted, result, len(prepared[0]))
    metrics.update({
        "q_brier": float(brier_score_loss(correct.astype(int), q)), "gc_brier": float("nan"),
        "wrong_submitted_count": int((~correct & submitted).sum()),
        "avg_loser_cost": float(m[~correct & submitted].mean()) if (~correct & submitted).any() else float("nan"),
        "mean_market_price": float(m[submitted].mean()) if submitted.any() else float("nan"),
        "m_coverage": float(legal.mean()), "realized_correct_fill_rate_submitted": 1.0 if (correct & submitted).any() else float("nan"),
        "submitted_fill_calibration_gap": 0.0 if submitted.any() else float("nan"),
    })
    return metrics, result, q, ev


def slice_result(result, mask: np.ndarray):
    return BacktestResult(
        bid=result.bid[mask], expected_ev=result.expected_ev[mask], fill_prob=result.fill_prob[mask],
        pnl=result.pnl[mask], filled=result.filled[mask], printed_filled=result.printed_filled[mask],
    )


def same_universe_metrics(prepared, mframe, market_result, anchor_result):
    accepted = prepared[1].reset_index(drop=True)
    legal = mframe["has_pre_108_trade"].to_numpy(bool)
    market = joint.backtest_metrics(accepted.loc[legal].reset_index(drop=True), slice_result(market_result, legal), int(legal.sum()))
    limit = joint.backtest_metrics(accepted.loc[legal].reset_index(drop=True), slice_result(anchor_result, legal), int(legal.sum()))
    return {"legal_m_count": int(legal.sum()), "market": market, "limit_anchor": limit, "market_minus_limit": float(market["sum_pnl"] - limit["sum_pnl"])}


def qa(mframe: pd.DataFrame, accepted: pd.DataFrame) -> dict[str, Any]:
    m = mframe["market_price"]
    legal = mframe["has_pre_108_trade"].astype(bool)
    p = accepted.reset_index(drop=True)["p_side"].astype(float)
    return {
        "sample_count": len(mframe), "has_pre_108_trade_count": int(legal.sum()), "m_coverage": float(legal.mean()),
        "late_trade_count": int((legal & (pd.to_datetime(mframe["m_trade_time"], utc=True) > pd.to_datetime(mframe["cutoff"], utc=True))).sum()),
        "m_distribution": {"min": float(m[legal].min()), "p10": float(m[legal].quantile(.1)), "median": float(m[legal].median()), "p90": float(m[legal].quantile(.9)), "max": float(m[legal].max()), "mean": float(m[legal].mean())},
        "m_p_side_correlation": float(np.corrcoef(m[legal], p[legal])[0, 1]) if legal.sum() > 1 else float("nan"),
        "missing_policy": "abstain_and_include_in_coverage_denominator",
    }


def artifacts(out: Path, prepared, config: str, bdev: Any, btest: Any, pred: pd.DataFrame, report: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(config, encoding="utf-8")
    dump_json(out / "metrics_bdev.json", bdev)
    dump_json(out / "metrics_btest.json", btest)
    pred.to_parquet(out / "predictions_btest.parquet", index=False)
    cols = joint.load_hazard(gmod.paths_for("w1")[2])[0]
    bad = sorted(c for c in cols if c in gmod.FORBIDDEN_EXACT or c.startswith("future_"))
    dump_json(out / "feature_manifest.json", {"q_feature_count": len(cols), "q_feature_columns": cols, "market_price_timestamp": "market_t0+68s", "market_price_is_policy_input_not_model_feature": True})
    dump_json(out / "leakage_check.json", {"feature_intersection": bad, "m_trade_time_after_cutoff_count": int(btest.get("qa", {}).get("late_trade_count", 0)) if isinstance(btest, dict) else 0, "passed": not bad})
    if bad:
        raise RuntimeError(f"leakage: {bad}")
    (out / "REPORT.md").write_text(report, encoding="utf-8")


def predictions(prepared, mframe, q, ev, result, anchor_result=None) -> pd.DataFrame:
    accepted = prepared[1].reset_index(drop=True)
    frame = pd.DataFrame({
        "sample_id": accepted.index.astype(str), "decision_time": mframe["cutoff"].astype(str),
        "selected_side": accepted["selected_side"].astype(str), "q": q,
        "market_price": mframe["market_price"], "bid": result.bid, "expected_ev": ev,
        "fill_probability": result.fill_prob, "fill_flag": result.filled, "filled": result.filled,
        "correct": accepted["correct"].astype(bool), "realized_pnl": result.pnl,
        "m_trade_time": mframe["m_trade_time"].astype(str), "has_pre_108_trade": mframe["has_pre_108_trade"],
        "market_cutoff": mframe["cutoff"].astype(str),
    })
    if anchor_result is not None:
        frame["anchor_bid"] = anchor_result.bid
        frame["anchor_filled"] = anchor_result.filled
        frame["anchor_realized_pnl"] = anchor_result.pnl
    return frame


def main() -> None:
    data = {}
    for name in FOLDS + ["btest"]:
        prepared = gmod.load_prepared(name)
        base_metrics, base_result, _ = anchor(prepared)
        data[name] = (prepared, build_market_prices(name, prepared), base_metrics, base_result)

    # M0: pure point-in-time data QA, no order policy.
    fold_qa = {f: qa(data[f][1], data[f][0][1]) for f in FOLDS}
    prepared, mf, base_metrics, base_result = data["btest"]
    test_qa = qa(mf, prepared[1])
    q0 = prepared[2]["raw_tree_blend"]
    m = mf["market_price"].to_numpy(float)
    empty = BacktestResult(np.zeros(len(m)), np.zeros(len(m)), np.full(len(m), np.nan), np.zeros(len(m)), np.zeros(len(m), bool), np.zeros(len(m), bool))
    m0_pred = predictions(prepared, mf, q0, q0-m, empty)
    artifacts(EXPERIMENTS/"M0_market_price_qa", prepared, "experiment_id: M0_market_price_qa\ndecision_offset_seconds: 68\nmissing_m_policy: abstain\n", {"folds": fold_qa}, {"qa": test_qa, "btest_evaluation_count": 1, "sum_pnl": None}, m0_pred, f"# M0 market-price QA\n\nB_test m coverage: `{test_qa['m_coverage']:.4f}`; late joined trades: `{test_qa['late_trade_count']}`. Missing m rows abstain and remain in the coverage denominator.\n")

    # M1 selection uses only w1-w4; w5-w6 are read only for the frozen winner.
    rows, detail = [], {}
    for q_name in Q_NAMES:
        for tau in TAUS:
            metrics = {f: market_policy(data[f][0], data[f][1], q_name, tau)[0] for f in TUNE}
            pnl = {f: metrics[f]["sum_pnl"] for f in TUNE}
            tune = np.asarray([pnl[f] for f in TUNE])
            key = f"{q_name}|{tau}"
            rows.append({"q_model": q_name, "tau": tau, **{f"{f}_pnl":pnl[f] for f in TUNE}, "tune_sum":float(tune.sum()), "tune_std":float(tune.std()), "tune_robust":float(tune.sum()-tune.std())})
            detail[key] = metrics
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0].to_dict()
    key = f"{winner['q_model']}|{winner['tau']}"
    for fold in HOLDOUT:
        detail[key][fold] = market_policy(data[fold][0], data[fold][1], str(winner["q_model"]), float(winner["tau"]))[0]
        winner[f"{fold}_pnl"] = detail[key][fold]["sum_pnl"]
    winner["holdout_sum"] = float(sum(winner[f"{f}_pnl"] for f in HOLDOUT))
    winner["holdout_both_positive"] = bool(all(winner[f"{f}_pnl"] > 0 for f in HOLDOUT))
    gate = bool(winner["holdout_sum"] > 0 and winner["holdout_both_positive"])
    m1out = EXPERIMENTS/"M1_market_ev_scan"
    m1out.mkdir(parents=True, exist_ok=True)
    table.to_csv(m1out/"search.csv", index=False)
    bdev = {"winner": winner, "winner_fold_metrics": detail[key], "holdout_gate_passed": gate, "selection_folds": TUNE, "holdout_folds": HOLDOUT}
    if gate:
        test_metrics, test_result, test_q, test_ev = market_policy(prepared, mf, str(winner["q_model"]), float(winner["tau"]))
        test_metrics["btest_evaluation_count"] = 1
        pred = predictions(prepared, mf, test_q, test_ev, test_result, base_result)
        status, score = "evaluated_once", test_metrics["sum_pnl"]
    else:
        test_metrics = {"status":"skipped_holdout_gate_failed", "btest_evaluation_count":0, "sum_pnl":None}
        test_result, test_q, test_ev = empty, q0, q0-m
        pred = predictions(prepared, mf, test_q, test_ev, test_result, base_result)
        status, score = "skipped", None
    artifacts(m1out, prepared, f"experiment_id: M1_market_ev_scan\nq_model: {winner['q_model']}\ntau: {winner['tau']}\ndecision_offset_seconds: 68\n", bdev, {"qa":test_qa, "metrics":test_metrics, "sum_pnl":score, "btest_evaluation_count":test_metrics["btest_evaluation_count"]}, pred, f"# M1 market EV scan\n\nSelected q={winner['q_model']}, tau={winner['tau']} on w1-w4. Holdout gate: `{gate}`. B_test status `{status}`; sum_pnl `{score}`.\n")

    # M2 is a paired accounting view of the single M1 B_test evaluation.
    if gate:
        paired = same_universe_metrics(prepared, mf, test_result, base_result)
        m2_btest = {"qa":test_qa, "paired_same_universe":paired, "shared_evaluation":"M1_market_ev_scan", "btest_evaluation_count":0, "sum_pnl":paired["market"]["sum_pnl"]}
        m2_pred = pred.loc[pred["has_pre_108_trade"]].reset_index(drop=True)
        m2score = paired["market"]["sum_pnl"]
    else:
        paired = None
        m2_btest = {"status":"skipped_m1_gate_failed", "shared_evaluation":"M1_market_ev_scan", "btest_evaluation_count":0, "sum_pnl":None}
        m2_pred = pred.iloc[:0]
        m2score = None
    m2out = EXPERIMENTS/"M2_same_universe_comparison"
    artifacts(m2out, prepared, f"experiment_id: M2_same_universe_comparison\nsource_candidate: M1_market_ev_scan\nq_model: {winner['q_model']}\ntau: {winner['tau']}\n", {"winner":winner, "winner_fold_metrics":detail[key], "holdout_gate_passed":gate}, m2_btest, m2_pred, f"# M2 same-universe comparison\n\nThis reuses the single frozen M1 evaluation. Paired result: `{paired}`. It is not compared directly with 42.43 unless the paired limit anchor is shown.\n")

    ledger_rows = [
        {"experiment_id":"M0_market_price_qa","track":"M","btest_status":"qa_once","btest_sum_pnl":None,"anchor_same_universe":None,"delta":None,"btest_reads":1,"holdout_gate_passed":None},
        {"experiment_id":"M1_market_ev_scan","track":"M","btest_status":status,"btest_sum_pnl":score,"anchor_same_universe":None,"delta":None,"btest_reads":int(gate),"holdout_gate_passed":gate},
        {"experiment_id":"M2_same_universe_comparison","track":"M","btest_status":"shared_with_M1" if gate else "skipped","btest_sum_pnl":m2score,"anchor_same_universe":paired["limit_anchor"]["sum_pnl"] if paired else None,"delta":paired["market_minus_limit"] if paired else None,"btest_reads":0,"holdout_gate_passed":gate},
    ]
    ledger = SESSION/"gc_market_stop_btest_ledger.csv"
    old = pd.read_csv(ledger) if ledger.exists() else pd.DataFrame()
    old = old.loc[~old["experiment_id"].isin([r["experiment_id"] for r in ledger_rows])] if len(old) else old
    pd.concat([old,pd.DataFrame(ledger_rows)],ignore_index=True,sort=False).to_csv(ledger,index=False)
    dump_json(SESSION/"m_series_trigger.json", {"m3_triggered":bool(paired and paired["market_minus_limit"]>0 and gate), "winner":winner, "paired":paired})
    print(json.dumps(ledger_rows, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
