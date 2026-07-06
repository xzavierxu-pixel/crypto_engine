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


gmod = load_module("g_shared_s", SESSION / "run_g_series.py")
joint = gmod.joint
FOLDS, TUNE, HOLDOUT = gmod.FOLDS, gmod.TUNE, gmod.HOLDOUT


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def anchor(prepared):
    return gmod.run_policy(prepared, 0.0, 0.85, 0.02, prepared[3])


def trade_files(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    dates = set(pd.date_range(start.normalize(), end.normalize(), freq="D").strftime("%Y-%m-%d"))
    return sorted(p for p in TRADE_DIR.glob("date=*.parquet") if p.stem.removeprefix("date=") in dates)


def build_paths(name: str, prepared, base_result) -> pd.DataFrame:
    cache = SESSION / "cache" / "trade_paths" / f"{name}.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    accepted = prepared[1].reset_index(drop=False).rename(columns={"index": "source_index"})
    start = pd.to_datetime(accepted["decision_time"], utc=True).min()
    end = pd.to_datetime(accepted["endDate"], utc=True).max()
    files = trade_files(start, end)
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
    correct = accepted["correct"].astype(bool).to_numpy()
    for i, row in enumerate(accepted.itertuples(index=False)):
        filled = bool(base_result.filled[i])
        decision = pd.Timestamp(row.decision_time)
        end_time = pd.Timestamp(row.endDate)
        grp = lookup.get((str(row.condition_id), str(row.selected_side).upper()))
        if grp is None:
            window = pd.DataFrame(columns=["price", "trade_time"])
        else:
            window = grp.loc[(grp["trade_time"] > decision) & (grp["trade_time"] <= end_time)]
        crossing = window.loc[window["price"].astype(float) <= float(base_result.bid[i]) + 1e-12] if filled else window.iloc[:0]
        observed = bool(len(crossing))
        entry_time = crossing.iloc[0]["trade_time"] if observed else (decision if filled and not correct[i] else pd.NaT)
        if filled and pd.isna(entry_time):
            entry_time = decision
        post = window.loc[window["trade_time"] > entry_time] if filled and not pd.isna(entry_time) else window.iloc[:0]
        rows.append(
            {
                "entry_time": entry_time,
                "entry_observed": observed,
                "forced_wrong_synthetic_entry": bool(filled and not correct[i] and not observed),
                "post_times": post["trade_time"].tolist(),
                "post_prices": post["price"].astype(float).tolist(),
                "post_trade_count": int(len(post)),
                "post_low": float(post["price"].min()) if len(post) else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    with cache.open("wb") as fh:
        pickle.dump(frame, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return frame


def simulate(prepared, base_result, paths: pd.DataFrame, stop_kind: str, c1: float, c2: float | None, tp: float | None = None):
    accepted = prepared[1].reset_index(drop=True)
    bid = base_result.bid
    filled = base_result.filled
    correct = accepted["correct"].astype(bool).to_numpy()
    stop = np.minimum(c1, c2 * bid) if c2 is not None else np.full(len(bid), c1)
    pnl = np.zeros(len(bid), dtype=float)
    stop_flag = np.zeros(len(bid), dtype=bool)
    tp_flag = np.zeros(len(bid), dtype=bool)
    exit_price = np.full(len(bid), np.nan)
    exit_time = np.full(len(bid), np.datetime64("NaT"), dtype="datetime64[ns]")
    for i in np.flatnonzero(filled):
        times = paths.iloc[i]["post_times"]
        prices = paths.iloc[i]["post_prices"]
        stop_hit = next(((t, p) for t, p in zip(times, prices) if p <= stop[i] + 1e-12), None)
        tp_hit = next(((t, p) for t, p in zip(times, prices) if tp is not None and p >= tp - 1e-12), None)
        if tp_hit is not None and (stop_hit is None or tp_hit[0] < stop_hit[0]):
            tp_flag[i] = True
            exit_price[i], exit_time[i] = float(tp), np.datetime64(pd.Timestamp(tp_hit[0]).tz_localize(None))
            pnl[i] = float(tp) - bid[i]
        elif stop_hit is not None:
            stop_flag[i] = True
            if stop_kind == "primary":
                px = stop[i]
            elif stop_kind == "conservative":
                px = float(stop_hit[1])
            elif stop_kind == "pressure":
                px = stop[i] - 0.02
            else:
                raise ValueError(stop_kind)
            exit_price[i], exit_time[i] = px, np.datetime64(pd.Timestamp(stop_hit[0]).tz_localize(None))
            pnl[i] = px - bid[i]
        else:
            pnl[i] = (1.0 - bid[i]) if correct[i] else -bid[i]
    result = type(base_result)(
        bid=bid, expected_ev=base_result.expected_ev, fill_prob=base_result.fill_prob,
        pnl=pnl, filled=filled, printed_filled=base_result.printed_filled,
    )
    metrics = joint.backtest_metrics(accepted, result, len(prepared[0]))
    submitted = bid > 0
    correct_submitted = correct & submitted
    q = prepared[2]["raw_tree_blend"]
    metrics.update(
        {
            "q_brier": float(brier_score_loss(correct.astype(int), q)),
            "gc_brier": float(brier_score_loss(filled[correct_submitted].astype(int), base_result.fill_prob[correct_submitted])) if correct_submitted.any() else float("nan"),
            "wrong_submitted_count": int((~correct & submitted).sum()),
            "avg_loser_cost": float(bid[~correct & submitted].mean()) if (~correct & submitted).any() else float("nan"),
            "stop_trigger_count": int(stop_flag.sum()),
            "stopped_wrong_count": int((stop_flag & ~correct).sum()),
            "false_stopped_winner_count": int((stop_flag & correct).sum()),
            "take_profit_count": int(tp_flag.sum()),
            "mean_stop_pnl": float(pnl[stop_flag].mean()) if stop_flag.any() else float("nan"),
            "entry_observed_count": int(paths["entry_observed"].sum()),
            "forced_wrong_synthetic_entry_count": int(paths["forced_wrong_synthetic_entry"].sum()),
        }
    )
    aux = {"stop": stop, "stop_flag": stop_flag, "tp_flag": tp_flag, "exit_price": exit_price, "exit_time": exit_time, "pnl": pnl}
    return metrics, aux


def prepared_paths(names: list[str]):
    out = {}
    for name in names:
        prepared = gmod.load_prepared(name)
        base_metrics, base_result, q = anchor(prepared)
        out[name] = (prepared, base_metrics, base_result, q, build_paths(name, prepared, base_result))
    return out


def leakage_artifacts(out: Path, prepared) -> None:
    cols = joint.load_hazard(gmod.paths_for("w1")[2])[0]
    bad = sorted(c for c in cols if c in gmod.FORBIDDEN_EXACT or c.startswith("future_"))
    dump_json(out / "feature_manifest.json", {"feature_count": len(cols), "feature_columns": cols, "post_fill_fields_are_backtest_only": True})
    dump_json(out / "leakage_check.json", {"feature_intersection": bad, "passed": not bad, "post_fill_fields_used_for_decision": []})
    if bad:
        raise RuntimeError(f"leakage: {bad}")


def pred_frame(prepared, base_result, q, paths, variants: dict[str, tuple[dict, dict]]) -> pd.DataFrame:
    accepted = prepared[1].reset_index(drop=True)
    out = pd.DataFrame({
        "sample_id": accepted.index.astype(str), "decision_time": accepted["decision_time"].astype(str),
        "selected_side": accepted["selected_side"].astype(str), "q": q, "bid": base_result.bid,
        "expected_ev": base_result.expected_ev, "fill_probability": base_result.fill_prob,
        "fill_flag": base_result.filled, "filled": base_result.filled,
        "correct": accepted["correct"].astype(bool), "entry_time": paths["entry_time"].astype(str),
        "entry_observed": paths["entry_observed"], "forced_wrong_synthetic_entry": paths["forced_wrong_synthetic_entry"],
    })
    for name, (_, aux) in variants.items():
        out[f"stop_price_{name}"] = aux["stop"]
        out[f"stop_triggered_{name}"] = aux["stop_flag"]
        out[f"take_profit_triggered_{name}"] = aux["tp_flag"]
        out[f"exit_price_{name}"] = aux["exit_price"]
        out[f"exit_time_{name}"] = pd.Series(aux["exit_time"]).astype(str)
        out[f"realized_pnl_{name}"] = aux["pnl"]
    out["realized_pnl"] = variants["primary"][1]["pnl"]
    return out


def write_common(out: Path, config: str, bdev: dict, btest: dict, pred: pd.DataFrame, prepared) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(config, encoding="utf-8")
    dump_json(out / "metrics_bdev.json", bdev)
    dump_json(out / "metrics_btest.json", btest)
    pred.to_parquet(out / "predictions_btest.parquet", index=False)
    leakage_artifacts(out, prepared)


def s0(data: dict[str, Any]) -> tuple[dict, dict[str, float]]:
    fold_metrics, anchor_metrics = {}, {}
    for name in FOLDS:
        prepared, base, result, _, paths = data[name]
        fold_metrics[name] = simulate(prepared, result, paths, "primary", 0.30, 0.50)[0]
        anchor_metrics[name] = base
    prepared, base, result, q, paths = data["btest"]
    variants = {k: simulate(prepared, result, paths, k, 0.30, 0.50) for k in ("primary", "conservative", "pressure")}
    btest = {"anchor": base, "variants": {k: v[0] for k, v in variants.items()}, "btest_evaluation_count": 1}
    out = EXPERIMENTS / "S0_fixed_stop_preflight"
    write_common(out, "experiment_id: S0_fixed_stop_preflight\nstop: min(0.30, 0.50 * bid)\nwrong_fill_forced: 1.0\n", {"folds": fold_metrics, "anchor_folds": anchor_metrics}, btest, pred_frame(prepared, result, q, paths, variants), prepared)
    primary = variants["primary"][0]
    (out / "REPORT.md").write_text(
        f"# S0 fixed-stop preflight\n\nPrimary B_test sum_pnl `{primary['sum_pnl']:.2f}` versus identical-order anchor `{base['sum_pnl']:.2f}`. "
        f"Stopped wrong orders: `{primary['stopped_wrong_count']}`; false-stopped winners: `{primary['false_stopped_winner_count']}`. "
        "Conservative and 0.02 pressure execution results are recorded beside the primary result. Forced wrong orders without an observed entry cross use decision_time as a synthetic entry solely to preserve the required forced-fill universe.\n",
        encoding="utf-8",
    )
    return {"experiment_id": "S0_fixed_stop_preflight", "track": "S", "btest_status": "evaluated_once", "btest_sum_pnl": primary["sum_pnl"], "anchor_same_universe": base["sum_pnl"], "delta": primary["sum_pnl"] - base["sum_pnl"], "btest_reads": 1, "holdout_gate_passed": None}, {"c1": 0.30, "c2": 0.50}


def scan_stop(data, candidates: list[dict[str, float | None]], tp: float | None = None):
    rows, details = [], {}
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        fold_metrics = {}
        for fold in TUNE:
            prepared, base, result, _, paths = data[fold]
            fold_metrics[fold] = simulate(prepared, result, paths, "primary", float(candidate["c1"]), candidate["c2"], tp)[0]
        pnl = {f: fold_metrics[f]["sum_pnl"] for f in TUNE}
        delta = {f: pnl[f] - data[f][1]["sum_pnl"] for f in TUNE}
        tune = np.asarray([pnl[f] for f in TUNE])
        rows.append({**candidate, "tp": tp, **{f"{f}_pnl": pnl[f] for f in TUNE}, **{f"{f}_delta": delta[f] for f in TUNE}, "tune_sum": float(tune.sum()), "tune_std": float(tune.std()), "tune_robust": float(tune.sum()-tune.std()), "tune_positive_delta_weeks": int(sum(delta[f] > 0 for f in TUNE))})
        details[key] = fold_metrics
    table = pd.DataFrame(rows)
    eligible = table.loc[table["tune_positive_delta_weeks"] >= 3]
    pool = eligible if len(eligible) else table
    winner = pool.sort_values(["tune_robust", "tune_sum"], ascending=False).iloc[0].to_dict()
    if pd.isna(winner["c2"]):
        winner["c2"] = None
    key = json.dumps({"c1": winner["c1"], "c2": winner["c2"]}, sort_keys=True)
    winner_metrics = details[key]
    for fold in HOLDOUT:
        prepared, _, result, _, paths = data[fold]
        winner_metrics[fold] = simulate(prepared, result, paths, "primary", float(winner["c1"]), winner["c2"], tp)[0]
        winner[f"{fold}_pnl"] = winner_metrics[fold]["sum_pnl"]
        winner[f"{fold}_delta"] = winner_metrics[fold]["sum_pnl"] - data[fold][1]["sum_pnl"]
    winner["holdout_sum"] = float(sum(winner[f"{f}_pnl"] for f in HOLDOUT))
    winner["holdout_both_positive"] = bool(all(winner[f"{f}_pnl"] > 0 for f in HOLDOUT))
    gate = bool(len(eligible) and winner["holdout_sum"] > 0 and winner["holdout_both_positive"])
    return table, winner, winner_metrics, gate, bool(len(eligible))


def s1(data) -> tuple[dict, dict[str, Any], bool]:
    candidates = [{"c1": c1, "c2": c2} for c1 in (0.20, 0.30, 0.40) for c2 in (0.40, 0.50, 0.60)] + [{"c1": s, "c2": None} for s in (0.20, 0.30)]
    table, winner, folds, gate, selection = scan_stop(data, candidates)
    out = EXPERIMENTS / "S1_stop_grid"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "search.csv", index=False)
    bdev = {"winner": winner, "winner_fold_metrics": folds, "selection_gate_passed": selection, "holdout_gate_passed": gate}
    if gate:
        prepared, base, result, q, paths = data["btest"]
        variants = {k: simulate(prepared, result, paths, k, float(winner["c1"]), winner["c2"]) for k in ("primary", "conservative", "pressure")}
        btest = {"anchor": base, "variants": {k: v[0] for k, v in variants.items()}, "btest_evaluation_count": 1}
        pred = pred_frame(prepared, result, q, paths, variants)
        score = variants["primary"][0]["sum_pnl"]
        status = "evaluated_once"
    else:
        prepared = data["w1"][0]
        btest = {"status": "skipped_selection_or_holdout_gate_failed", "btest_evaluation_count": 0, "sum_pnl": None}
        pred = pd.DataFrame(columns=["sample_id", "decision_time", "selected_side", "q", "bid", "expected_ev", "fill_probability", "filled", "correct", "realized_pnl"])
        score, status = None, "skipped"
    config = f"experiment_id: S1_stop_grid\nstop_c1: {winner['c1']}\nstop_c2: {winner['c2']}\nwrong_fill_forced: 1.0\n"
    write_common(out, config, bdev, btest, pred, prepared)
    (out / "REPORT.md").write_text(f"# S1 stop grid\n\nSelected c1={winner['c1']}, c2={winner['c2']} on w1-w4. Holdout gate: `{gate}`. B_test status: `{status}`; primary sum_pnl: `{score}`. Three execution-price variants are reported when evaluated.\n", encoding="utf-8")
    anchor_score = data["btest"][1]["sum_pnl"]
    return {"experiment_id": "S1_stop_grid", "track": "S", "btest_status": status, "btest_sum_pnl": score, "anchor_same_universe": anchor_score, "delta": None if score is None else score-anchor_score, "btest_reads": int(gate), "holdout_gate_passed": gate}, winner, gate


def s2(data, stop_winner: dict[str, Any], prerequisite: bool) -> dict:
    out = EXPERIMENTS / "S2_take_profit_overlay"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tp in (0.90, 0.95):
        folds = {}
        for fold in TUNE:
            prepared, _, result, _, paths = data[fold]
            folds[fold] = simulate(prepared, result, paths, "primary", float(stop_winner["c1"]), stop_winner["c2"], tp)[0]
        pnl = np.asarray([folds[f]["sum_pnl"] for f in TUNE])
        delta = [folds[f]["sum_pnl"] - data[f][1]["sum_pnl"] for f in TUNE]
        rows.append({"tp":tp,"tune_sum":float(pnl.sum()),"tune_std":float(pnl.std()),"tune_robust":float(pnl.sum()-pnl.std()),"tune_positive_delta_weeks":int(sum(v>0 for v in delta)),"folds":folds})
    chosen = max(rows, key=lambda r: r["tune_robust"])
    holdout = {}
    if prerequisite:
        for fold in HOLDOUT:
            prepared, _, result, _, paths = data[fold]
            holdout[fold] = simulate(prepared, result, paths, "primary", float(stop_winner["c1"]), stop_winner["c2"], float(chosen["tp"]))[0]
        chosen["folds"].update(holdout)
    gate = bool(prerequisite and chosen["tune_positive_delta_weeks"] >= 3 and all(holdout[f]["sum_pnl"] > 0 for f in HOLDOUT))
    bdev = {"candidates": rows, "winner": chosen, "holdout_gate_passed": gate}
    if gate:
        prepared, base, result, q, paths = data["btest"]
        variants = {k: simulate(prepared, result, paths, k, float(stop_winner["c1"]), stop_winner["c2"], float(chosen["tp"])) for k in ("primary", "conservative", "pressure")}
        btest = {"anchor": base, "variants": {k: v[0] for k, v in variants.items()}, "btest_evaluation_count": 1}
        pred = pred_frame(prepared, result, q, paths, variants)
        score, status = variants["primary"][0]["sum_pnl"], "evaluated_once"
    else:
        prepared = data["w1"][0]
        btest = {"status": "skipped_prerequisite_or_gate_failed", "btest_evaluation_count": 0, "sum_pnl": None}
        pred = pd.DataFrame(columns=["sample_id", "decision_time", "selected_side", "q", "bid", "expected_ev", "fill_probability", "filled", "correct", "realized_pnl"])
        score, status = None, "skipped"
    config = f"experiment_id: S2_take_profit_overlay\nstop_c1: {stop_winner['c1']}\nstop_c2: {stop_winner['c2']}\ntake_profit: {chosen['tp']}\n"
    write_common(out, config, bdev, btest, pred, prepared)
    (out / "REPORT.md").write_text(f"# S2 take-profit overlay\n\nFrozen S1 stop with tp={chosen['tp']}. Gate: `{gate}`. B_test status: `{status}`; primary sum_pnl: `{score}`.\n", encoding="utf-8")
    anchor_score = data["btest"][1]["sum_pnl"]
    return {"experiment_id": "S2_take_profit_overlay", "track": "S", "btest_status": status, "btest_sum_pnl": score, "anchor_same_universe": anchor_score, "delta": None if score is None else score-anchor_score, "btest_reads": int(gate), "holdout_gate_passed": gate}


def update_ledger(rows: list[dict[str, Any]]) -> None:
    path = SESSION / "gc_market_stop_btest_ledger.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    old = old.loc[~old["experiment_id"].isin([r["experiment_id"] for r in rows])] if len(old) else old
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True, sort=False).to_csv(path, index=False)


def main() -> None:
    data = prepared_paths(FOLDS + ["btest"])
    row0, _ = s0(data)
    row1, winner, gate = s1(data)
    row2 = s2(data, winner, gate)
    update_ledger([row0, row1, row2])
    print(json.dumps([row0, row1, row2], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
