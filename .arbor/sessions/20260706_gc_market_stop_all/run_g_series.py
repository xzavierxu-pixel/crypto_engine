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
CACHE = SESSION / "cache" / "g_prepared"
SOURCE = ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
ROLLING = ROOT / ".arbor/sessions/20260703_prefinal_rolling/folds"

spec = importlib.util.spec_from_file_location("joint_g_series", SOURCE)
joint = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(joint)

TRAIN = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet"
BTEST = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet"
HAZARD = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]
FLOORS = [0.90, 0.925, 0.95]
MIN_EVS = [0.0, 0.01, 0.02, 0.03, 0.05]
FORBIDDEN_EXACT = set(joint.FORBIDDEN_EXACT) | {
    "chosen_low_trade_time", "time_to_chosen_low_sec",
}


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def paths_for(name: str) -> tuple[Path, Path, Path]:
    if name == "btest":
        return TRAIN, BTEST, HAZARD
    d = ROLLING / name
    return d / "data/train.parquet", d / "data/dev.parquet", d / "models/hazard_survival_cdf.pt"


def load_prepared(name: str):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    prepared = joint.prepare(*paths_for(name), seed=20260703)
    with path.open("wb") as fh:
        pickle.dump(prepared, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return prepared


def empirical_cdf(train_path: Path, accepted_dev: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    train = pd.read_parquet(train_path)
    train = train.loc[train["threshold_accepted"].astype(bool) & train["correct"].astype(bool)].copy()
    global_cdf = np.asarray([(train["chosen_low"].to_numpy(float) <= b).mean() for b in grid])
    train_key = np.floor(np.clip(train["p_side"].to_numpy(float), 0, 0.999999) / 0.025).astype(int)
    dev_key = np.floor(np.clip(accepted_dev["p_side"].to_numpy(float), 0, 0.999999) / 0.025).astype(int)
    out = np.tile(global_cdf, (len(accepted_dev), 1))
    for key in np.unique(dev_key):
        lows = train.loc[train_key == key, "chosen_low"].to_numpy(float)
        if len(lows):
            out[dev_key == key] = np.asarray([(lows <= b).mean() for b in grid])
    return out


def gc_for(name: str, prepared, recalibrated: bool) -> tuple[np.ndarray, dict[str, float]]:
    _, accepted, _, hazard_gc, grid, _, _ = prepared
    correct = accepted["correct"].astype(bool).to_numpy()
    target = (accepted.loc[correct, "chosen_low"].to_numpy(float)[:, None] <= grid[None, :]).astype(float)
    hazard_brier = float(np.mean((hazard_gc[correct] - target) ** 2))
    if not recalibrated:
        return hazard_gc, {"hazard_grid_brier": hazard_brier, "blended_grid_brier": hazard_brier}
    empirical = empirical_cdf(paths_for(name)[0], accepted, grid)
    blended = 0.85 * hazard_gc + 0.15 * empirical
    blended = np.maximum.accumulate(np.clip(blended, 0.0, 1.0), axis=1)
    blend_brier = float(np.mean((blended[correct] - target) ** 2))
    return blended, {"hazard_grid_brier": hazard_brier, "blended_grid_brier": blend_brier}


def run_policy(prepared, shrink: float, floor: float, min_ev: float, gc: np.ndarray):
    dev, accepted, qs, _, grid, _, _ = prepared
    q = (1.0 - shrink) * qs["raw_tree_blend"] + shrink * 0.5
    bid, ev, fill = joint.choose_survival_expected_return_bids(
        q, gc, grid, min_bid=0.01, min_ev=min_ev, min_fill_probability=floor
    )
    result = joint.backtest_with_bid(accepted, bid, ev, fill)
    metrics = joint.backtest_metrics(accepted, result, len(dev))
    correct = accepted["correct"].astype(bool).to_numpy()
    submitted = bid > 0
    correct_submitted = correct & submitted
    metrics.update(
        {
            "q_brier": float(brier_score_loss(correct.astype(int), q)),
            "gc_brier": float(brier_score_loss(result.filled[correct_submitted].astype(int), fill[correct_submitted]))
            if correct_submitted.any() else float("nan"),
            "wrong_submitted_count": int((~correct & submitted).sum()),
            "avg_loser_cost": float(bid[~correct & submitted].mean()) if (~correct & submitted).any() else float("nan"),
        }
    )
    return metrics, result, q


def scan(track: str, prepared: dict[str, Any], recalibrated: bool, shrinks: list[float]):
    gc_map, gc_diag = {}, {}
    for fold in TUNE:
        gc_map[fold], gc_diag[fold] = gc_for(fold, prepared[fold], recalibrated)
    anchor = {
        fold: run_policy(prepared[fold], 0.0, 0.85, 0.02, prepared[fold][3])[0]
        for fold in TUNE
    }
    rows, detailed = [], {}
    for shrink in shrinks:
        for floor in FLOORS:
            for min_ev in MIN_EVS:
                key = f"s={shrink}|f={floor}|e={min_ev}"
                metrics = {
                    fold: run_policy(prepared[fold], shrink, floor, min_ev, gc_map[fold])[0]
                    for fold in TUNE
                }
                pnl = {fold: metrics[fold]["sum_pnl"] for fold in TUNE}
                delta = {fold: pnl[fold] - anchor[fold]["sum_pnl"] for fold in TUNE}
                tune = np.asarray([pnl[f] for f in TUNE])
                row = {
                    "shrink": shrink, "gc_floor": floor, "min_ev": min_ev,
                    **{f"{f}_pnl": pnl[f] for f in TUNE},
                    **{f"{f}_delta": delta[f] for f in TUNE},
                    "tune_sum": float(tune.sum()),
                    "tune_std": float(tune.std()),
                    "tune_robust": float(tune.sum() - tune.std()),
                    "tune_positive_delta_weeks": int(sum(delta[f] > 0 for f in TUNE)),
                }
                rows.append(row)
                detailed[key] = metrics
    table = pd.DataFrame(rows)
    eligible = table.loc[table["tune_positive_delta_weeks"] >= 3].copy()
    selection_passed = not eligible.empty
    selection_pool = eligible if selection_passed else table
    winner = selection_pool.sort_values(["tune_robust", "tune_sum"], ascending=False).iloc[0].to_dict()
    winner_key = f"s={winner['shrink']}|f={winner['gc_floor']}|e={winner['min_ev']}"
    winner_metrics = detailed[winner_key]
    for fold in HOLDOUT:
        gc_map[fold], gc_diag[fold] = gc_for(fold, prepared[fold], recalibrated)
        anchor[fold] = run_policy(prepared[fold], 0.0, 0.85, 0.02, prepared[fold][3])[0]
        winner_metrics[fold] = run_policy(prepared[fold], float(winner["shrink"]), float(winner["gc_floor"]), float(winner["min_ev"]), gc_map[fold])[0]
        winner[f"{fold}_pnl"] = winner_metrics[fold]["sum_pnl"]
        winner[f"{fold}_delta"] = winner_metrics[fold]["sum_pnl"] - anchor[fold]["sum_pnl"]
    winner["holdout_sum"] = float(sum(winner[f"{f}_pnl"] for f in HOLDOUT))
    winner["holdout_both_positive"] = bool(all(winner[f"{f}_pnl"] > 0 for f in HOLDOUT))
    gate = bool(selection_passed and winner["holdout_sum"] > 0 and winner["holdout_both_positive"])
    return table, winner, winner_metrics, anchor, gc_diag, gate, selection_passed


def predictions(accepted: pd.DataFrame, q: np.ndarray, result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": accepted.index.astype(str),
            "decision_time": accepted["timestamp"].astype(str).to_numpy(),
            "selected_side": accepted["selected_side"].astype(str).to_numpy(),
            "q": q,
            "bid": result.bid,
            "expected_ev": result.expected_ev,
            "fill_probability": result.fill_prob,
            "fill_flag": result.filled,
            "filled": result.filled,
            "correct": accepted["correct"].astype(bool).to_numpy(),
            "realized_pnl": result.pnl,
            "printed_filled": result.printed_filled,
        }
    )


def write_experiment(
    experiment_id: str,
    prepared: dict[str, Any],
    recalibrated: bool,
    shrinks: list[float],
    feature_columns: list[str],
) -> dict[str, Any]:
    out = EXPERIMENTS / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    table, winner, folds, anchor, gc_diag, gate, selection_passed = scan(experiment_id, prepared, recalibrated, shrinks)
    table.to_csv(out / "search.csv", index=False)
    bdev = {
        "selection_folds": TUNE, "untouched_gate_folds": HOLDOUT,
        "selection_rule": "maximize sum(w1..w4)-std(w1..w4), requiring >=3 positive deltas versus exact fold anchor",
        "winner": winner, "winner_fold_metrics": folds,
        "anchor_fold_metrics": anchor, "gc_probability_diagnostics": gc_diag,
        "selection_gate_passed": selection_passed, "holdout_gate_passed": gate,
    }
    dump_json(out / "metrics_bdev.json", bdev)

    bad = sorted(c for c in feature_columns if c in FORBIDDEN_EXACT or c.startswith("future_"))
    dump_json(out / "feature_manifest.json", {"feature_count": len(feature_columns), "feature_columns": feature_columns})
    dump_json(out / "leakage_check.json", {"forbidden_columns": sorted(FORBIDDEN_EXACT), "feature_intersection": bad, "passed": not bad})
    if bad:
        raise RuntimeError(f"leakage guard failed: {bad}")
    (out / "config_used.yaml").write_text(
        f"experiment_id: {experiment_id}\nq_model: raw_tree_blend\n"
        f"q_shrink: {winner['shrink']}\ngc_floor: {winner['gc_floor']}\nmin_ev: {winner['min_ev']}\n"
        f"gc_recalibration: {'hazard_85pct_plus_empirical_p_side_bin_15pct' if recalibrated else 'none'}\n"
        "wrong_fill_forced: 1.0\nselection_folds: [w1, w2, w3, w4]\nholdout_gate_folds: [w5, w6]\n",
        encoding="utf-8",
    )

    test_metrics: dict[str, Any]
    if gate:
        test_prepared = load_prepared("btest")
        test_gc, test_gc_diag = gc_for("btest", test_prepared, recalibrated)
        test_metrics, result, q = run_policy(
            test_prepared, float(winner["shrink"]), float(winner["gc_floor"]), float(winner["min_ev"]), test_gc
        )
        test_metrics["anchor_delta"] = float(test_metrics["sum_pnl"] - 42.43)
        test_metrics["btest_evaluation_count"] = 1
        test_metrics["gc_probability_diagnostics"] = test_gc_diag
        predictions(test_prepared[1], q, result).to_parquet(out / "predictions_btest.parquet", index=False)
        status = "evaluated_once"
    else:
        previous_path = out / "metrics_btest.json"
        prior_reads = 0
        if previous_path.exists():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if previous.get("btest_evaluation_count", 0):
                dump_json(out / "protocol_violation_btest_diagnostic.json", previous)
                prior_reads = int(previous["btest_evaluation_count"])
        test_metrics = {
            "status": "skipped_selection_or_holdout_gate_failed",
            "selection_gate_passed": selection_passed,
            "holdout_gate_passed": False,
            "btest_evaluation_count": prior_reads,
            "protocol_violation": "An earlier implementation incorrectly used a fallback candidate and read B_test; its result is quarantined and excluded from ranking." if prior_reads else None,
            "sum_pnl": None, "anchor_delta": None,
        }
        predictions(
            prepared["w1"][1].iloc[:0], np.asarray([], dtype=float),
            type("Empty", (), {"bid": np.array([]), "expected_ev": np.array([]), "fill_prob": np.array([]),
                                "filled": np.array([], dtype=bool), "pnl": np.array([]), "printed_filled": np.array([], dtype=bool)})(),
        ).to_parquet(out / "predictions_btest.parquet", index=False)
        status = "protocol_invalid_quarantined" if prior_reads else "skipped"
    dump_json(out / "metrics_btest.json", test_metrics)
    score_text = "not run" if test_metrics["sum_pnl"] is None else f"{test_metrics['sum_pnl']:.2f}"
    delta_text = "not available" if test_metrics["anchor_delta"] is None else f"{test_metrics['anchor_delta']:+.2f}"
    (out / "REPORT.md").write_text(
        f"# {experiment_id}\n\n"
        f"Selected only on w1-w4: shrink={winner['shrink']}, Gc floor={winner['gc_floor']}, min_ev={winner['min_ev']}. "
        f"The w1-w4 selection gate {'passed' if selection_passed else 'failed'}; the untouched w5-w6 gate {'passed' if gate else 'was not eligible/passed'}.\n\n"
        f"B_test status: `{status}`; sum_pnl: `{score_text}`; delta to frozen 42.43 anchor: `{delta_text}`.\n\n"
        "The accepted universe, direction choices, correct-fill rule, and forced-wrong-fill rule match G0. "
        "Win/loss, submitted-fill calibration, q Brier, and Gc Brier are recorded in the metric files.\n",
        encoding="utf-8",
    )
    return {
        "experiment_id": experiment_id, "track": "G", "btest_status": status,
        "btest_sum_pnl": test_metrics["sum_pnl"], "anchor_same_universe": 42.43,
        "delta": test_metrics["anchor_delta"], "btest_reads": test_metrics["btest_evaluation_count"],
        "holdout_gate_passed": gate,
    }


def update_ledger(rows: list[dict[str, Any]]) -> None:
    path = SESSION / "gc_market_stop_btest_ledger.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty and "experiment_id" in old:
        old = old.loc[~old["experiment_id"].isin([r["experiment_id"] for r in rows])]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True, sort=False).to_csv(path, index=False)


def main() -> None:
    prepared = {fold: load_prepared(fold) for fold in FOLDS}
    feature_columns = joint.load_hazard(paths_for("w1")[2])[0]
    rows = [
        write_experiment("G1_high_fill_scan", prepared, False, [0.0], feature_columns),
        write_experiment("G2_high_fill_conservative_q", prepared, False, [0.0, 0.1, 0.2], feature_columns),
        write_experiment("G3_recalibrated_gc", prepared, True, [0.0], feature_columns),
    ]
    update_ledger(rows)
    print(json.dumps(rows, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
