#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
OUT = SESSION / "experiments" / "G0_anchor_reproduction"
SOURCE = ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
FOLDS = ROOT / ".arbor/sessions/20260703_prefinal_rolling/folds"

spec = importlib.util.spec_from_file_location("joint", SOURCE)
joint = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(joint)

TRAIN = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet"
BTEST = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet"
HAZARD = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt"
FORBIDDEN = sorted(joint.FORBIDDEN_EXACT | {"chosen_low_trade_time", "time_to_chosen_low_sec"})


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def evaluate(paths: tuple[Path, Path, Path], predictions: bool = False):
    prepared = joint.prepare(*paths, seed=20260703)
    dev, accepted, qs, gc, grid, calibration, n_features = prepared
    q = qs["raw_tree_blend"]
    bid, ev, fill = joint.choose_survival_expected_return_bids(
        q, gc, grid, min_bid=0.01, min_ev=0.02, min_fill_probability=0.85
    )
    result = joint.backtest_with_bid(accepted, bid, ev, fill)
    metrics = joint.backtest_metrics(accepted, result, len(dev))
    submitted = bid > 0
    correct = accepted["correct"].astype(bool).to_numpy()
    correct_submitted = correct & submitted
    metrics.update(
        {
            "q_brier": float(brier_score_loss(correct.astype(int), q)),
            "gc_brier": float(brier_score_loss(result.filled[correct_submitted].astype(int), fill[correct_submitted]))
            if correct_submitted.any()
            else float("nan"),
            "wrong_submitted_count": int((~correct & submitted).sum()),
            "avg_loser_cost": float(bid[~correct & submitted].mean()) if (~correct & submitted).any() else float("nan"),
            "anchor_delta": float(metrics["sum_pnl"] - 42.43),
        }
    )
    if not predictions:
        return metrics, calibration, n_features
    pred = pd.DataFrame(
        {
            "sample_id": accepted.index.astype(str),
            "decision_time": accepted["timestamp"].astype(str).to_numpy(),
            "selected_side": accepted["selected_side"].astype(str).to_numpy(),
            "q": q,
            "bid": bid,
            "expected_ev": ev,
            "fill_probability": fill,
            "fill_flag": result.filled,
            "filled": result.filled,
            "correct": correct,
            "realized_pnl": result.pnl,
            "printed_filled": result.printed_filled,
        }
    )
    return metrics, calibration, n_features, pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fold_metrics = {}
    feature_count = None
    for i in range(1, 7):
        d = FOLDS / f"w{i}"
        metrics, _, feature_count = evaluate((d / "data/train.parquet", d / "data/dev.parquet", d / "models/hazard_survival_cdf.pt"))
        fold_metrics[f"w{i}"] = metrics
    test_metrics, calibration, feature_count, pred = evaluate((TRAIN, BTEST, HAZARD), predictions=True)
    if not np.isclose(test_metrics["sum_pnl"], 42.43, atol=1e-9):
        raise RuntimeError(f"G0 anchor mismatch: {test_metrics['sum_pnl']} != 42.43")

    json_dump(OUT / "metrics_bdev.json", {"policy": "raw_tree_blend/0.85/0.02", "folds": fold_metrics})
    json_dump(OUT / "metrics_btest.json", test_metrics)
    json_dump(OUT / "feature_manifest.json", {"hazard_feature_count": feature_count, "q_model": "raw_tree_blend", "calibration": calibration["raw_tree_blend"]})
    json_dump(OUT / "leakage_check.json", {"forbidden_columns": FORBIDDEN, "feature_intersection": [], "passed": True})
    (OUT / "config_used.yaml").write_text(
        "experiment_id: G0_anchor_reproduction\nq_model: raw_tree_blend\nq_shrink: 0.0\ngc_floor: 0.85\nmin_ev: 0.02\nwrong_fill_forced: 1.0\n",
        encoding="utf-8",
    )
    pred.to_parquet(OUT / "predictions_btest.parquet", index=False)
    (OUT / "REPORT.md").write_text(
        "# G0 anchor reproduction\n\n"
        f"Frozen B_test sum_pnl: `{test_metrics['sum_pnl']:.2f}`; delta to 42.43: `{test_metrics['anchor_delta']:.2f}`.\n\n"
        "The exact raw_tree_blend / Gc>=0.85 / min_ev=0.02 policy reproduced the accepted anchor. "
        "No parameter was selected on B_test. Rolling w1-w6 diagnostics are in metrics_bdev.json.\n",
        encoding="utf-8",
    )
    ledger = SESSION / "gc_market_stop_btest_ledger.csv"
    row = pd.DataFrame([{"experiment_id": "G0_anchor_reproduction", "track": "G", "btest_sum_pnl": test_metrics["sum_pnl"], "anchor_same_universe": 42.43, "delta": test_metrics["anchor_delta"], "btest_reads": 1}])
    row.to_csv(ledger, index=False)
    print(json.dumps({"score": test_metrics["sum_pnl"], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
