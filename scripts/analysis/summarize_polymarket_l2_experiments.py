from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "price_estimator/expected_return/experiments"


def main() -> None:
    names = [
        "20260702_l2_P0_A0_common_window",
        "20260702_l2_A2_direction_l2",
        "20260702_l2_A3_price_l2",
        "20260702_l2_A4_full",
        "20260702_l2_A4_market_direction_full",
        "20260702_l2_A5_no_book_depth",
        "20260702_l2_A6_no_order_flow",
        "20260702_l2_A7_no_trade_dynamics",
        "20260702_l2_A8_no_cross_side",
        "20260702_l2_A4_xgb_q_hazard",
        "20260702_l2_market_direction_xgb_q_hazard",
    ]
    rows = []
    for name in names:
        report_path = EXPERIMENT_ROOT / name / "reports/summary_metrics.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report["validation_metrics"]
        rows.append(
            {
                "experiment_id": name,
                "report_path": str(report_path.relative_to(ROOT)),
                "sum_pnl": metrics["sum_pnl"],
                "sample_count": metrics["sample_count"],
                "coverage": metrics["coverage"],
                "accepted_sample_accuracy": metrics["accepted_sample_accuracy"],
                "selection_score": metrics["selection_score"],
                "utility": metrics["utility"],
                "accepted_count": metrics["accepted_count"],
                "up_prediction_count": metrics["up_prediction_count"],
                "down_prediction_count": metrics["down_prediction_count"],
                "order_count": metrics["order_count"],
                "trade_count": metrics["trade_count"],
                "fill_rate": metrics["fill_rate"],
                "mean_accepted_pnl": metrics["mean_accepted_pnl"],
                "coverage_constraint_satisfied": metrics["coverage"] >= 0.70,
            }
        )
    baseline = next(row for row in rows if row["experiment_id"] == names[0])
    best = max(rows, key=lambda row: row["sum_pnl"])
    payload = {
        "primary_metric": "validation sum_pnl",
        "target_validation_sum_pnl": 1000.0,
        "target_satisfied": best["sum_pnl"] > 1000.0,
        "common_window_baseline": baseline,
        "best_experiment": best,
        "best_delta_vs_common_window": best["sum_pnl"] - baseline["sum_pnl"],
        "direction_coverage_constraint": 0.70,
        "all_reported_experiments_satisfy_direction_coverage": all(
            row["coverage_constraint_satisfied"] for row in rows
        ),
        "validation_note": "Threshold-tuned optimistic validation; validation did not fit model, probability calibrator, imputer, selector, or price policy.",
        "promotion_decision": "do_not_promote" if best["sum_pnl"] <= baseline["sum_pnl"] or best["sum_pnl"] <= 0 else "diagnostic_only_positive_not_target",
        "experiments": rows,
    }
    output_dir = ROOT / "artifacts/data_v2/polymarket_l2/experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "20260702_l2_experiment_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Polymarket L2 common-window experiment summary",
        "",
        f"Target: validation `sum_pnl > 1000`. Achieved: **{payload['target_satisfied']}**.",
        "",
        "| Experiment | sum_pnl | coverage | accepted accuracy | orders | trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment_id']} | {row['sum_pnl']:.2f} | {row['coverage']:.4f} | "
            f"{row['accepted_sample_accuracy']:.4f} | {int(row['order_count'])} | {int(row['trade_count'])} |"
        )
    lines.extend(
        [
            "",
            f"Common-window baseline sum_pnl: {baseline['sum_pnl']:.2f}.",
            f"Best observed sum_pnl: {best['sum_pnl']:.2f} ({best['experiment_id']}).",
            "No model is promoted because the requested target was not achieved and the best result remains negative.",
        ]
    )
    (output_dir / "20260702_l2_experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
