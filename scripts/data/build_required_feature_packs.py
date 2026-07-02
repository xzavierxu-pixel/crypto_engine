from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze feature packs consumed by the two L2 baselines")
    parser.add_argument(
        "--classifier-manifest", type=Path,
        default=ROOT / "execution_engine/deploy/baseline/artifact_manifest.json",
    )
    parser.add_argument(
        "--price-report", type=Path,
        default=ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/reports/summary_metrics.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/data_v2/polymarket_l2/required_feature_packs.json",
    )
    args = parser.parse_args()
    classifier = json.loads(args.classifier_manifest.read_text(encoding="utf-8"))
    price = json.loads(args.price_report.read_text(encoding="utf-8"))
    classifier_columns = [str(column) for column in classifier["feature_columns"]]
    price_columns = [str(column) for column in price["feature_columns"]]
    second_level = sorted({column for column in classifier_columns + price_columns if column.startswith(("sl_", "fm_"))})
    shared_core = sorted({column for column in classifier_columns + price_columns if column not in second_level and column not in {
        "selected_side", "p_up", "p_side", "direction_confidence", "p_bin", "p_side_bucket", "market_time_bucket"
    }})
    downstream = sorted(set(price_columns).difference(classifier_columns))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classifier_experiment_id": "20260611_catboost_calendar_coordinate_search",
        "classifier_manifest": str(args.classifier_manifest),
        "classifier_feature_count": len(classifier_columns),
        "price_estimator_experiment_id": "20260619_expected_return_h2_hazard_smooth",
        "price_estimator_report": str(args.price_report),
        "price_estimator_feature_count": len(price_columns),
        "union_feature_count": len(set(classifier_columns).union(price_columns)),
        "packs": {
            "second_level_v2": {
                "consumers": ["classifier", "price_estimator"],
                "columns": second_level,
                "backfill_required_on_l2_window": True,
            },
            "shared_core": {
                "consumers": ["classifier", "price_estimator"],
                "columns": shared_core,
                "backfill_required_on_l2_window": True,
            },
            "classifier_probability_and_calendar": {
                "consumers": ["price_estimator"],
                "columns": downstream,
                "backfill_required_on_l2_window": False,
                "lineage": "calibrated classifier output plus decision-time calendar derivations",
            },
            "polymarket_l2_first_minute_v1": {
                "consumers": ["classifier", "price_estimator"],
                "columns": "resolved from built pm_l2_1m_ schema",
                "backfill_required_on_l2_window": True,
            },
        },
        "excluded_unused_packs": "all packs not represented by actual baseline feature columns",
    }
    _atomic_json(payload, args.output)


if __name__ == "__main__":
    main()
