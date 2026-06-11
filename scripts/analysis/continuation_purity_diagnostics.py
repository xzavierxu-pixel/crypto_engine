from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.reversal_hybrid import (  # noqa: E402
    OBJECTIVE_METRIC_FIELDS,
    compute_decision_metrics,
    compute_reversal_continuation_metrics,
)


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_continuation_purity_diagnostics.yaml")
LEAKAGE_BLOCKLIST = {
    "target",
    "future_close",
    "abs_return",
    "signed_return",
    "stage1_target",
    "stage2_target",
    "original_btc_direction_target",
}


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    manifest = _read_json(experiment_dir / "artifact_manifest.json")
    return (
        pd.read_parquet(experiment_dir / "development_frame.parquet"),
        pd.read_parquet(experiment_dir / "validation_frame.parquet"),
        pd.read_parquet(experiment_dir / "train_predictions.parquet"),
        pd.read_parquet(experiment_dir / "validation_predictions.parquet"),
        list(manifest["feature_columns"]),
    )


def _select_features(all_features: list[str], patterns: list[str]) -> list[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    safe = [name for name in all_features if name not in LEAKAGE_BLOCKLIST and "target" not in name.lower()]
    selected = [name for name in safe if any(pattern.search(name) for pattern in compiled)]
    if not selected:
        raise ValueError("feature set selected no features")
    return selected


def _first_minute_up(frame: pd.DataFrame) -> pd.Series:
    source = frame["fm_ret"] if "fm_ret" in frame.columns else frame["ret_1"]
    return pd.to_numeric(source, errors="coerce") >= 0.0


def _continuation_decision(predictions: pd.DataFrame) -> pd.Series:
    fm_side = predictions["first_minute_side"].astype("object")
    decisions = pd.Series("ABSTAIN", index=predictions.index, dtype="object")
    decisions.loc[fm_side == "YES"] = "UP"
    decisions.loc[fm_side == "NO"] = "DOWN"
    return decisions


def _required_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {field: float(metrics[field]) for field in OBJECTIVE_METRIC_FIELDS}


def _window_summary(frame: pd.DataFrame) -> dict[str, str | int]:
    column = "timestamp" if "timestamp" in frame.columns else "market_t0"
    timestamps = pd.to_datetime(frame[column], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamps.min()), "end": str(timestamps.max())}


def _best_from_records(records: list[dict[str, Any]], min_coverage: float) -> dict[str, Any]:
    eligible = [row for row in records if row["coverage"] >= min_coverage and row["utility"] > 0.0]
    best = max(
        eligible if eligible else records,
        key=lambda row: (
            row["accepted_sample_accuracy"],
            row["selection_score"],
            row["utility"],
            row["coverage"],
            row["accepted_count"],
        ),
    )
    return {**best, "constraint_satisfied": bool(eligible), "objective": "accepted_sample_accuracy"}


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_inputs(
        Path(config["baseline_experiment"])
    )
    features = _select_features(all_features, list(config["feature_set"]["patterns"]))
    continuation_target = (_first_minute_up(train_frame) == (train_frame["target"].astype(int) == 1)).astype(int)
    continuation_model = lgb.LGBMClassifier(**{**config["continuation_model"], "objective": "binary", "verbosity": -1})
    continuation_model.fit(train_frame[features], continuation_target)
    continuation_score = pd.Series(
        continuation_model.predict_proba(validation_frame[features])[:, 1],
        index=validation_frame.index,
    ).clip(0.0, 1.0)
    continuation_decision = _continuation_decision(validation_predictions)

    catboost_model = CatBoostClassifier(**config["catboost_model"])
    catboost_model.fit(train_frame[features], train_frame["target"].astype(int))
    catboost_p_up = pd.Series(
        catboost_model.predict_proba(validation_frame[features])[:, 1],
        index=validation_frame.index,
    ).clip(0.0, 1.0)

    min_coverage = float(config["objective"]["min_coverage"])
    continuation_records = []
    for cutoff in config["continuation_only_search"]["cutoffs"]:
        decisions = pd.Series("ABSTAIN", index=validation_predictions.index, dtype="object")
        keep = continuation_score >= float(cutoff)
        decisions.loc[keep] = continuation_decision.loc[keep]
        metrics = compute_decision_metrics(
            validation_predictions["target"],
            continuation_score,
            decisions,
            selected_t_up=float(cutoff),
            selected_t_down=0.0,
        )
        metrics.update(compute_reversal_continuation_metrics(validation_predictions, decisions))
        continuation_records.append({"cutoff": float(cutoff), **metrics})

    hybrid_records = []
    for confidence in config["hybrid_fill_search"]["catboost_confidences"]:
        for cutoff in config["hybrid_fill_search"]["continuation_cutoffs"]:
            decisions = pd.Series("ABSTAIN", index=validation_predictions.index, dtype="object")
            high_confidence = (catboost_p_up - 0.5).abs() >= float(confidence)
            decisions.loc[high_confidence & (catboost_p_up >= 0.5)] = "UP"
            decisions.loc[high_confidence & (catboost_p_up < 0.5)] = "DOWN"
            fill = (decisions == "ABSTAIN") & (continuation_score >= float(cutoff))
            decisions.loc[fill] = continuation_decision.loc[fill]
            metrics = compute_decision_metrics(
                validation_predictions["target"],
                catboost_p_up,
                decisions,
                selected_t_up=float(confidence),
                selected_t_down=float(cutoff),
            )
            metrics.update(compute_reversal_continuation_metrics(validation_predictions, decisions))
            hybrid_records.append({"catboost_confidence": float(confidence), "continuation_cutoff": float(cutoff), **metrics})

    continuation_frontier = pd.DataFrame.from_records(continuation_records)
    hybrid_frontier = pd.DataFrame.from_records(hybrid_records)
    continuation_frontier.to_csv(output_dir / "continuation_only_frontier.csv", index=False)
    hybrid_frontier.to_csv(output_dir / "hybrid_fill_frontier.csv", index=False)
    variant_results = [
        {"variant": "continuation_only", "validation_metrics": _best_from_records(continuation_records, min_coverage)},
        {"variant": "hybrid_catboost_confidence_with_continuation_fill", "validation_metrics": _best_from_records(hybrid_records, min_coverage)},
    ]
    best = max(
        variant_results,
        key=lambda row: (
            row["validation_metrics"]["accepted_sample_accuracy"],
            row["validation_metrics"]["selection_score"],
            row["validation_metrics"]["utility"],
            row["validation_metrics"]["coverage"],
        ),
    )
    summary = pd.DataFrame(
        [
            {
                "variant": row["variant"],
                "coverage": row["validation_metrics"]["coverage"],
                "accepted_sample_accuracy": row["validation_metrics"]["accepted_sample_accuracy"],
                "selection_score": row["validation_metrics"]["selection_score"],
                "utility": row["validation_metrics"]["utility"],
                "accepted_count": row["validation_metrics"]["accepted_count"],
                "target_met": bool(
                    row["validation_metrics"]["coverage"] >= min_coverage
                    and row["validation_metrics"]["accepted_sample_accuracy"]
                    >= float(config["objective"]["target_accepted_sample_accuracy"])
                ),
            }
            for row in variant_results
        ]
    ).sort_values(["accepted_sample_accuracy", "selection_score"], ascending=False)
    summary_path = output_dir / "variant_summary.csv"
    summary.to_csv(summary_path, index=False)
    report = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "primary_metric": "validation accepted_sample_accuracy with coverage >= min_coverage",
        "mode": config["mode"],
        "objective": config["objective"],
        "train_window": _window_summary(train_frame),
        "validation_window": _window_summary(validation_frame),
        "best_variant": best["variant"],
        "validation_metrics": best["validation_metrics"],
        "required_validation_metrics": _required_metrics(best["validation_metrics"]),
        "variant_results": variant_results,
        "variant_summary_path": str(summary_path),
        "target_met": bool(
            best["validation_metrics"]["coverage"] >= min_coverage
            and best["validation_metrics"]["accepted_sample_accuracy"]
            >= float(config["objective"]["target_accepted_sample_accuracy"])
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuation purity and hybrid fill diagnostics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run(args.config)
    metrics = report["validation_metrics"]
    print(
        json.dumps(
            {
                "report_path": str(Path(report["variant_summary_path"]).with_name("report.json")),
                "target_met": report["target_met"],
                "best_variant": report["best_variant"],
                "coverage": metrics["coverage"],
                "accepted_sample_accuracy": metrics["accepted_sample_accuracy"],
                "selection_score": metrics["selection_score"],
                "utility": metrics["utility"],
                "accepted_count": metrics["accepted_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
