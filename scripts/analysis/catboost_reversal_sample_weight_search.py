from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

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


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_catboost_reversal_sample_weight_search.yaml")
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


def _is_reversal(predictions: pd.DataFrame) -> pd.Series:
    final_side = pd.Series(np.where(predictions["target"].astype(int) == 1, "YES", "NO"), index=predictions.index)
    fm_side = predictions["first_minute_side"].astype("object")
    return fm_side.isin(["YES", "NO"]) & (fm_side != final_side)


def _threshold_values(search: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    step = float(search["step"])
    up = np.round(np.arange(float(search["t_up_min"]), float(search["t_up_max"]) + step / 2.0, step), 6)
    down = np.round(np.arange(float(search["t_down_min"]), float(search["t_down_max"]) + step / 2.0, step), 6)
    return up, down


def _decisions(p_up: pd.Series, *, t_up: float, t_down: float) -> pd.Series:
    decisions = pd.Series("ABSTAIN", index=p_up.index, dtype="object")
    decisions.loc[p_up >= float(t_up)] = "UP"
    decisions.loc[p_up <= float(t_down)] = "DOWN"
    return decisions


def _search(predictions: pd.DataFrame, p_up: pd.Series, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    min_coverage = float(config["objective"]["min_coverage"])
    for t_up in _threshold_values(config["threshold_search"])[0]:
        for t_down in _threshold_values(config["threshold_search"])[1]:
            if float(t_down) >= float(t_up):
                continue
            decisions = _decisions(p_up, t_up=float(t_up), t_down=float(t_down))
            metrics = compute_decision_metrics(
                predictions["target"],
                p_up,
                decisions,
                selected_t_up=float(t_up),
                selected_t_down=float(t_down),
            )
            metrics.update(compute_reversal_continuation_metrics(predictions, decisions))
            row = {"t_up": float(t_up), "t_down": float(t_down), **metrics}
            records.append(row)
            if metrics["coverage"] >= min_coverage and metrics["utility"] > 0.0:
                eligible.append(row)
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
    best_decisions = _decisions(p_up, t_up=float(best["t_up"]), t_down=float(best["t_down"]))
    return pd.DataFrame.from_records(records), {
        **best,
        "constraint_satisfied": bool(eligible),
        "objective": "accepted_sample_accuracy",
        "hard_constraint": "coverage_only",
    }, best_decisions


def _required_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {field: float(metrics[field]) for field in OBJECTIVE_METRIC_FIELDS}


def _window_summary(frame: pd.DataFrame) -> dict[str, str | int]:
    column = "timestamp" if "timestamp" in frame.columns else "market_t0"
    timestamps = pd.to_datetime(frame[column], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamps.min()), "end": str(timestamps.max())}


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_inputs(
        Path(config["baseline_experiment"])
    )
    features = _select_features(all_features, list(config["feature_set"]["patterns"]))
    reversal_mask = _is_reversal(train_predictions)
    results = []
    for reversal_weight in [float(value) for value in config["sample_weight"]["reversal_weights"]]:
        sample_weight = pd.Series(1.0, index=train_frame.index)
        sample_weight.loc[reversal_mask] = reversal_weight
        model = CatBoostClassifier(**config["model"])
        model.fit(train_frame[features], train_frame["target"].astype(int), sample_weight=sample_weight)
        train_p_up = pd.Series(model.predict_proba(train_frame[features])[:, 1], index=train_frame.index).clip(0.0, 1.0)
        validation_p_up = pd.Series(
            model.predict_proba(validation_frame[features])[:, 1],
            index=validation_frame.index,
        ).clip(0.0, 1.0)
        frontier, validation_metrics, validation_decisions = _search(validation_predictions, validation_p_up, config)
        frontier_path = output_dir / f"reversal_weight_{reversal_weight:g}_frontier.csv"
        frontier.to_csv(frontier_path, index=False)
        train_decisions = _decisions(
            train_p_up,
            t_up=float(validation_metrics["t_up"]),
            t_down=float(validation_metrics["t_down"]),
        )
        train_metrics = compute_decision_metrics(
            train_predictions["target"],
            train_p_up,
            train_decisions,
            selected_t_up=float(validation_metrics["t_up"]),
            selected_t_down=float(validation_metrics["t_down"]),
        )
        train_metrics.update(compute_reversal_continuation_metrics(train_predictions, train_decisions))
        validation_metrics = compute_decision_metrics(
            validation_predictions["target"],
            validation_p_up,
            validation_decisions,
            selected_t_up=float(validation_metrics["t_up"]),
            selected_t_down=float(validation_metrics["t_down"]),
        )
        validation_metrics.update(compute_reversal_continuation_metrics(validation_predictions, validation_decisions))
        validation_metrics.update(
            {
                "t_up": float(validation_metrics["selected_t_up"]),
                "t_down": float(validation_metrics["selected_t_down"]),
                "constraint_satisfied": bool(validation_metrics["coverage"] >= float(config["objective"]["min_coverage"])),
                "objective": "accepted_sample_accuracy",
                "hard_constraint": "coverage_only",
            }
        )
        results.append(
            {
                "variant": f"reversal_weight_{reversal_weight:g}",
                "reversal_sample_weight": reversal_weight,
                "feature_count": len(features),
                "train_reversal_sample_count": float(reversal_mask.sum()),
                "train_continuation_sample_count": float((~reversal_mask).sum()),
                "frontier_path": str(frontier_path),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "target_met": bool(
                    validation_metrics["coverage"] >= float(config["objective"]["min_coverage"])
                    and validation_metrics["accepted_sample_accuracy"]
                    >= float(config["objective"]["target_accepted_sample_accuracy"])
                    and validation_metrics["utility"] > 0.0
                ),
            }
        )

    best_result = max(
        results,
        key=lambda row: (
            row["validation_metrics"]["coverage"] >= float(config["objective"]["min_coverage"]),
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
                "reversal_sample_weight": row["reversal_sample_weight"],
                "coverage": row["validation_metrics"]["coverage"],
                "accepted_sample_accuracy": row["validation_metrics"]["accepted_sample_accuracy"],
                "selection_score": row["validation_metrics"]["selection_score"],
                "utility": row["validation_metrics"]["utility"],
                "accepted_count": row["validation_metrics"]["accepted_count"],
                "selected_t_up": row["validation_metrics"]["selected_t_up"],
                "selected_t_down": row["validation_metrics"]["selected_t_down"],
                "continuation_accepted_accuracy": row["validation_metrics"]["continuation_accepted_accuracy"],
                "reversal_accepted_accuracy": row["validation_metrics"]["reversal_accepted_accuracy"],
                "target_met": row["target_met"],
            }
            for row in results
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
        "threshold_search": config["threshold_search"],
        "train_window": _window_summary(train_frame),
        "validation_window": _window_summary(validation_frame),
        "feature_count": len(features),
        "feature_set_patterns": list(config["feature_set"]["patterns"]),
        "best_variant": best_result["variant"],
        "train_metrics": best_result["train_metrics"],
        "validation_metrics": best_result["validation_metrics"],
        "required_train_metrics": _required_metrics(best_result["train_metrics"]),
        "required_validation_metrics": _required_metrics(best_result["validation_metrics"]),
        "variant_results": results,
        "variant_summary_path": str(summary_path),
        "target_met": bool(best_result["target_met"]),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reversal sample weight CatBoost threshold search.")
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
                "continuation_accepted_accuracy": metrics["continuation_accepted_accuracy"],
                "reversal_accepted_accuracy": metrics["reversal_accepted_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
