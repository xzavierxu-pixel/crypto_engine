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


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_catboost_two_head_side_policy_search.yaml")
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


def _policy_decisions(
    predictions: pd.DataFrame,
    p_standard: pd.Series,
    p_weighted: pd.Series,
    *,
    continuation_t_up: float,
    continuation_t_down: float,
    reversal_t_up: float,
    reversal_t_down: float,
) -> pd.Series:
    fm_side = predictions["first_minute_side"].astype("object")
    decisions = pd.Series("ABSTAIN", index=predictions.index, dtype="object")
    fm_yes = fm_side == "YES"
    fm_no = fm_side == "NO"

    # Standard model is trusted only when it agrees with the first-minute direction.
    decisions.loc[fm_yes & (p_standard >= float(continuation_t_up))] = "UP"
    decisions.loc[fm_no & (p_standard <= float(continuation_t_down))] = "DOWN"

    # Weighted model can add explicit reversal decisions where continuation did not fire.
    abstain = decisions == "ABSTAIN"
    decisions.loc[abstain & fm_yes & (p_weighted <= float(reversal_t_down))] = "DOWN"
    decisions.loc[abstain & fm_no & (p_weighted >= float(reversal_t_up))] = "UP"
    return decisions


def _evaluate(
    predictions: pd.DataFrame,
    p_standard: pd.Series,
    p_weighted: pd.Series,
    *,
    continuation_t_up: float,
    continuation_t_down: float,
    reversal_t_up: float,
    reversal_t_down: float,
) -> dict[str, Any]:
    decisions = _policy_decisions(
        predictions,
        p_standard,
        p_weighted,
        continuation_t_up=continuation_t_up,
        continuation_t_down=continuation_t_down,
        reversal_t_up=reversal_t_up,
        reversal_t_down=reversal_t_down,
    )
    p_reporting = p_standard.where(decisions.isin(["UP", "DOWN"]), p_standard).clip(0.0, 1.0)
    metrics = compute_decision_metrics(
        predictions["target"],
        p_reporting,
        decisions,
        selected_t_up=float(continuation_t_up),
        selected_t_down=float(continuation_t_down),
    )
    metrics.update(compute_reversal_continuation_metrics(predictions, decisions))
    return {
        "continuation_t_up": float(continuation_t_up),
        "continuation_t_down": float(continuation_t_down),
        "reversal_t_up": float(reversal_t_up),
        "reversal_t_down": float(reversal_t_down),
        **metrics,
    }


def _search(
    predictions: pd.DataFrame,
    p_standard: pd.Series,
    p_weighted: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    search = config["threshold_search"]
    min_coverage = float(config["objective"]["min_coverage"])
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for continuation_t_up in search["continuation_t_up_values"]:
        for continuation_t_down in search["continuation_t_down_values"]:
            for reversal_t_up in search["reversal_t_up_values"]:
                for reversal_t_down in search["reversal_t_down_values"]:
                    row = _evaluate(
                        predictions,
                        p_standard,
                        p_weighted,
                        continuation_t_up=float(continuation_t_up),
                        continuation_t_down=float(continuation_t_down),
                        reversal_t_up=float(reversal_t_up),
                        reversal_t_down=float(reversal_t_down),
                    )
                    records.append(row)
                    if row["coverage"] >= min_coverage and row["utility"] > 0.0:
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
    return pd.DataFrame.from_records(records), {
        **best,
        "constraint_satisfied": bool(eligible),
        "objective": "accepted_sample_accuracy",
        "hard_constraint": "coverage_only",
    }


def _required_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {field: float(metrics[field]) for field in OBJECTIVE_METRIC_FIELDS}


def _window_summary(frame: pd.DataFrame) -> dict[str, str | int]:
    column = "timestamp" if "timestamp" in frame.columns else "market_t0"
    timestamps = pd.to_datetime(frame[column], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamps.min()), "end": str(timestamps.max())}


def _fit_probability(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    *,
    sample_weight: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    model = CatBoostClassifier(**config["model"])
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(train_frame[features], train_frame["target"].astype(int), **fit_kwargs)
    train_p = pd.Series(model.predict_proba(train_frame[features])[:, 1], index=train_frame.index).clip(0.0, 1.0)
    validation_p = pd.Series(model.predict_proba(validation_frame[features])[:, 1], index=validation_frame.index).clip(0.0, 1.0)
    return train_p, validation_p


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_inputs(
        Path(config["baseline_experiment"])
    )
    features = _select_features(all_features, list(config["feature_set"]["patterns"]))
    train_standard_p, validation_standard_p = _fit_probability(train_frame, validation_frame, features, config)
    reversal_mask = _is_reversal(train_predictions)

    results = []
    for reversal_weight in [float(value) for value in config["sample_weight"]["reversal_weights"]]:
        sample_weight = pd.Series(1.0, index=train_frame.index)
        sample_weight.loc[reversal_mask] = reversal_weight
        train_weighted_p, validation_weighted_p = _fit_probability(
            train_frame,
            validation_frame,
            features,
            config,
            sample_weight=sample_weight,
        )
        frontier, validation_metrics = _search(validation_predictions, validation_standard_p, validation_weighted_p, config)
        frontier_path = output_dir / f"reversal_weight_{reversal_weight:g}_frontier.csv"
        frontier.to_csv(frontier_path, index=False)
        train_metrics = _evaluate(
            train_predictions,
            train_standard_p,
            train_weighted_p,
            continuation_t_up=float(validation_metrics["continuation_t_up"]),
            continuation_t_down=float(validation_metrics["continuation_t_down"]),
            reversal_t_up=float(validation_metrics["reversal_t_up"]),
            reversal_t_down=float(validation_metrics["reversal_t_down"]),
        )
        results.append(
            {
                "variant": f"two_head_reversal_weight_{reversal_weight:g}",
                "reversal_sample_weight": reversal_weight,
                "feature_count": len(features),
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
                "continuation_t_up": row["validation_metrics"]["continuation_t_up"],
                "continuation_t_down": row["validation_metrics"]["continuation_t_down"],
                "reversal_t_up": row["validation_metrics"]["reversal_t_up"],
                "reversal_t_down": row["validation_metrics"]["reversal_t_down"],
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
    parser = argparse.ArgumentParser(description="Run two-head continuation/reversal side policy search.")
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
