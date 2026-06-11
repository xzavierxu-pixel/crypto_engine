from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import lightgbm as lgb
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


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_follow_reversal_threshold_search.yaml")
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


def _follow_target(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.Series:
    return ((predictions["first_minute_side"].astype("object") == "YES") == (frame["target"].astype(int) == 1)).astype(int)


def _relative_decisions(predictions: pd.DataFrame, p_follow: pd.Series, *, t_follow: float, t_reversal: float) -> pd.Series:
    fm_side = predictions["first_minute_side"].astype("object")
    decisions = pd.Series("ABSTAIN", index=predictions.index, dtype="object")
    decisions.loc[(fm_side == "YES") & (p_follow >= float(t_follow))] = "UP"
    decisions.loc[(fm_side == "NO") & (p_follow >= float(t_follow))] = "DOWN"
    decisions.loc[(fm_side == "YES") & (p_follow <= float(t_reversal))] = "DOWN"
    decisions.loc[(fm_side == "NO") & (p_follow <= float(t_reversal))] = "UP"
    return decisions


def _final_p_up(predictions: pd.DataFrame, p_follow: pd.Series) -> pd.Series:
    fm_yes = predictions["first_minute_side"].astype("object") == "YES"
    return p_follow.where(fm_yes, 1.0 - p_follow).clip(0.0, 1.0)


def _search(predictions: pd.DataFrame, p_follow: pd.Series, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    min_coverage = float(config["objective"]["min_coverage"])
    p_up = _final_p_up(predictions, p_follow)
    for t_follow in config["threshold_search"]["t_follow_values"]:
        for t_reversal in config["threshold_search"]["t_reversal_values"]:
            if float(t_reversal) >= float(t_follow):
                continue
            decisions = _relative_decisions(
                predictions,
                p_follow,
                t_follow=float(t_follow),
                t_reversal=float(t_reversal),
            )
            metrics = compute_decision_metrics(
                predictions["target"],
                p_up,
                decisions,
                selected_t_up=float(t_follow),
                selected_t_down=float(t_reversal),
            )
            metrics.update(compute_reversal_continuation_metrics(predictions, decisions))
            row = {"t_follow": float(t_follow), "t_reversal": float(t_reversal), **metrics}
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


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_inputs(
        Path(config["baseline_experiment"])
    )
    features = _select_features(all_features, list(config["feature_set"]["patterns"]))
    y_train = _follow_target(train_frame, train_predictions)
    variants = []
    models = {
        "catboost_follow_relative": CatBoostClassifier(**config["catboost_model"]),
        "lightgbm_follow_relative": lgb.LGBMClassifier(**{**config["lightgbm_model"], "objective": "binary", "verbosity": -1}),
    }
    for name, model in models.items():
        model.fit(train_frame[features], y_train)
        train_p_follow = pd.Series(model.predict_proba(train_frame[features])[:, 1], index=train_frame.index).clip(0.0, 1.0)
        validation_p_follow = pd.Series(model.predict_proba(validation_frame[features])[:, 1], index=validation_frame.index).clip(0.0, 1.0)
        frontier, best = _search(validation_predictions, validation_p_follow, config)
        train_decisions = _relative_decisions(
            train_predictions,
            train_p_follow,
            t_follow=float(best["t_follow"]),
            t_reversal=float(best["t_reversal"]),
        )
        train_metrics = compute_decision_metrics(
            train_predictions["target"],
            _final_p_up(train_predictions, train_p_follow),
            train_decisions,
            selected_t_up=float(best["t_follow"]),
            selected_t_down=float(best["t_reversal"]),
        )
        train_metrics.update(compute_reversal_continuation_metrics(train_predictions, train_decisions))
        frontier.to_csv(output_dir / f"{name}_frontier.csv", index=False)
        variants.append(
            {
                "variant": name,
                "feature_count": len(features),
                "train_metrics": train_metrics,
                "validation_metrics": best,
                "accepted": bool(
                    best["coverage"] >= float(config["objective"]["min_coverage"])
                    and best["accepted_sample_accuracy"] >= float(config["objective"]["target_accepted_sample_accuracy"])
                    and best["utility"] > 0.0
                ),
            }
        )
    best_variant = max(
        variants,
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
                "target_met": row["accepted"],
            }
            for row in variants
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
        "best_variant": best_variant["variant"],
        "train_metrics": best_variant["train_metrics"],
        "validation_metrics": best_variant["validation_metrics"],
        "required_train_metrics": _required_metrics(best_variant["train_metrics"]),
        "required_validation_metrics": _required_metrics(best_variant["validation_metrics"]),
        "variant_results": variants,
        "variant_summary_path": str(summary_path),
        "target_met": bool(best_variant["accepted"]),
        "accepted": bool(best_variant["accepted"]),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run follow/reversal relative threshold search.")
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
