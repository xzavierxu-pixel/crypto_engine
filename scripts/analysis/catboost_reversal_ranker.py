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

from src.model.reversal_hybrid import compute_decision_metrics, compute_reversal_continuation_metrics  # noqa: E402


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_catboost_reversal_ranker.yaml")
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


def _target(frame: pd.DataFrame, predictions: pd.DataFrame, mode: str) -> pd.Series:
    final_up = frame["target"].astype(int) == 1
    if mode == "final_direction":
        return final_up.astype(int)
    if mode == "follow_relative":
        return ((predictions["first_minute_side"].astype("object") == "YES") == final_up).astype(int)
    raise ValueError(f"unsupported model target: {mode}")


def _final_p_up(probability: pd.Series, predictions: pd.DataFrame, mode: str) -> pd.Series:
    p = probability.astype("float64").clip(0.0, 1.0)
    if mode == "final_direction":
        return p
    if mode == "follow_relative":
        fm_yes = predictions["first_minute_side"].astype("object") == "YES"
        return p.where(fm_yes, 1.0 - p).clip(0.0, 1.0)
    raise ValueError(f"unsupported model target: {mode}")


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


def _search(frame: pd.DataFrame, p_up: pd.Series, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    min_coverage = float(config["objective"]["min_coverage"])
    for t_up in _threshold_values(config["threshold_search"])[0]:
        for t_down in _threshold_values(config["threshold_search"])[1]:
            if float(t_down) >= float(t_up):
                continue
            decisions = _decisions(p_up, t_up=float(t_up), t_down=float(t_down))
            metrics = compute_decision_metrics(
                frame["target"],
                p_up,
                decisions,
                selected_t_up=float(t_up),
                selected_t_down=float(t_down),
            )
            metrics.update(compute_reversal_continuation_metrics(frame, decisions))
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
    return pd.DataFrame.from_records(records), {
        **best,
        "constraint_satisfied": bool(eligible),
        "objective": "accepted_sample_accuracy",
        "hard_constraint": "coverage_only",
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_inputs(
        Path(config["baseline_experiment"])
    )
    features = _select_features(all_features, list(config["feature_set"]["patterns"]))
    results = []
    for variant in config["variants"]:
        mode = str(variant["model_target"])
        model = CatBoostClassifier(**config["model"])
        model.fit(train_frame[features], _target(train_frame, train_predictions, mode))
        p_up = _final_p_up(
            pd.Series(model.predict_proba(validation_frame[features])[:, 1], index=validation_frame.index),
            validation_predictions,
            mode,
        )
        frontier, best = _search(validation_predictions, p_up, config)
        frontier.to_csv(output_dir / f"{variant['name']}_frontier.csv", index=False)
        results.append(
            {
                "variant": variant["name"],
                "model_target": mode,
                "feature_count": len(features),
                "validation_metrics": best,
                "accepted": bool(
                    best["coverage"] >= float(config["objective"]["min_coverage"])
                    and best["accepted_sample_accuracy"] >= float(config["objective"]["target_accepted_sample_accuracy"])
                    and best["utility"] > 0.0
                ),
            }
        )
    best_result = max(
        results,
        key=lambda row: (
            row["validation_metrics"]["coverage"] >= float(config["objective"]["min_coverage"]),
            row["validation_metrics"]["accepted_sample_accuracy"],
            row["validation_metrics"]["selection_score"],
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
        "best_variant": best_result["variant"],
        "validation_metrics": best_result["validation_metrics"],
        "variant_results": results,
        "variant_summary_path": str(summary_path),
        "target_met": bool(best_result["accepted"]),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated CatBoost ranker variants.")
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
