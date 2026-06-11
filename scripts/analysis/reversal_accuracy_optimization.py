from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.reversal_hybrid import (  # noqa: E402
    OBJECTIVE_METRIC_FIELDS,
    compute_decision_metrics,
    compute_reversal_continuation_metrics,
    p_follow_from_direction_probability,
)


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_reversal_accuracy_optimization.yaml")
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
    feature_columns = list(manifest["feature_columns"])
    train_frame = pd.read_parquet(experiment_dir / "development_frame.parquet")
    validation_frame = pd.read_parquet(experiment_dir / "validation_frame.parquet")
    train_predictions = pd.read_parquet(experiment_dir / "train_predictions.parquet")
    validation_predictions = pd.read_parquet(experiment_dir / "validation_predictions.parquet")
    return train_frame, validation_frame, train_predictions, validation_predictions, feature_columns


def _load_predictions(experiment_dir: Path | None, split: str) -> pd.DataFrame | None:
    if experiment_dir is None:
        return None
    path = experiment_dir / f"{split}_predictions.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def _select_features(all_features: list[str], feature_set: dict[str, Any]) -> list[str]:
    safe = [name for name in all_features if name not in LEAKAGE_BLOCKLIST and "target" not in name.lower()]
    patterns = _compile_patterns(list(feature_set.get("patterns", [])))
    mode = str(feature_set.get("mode", "keep_regex"))
    if mode != "keep_regex":
        raise ValueError(f"unsupported feature_set mode: {mode}")
    selected = [name for name in safe if any(pattern.search(name) for pattern in patterns)]
    if not selected:
        raise ValueError("feature_set selected no features.")
    return selected


def _first_minute_yes(predictions: pd.DataFrame) -> pd.Series:
    return predictions["first_minute_side"].astype("object") == "YES"


def _training_target(frame: pd.DataFrame, predictions: pd.DataFrame, target_mode: str) -> pd.Series:
    final_yes = frame["target"].astype(int) == 1
    if target_mode == "final_direction":
        return final_yes.astype(int)
    if target_mode == "follow_relative":
        return (_first_minute_yes(predictions) == final_yes).astype(int)
    raise ValueError(f"unsupported model_target: {target_mode}")


def _final_p_up(probability: pd.Series, predictions: pd.DataFrame, target_mode: str) -> pd.Series:
    p = probability.astype("float64").clip(0.0, 1.0)
    if target_mode == "final_direction":
        return p
    if target_mode == "follow_relative":
        return p.where(_first_minute_yes(predictions), 1.0 - p).astype("float64").clip(0.0, 1.0)
    raise ValueError(f"unsupported model_target: {target_mode}")


def _augment_features(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    features: list[str],
    follow_predictions: pd.DataFrame | None,
    *,
    add_meta_features: bool,
) -> pd.DataFrame:
    output = frame.loc[:, features].copy()
    if not add_meta_features:
        return output
    p_base = pd.to_numeric(predictions["p_up"], errors="coerce").astype("float64").clip(0.0, 1.0)
    output["meta_base_p_up"] = p_base
    output["meta_base_confidence"] = (p_base - 0.5).abs()
    output["meta_base_accepted"] = (predictions["decision"] != "ABSTAIN").astype("float64")
    output["meta_base_side_matches_first_minute"] = (
        predictions["decision"].replace({"UP": "YES", "DOWN": "NO", "ABSTAIN": None})
        == predictions["first_minute_side"]
    ).astype("float64")
    output["meta_first_minute_yes"] = _first_minute_yes(predictions).astype("float64")
    if "first_minute_return" in predictions:
        first_minute_return = pd.to_numeric(predictions["first_minute_return"], errors="coerce").astype("float64")
        output["meta_first_minute_return"] = first_minute_return
        output["meta_first_minute_abs_return"] = first_minute_return.abs()
    if follow_predictions is not None:
        follow = follow_predictions.reindex(predictions.index)
        p_follow = p_follow_from_direction_probability(follow)
        output["meta_follow_p_follow"] = p_follow
        output["meta_follow_p_reversal"] = 1.0 - p_follow
    return output


def _fit_model(train_x: pd.DataFrame, y_train: pd.Series, config: dict[str, Any]) -> lgb.LGBMClassifier:
    params = dict(config["model_defaults"])
    params.pop("type", None)
    params.setdefault("objective", "binary")
    params.setdefault("verbosity", -1)
    positives = int(y_train.sum())
    negatives = int((y_train == 0).sum())
    if positives:
        params.setdefault("scale_pos_weight", negatives / positives)
    model = lgb.LGBMClassifier(**params)
    model.fit(train_x, y_train.astype(int))
    return model


def _threshold_values(search: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    step = float(search["step"])
    t_up = np.round(np.arange(float(search["t_up_min"]), float(search["t_up_max"]) + step / 2.0, step), 6)
    t_down = np.round(np.arange(float(search["t_down_min"]), float(search["t_down_max"]) + step / 2.0, step), 6)
    return t_up, t_down


def _decisions_from_probability(p_up: pd.Series, *, t_up: float, t_down: float) -> pd.Series:
    decisions = pd.Series("ABSTAIN", index=p_up.index, dtype="object")
    decisions.loc[p_up >= float(t_up)] = "UP"
    decisions.loc[p_up <= float(t_down)] = "DOWN"
    return decisions


def _search_thresholds(
    predictions: pd.DataFrame,
    p_up: pd.Series,
    config: dict[str, Any],
    *,
    blend_weight: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    min_coverage = float(config["objective"]["min_coverage"])
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    t_up_values, t_down_values = _threshold_values(config["threshold_search"])
    for t_up in t_up_values:
        for t_down in t_down_values:
            if float(t_down) >= float(t_up):
                continue
            decisions = _decisions_from_probability(p_up, t_up=float(t_up), t_down=float(t_down))
            metrics = compute_decision_metrics(
                predictions["target"],
                p_up,
                decisions,
                selected_t_up=float(t_up),
                selected_t_down=float(t_down),
            )
            metrics.update(compute_reversal_continuation_metrics(predictions, decisions))
            row = {"blend_weight": float(blend_weight), "t_up": float(t_up), "t_down": float(t_down), **metrics}
            records.append(row)
            if metrics["coverage"] >= min_coverage and metrics["utility"] > 0.0:
                eligible.append(row)
    if not records:
        raise ValueError("threshold search produced no candidates.")
    pool = eligible if eligible else records
    best = max(
        pool,
        key=lambda row: (
            row["accepted_sample_accuracy"],
            row["selection_score"],
            row["utility"],
            row["coverage"],
            row["accepted_count"],
            -abs(row["t_up"] - 0.5) - abs(row["t_down"] - 0.5),
        ),
    )
    best_decisions = _decisions_from_probability(p_up, t_up=float(best["t_up"]), t_down=float(best["t_down"]))
    return pd.DataFrame.from_records(records), {
        **best,
        "constraint_satisfied": bool(eligible),
        "objective": "accepted_sample_accuracy",
        "hard_constraint": "coverage_only",
        "fallback_reason": None if eligible else "no candidate satisfied coverage and positive utility",
    }, best_decisions


def _window_summary(frame: pd.DataFrame) -> dict[str, str | int]:
    column = "timestamp" if "timestamp" in frame.columns else "market_t0"
    timestamps = pd.to_datetime(frame[column], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamps.min()), "end": str(timestamps.max())}


def _required_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {field: float(metrics[field]) for field in OBJECTIVE_METRIC_FIELDS}


def _evaluate_variant(
    variant: dict[str, Any],
    config: dict[str, Any],
    all_features: list[str],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    train_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    follow_train_predictions: pd.DataFrame | None,
    follow_validation_predictions: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_set = config["feature_sets"][variant["feature_set"]]
    features = _select_features(all_features, feature_set)
    target_mode = str(variant.get("model_target", "final_direction"))
    add_meta = bool(variant.get("add_meta_features", False))
    train_x = _augment_features(train_frame, train_predictions, features, follow_train_predictions, add_meta_features=add_meta)
    validation_x = _augment_features(
        validation_frame,
        validation_predictions,
        features,
        follow_validation_predictions,
        add_meta_features=add_meta,
    )
    model = _fit_model(train_x, _training_target(train_frame, train_predictions, target_mode), config)
    train_raw = pd.Series(model.predict_proba(train_x)[:, 1], index=train_frame.index)
    validation_raw = pd.Series(model.predict_proba(validation_x)[:, 1], index=validation_frame.index)
    train_model_p_up = _final_p_up(train_raw, train_predictions, target_mode)
    validation_model_p_up = _final_p_up(validation_raw, validation_predictions, target_mode)

    best_payload: dict[str, Any] | None = None
    best_frontier: pd.DataFrame | None = None
    best_validation_decisions: pd.Series | None = None
    best_train_p_up: pd.Series | None = None
    best_validation_p_up: pd.Series | None = None
    base_train_p_up = pd.to_numeric(train_predictions["p_up"], errors="coerce").astype("float64").clip(0.0, 1.0)
    base_validation_p_up = pd.to_numeric(validation_predictions["p_up"], errors="coerce").astype("float64").clip(0.0, 1.0)
    for weight in variant.get("blend_weights", [1.0]):
        w = float(weight)
        train_p_up = (w * train_model_p_up + (1.0 - w) * base_train_p_up).clip(0.0, 1.0)
        validation_p_up = (w * validation_model_p_up + (1.0 - w) * base_validation_p_up).clip(0.0, 1.0)
        frontier, best, validation_decisions = _search_thresholds(
            validation_predictions,
            validation_p_up,
            config,
            blend_weight=w,
        )
        if best_payload is None or (
            best["accepted_sample_accuracy"],
            best["selection_score"],
            best["utility"],
            best["coverage"],
        ) > (
            best_payload["best_gate"]["accepted_sample_accuracy"],
            best_payload["best_gate"]["selection_score"],
            best_payload["best_gate"]["utility"],
            best_payload["best_gate"]["coverage"],
        ):
            best_payload = {"best_gate": best, "blend_weight": w}
            best_frontier = frontier
            best_validation_decisions = validation_decisions
            best_train_p_up = train_p_up
            best_validation_p_up = validation_p_up
    if best_payload is None or best_frontier is None or best_validation_decisions is None:
        raise RuntimeError(f"variant produced no result: {variant['name']}")
    best_gate = best_payload["best_gate"]
    train_decisions = _decisions_from_probability(
        best_train_p_up,
        t_up=float(best_gate["t_up"]),
        t_down=float(best_gate["t_down"]),
    )
    train_metrics = compute_decision_metrics(
        train_predictions["target"],
        best_train_p_up,
        train_decisions,
        selected_t_up=float(best_gate["t_up"]),
        selected_t_down=float(best_gate["t_down"]),
    )
    train_metrics.update(compute_reversal_continuation_metrics(train_predictions, train_decisions))
    validation_metrics = compute_decision_metrics(
        validation_predictions["target"],
        best_validation_p_up,
        best_validation_decisions,
        selected_t_up=float(best_gate["t_up"]),
        selected_t_down=float(best_gate["t_down"]),
    )
    validation_metrics.update(compute_reversal_continuation_metrics(validation_predictions, best_validation_decisions))
    result = {
        "variant": variant["name"],
        "direction": variant.get("direction"),
        "model_target": target_mode,
        "feature_set": variant["feature_set"],
        "feature_count": int(train_x.shape[1]),
        "raw_feature_count": len(features),
        "add_meta_features": add_meta,
        "blend_weight": float(best_payload["blend_weight"]),
        "best_gate": best_gate,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "accepted": bool(
            validation_metrics["coverage"] >= float(config["objective"]["min_coverage"])
            and validation_metrics["accepted_sample_accuracy"]
            >= float(config["objective"]["target_accepted_sample_accuracy"])
            and validation_metrics["utility"] > 0.0
        ),
    }
    return result, best_frontier


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    baseline_experiment = Path(config["baseline_experiment"])
    follow_experiment = Path(config["follow_experiment"]) if config.get("follow_experiment") else None
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_inputs(baseline_experiment)
    follow_train_predictions = _load_predictions(follow_experiment, "train")
    follow_validation_predictions = _load_predictions(follow_experiment, "validation")

    baseline_metrics = compute_decision_metrics(
        validation_predictions["target"],
        validation_predictions["p_up"],
        validation_predictions["decision"],
        selected_t_up=float(validation_predictions["selected_t_up"].iloc[0]),
        selected_t_down=float(validation_predictions["selected_t_down"].iloc[0]),
    )
    baseline_metrics.update(compute_reversal_continuation_metrics(validation_predictions, validation_predictions["decision"]))

    variant_results: list[dict[str, Any]] = []
    for variant in config["variants"]:
        result, frontier = _evaluate_variant(
            variant,
            config,
            all_features,
            train_frame,
            validation_frame,
            train_predictions,
            validation_predictions,
            follow_train_predictions,
            follow_validation_predictions,
        )
        variant_results.append(result)
        frontier.to_csv(output_dir / f"{variant['name']}_frontier.csv", index=False)

    best_variant = max(
        variant_results,
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
                "direction": row["direction"],
                "feature_count": row["feature_count"],
                "blend_weight": row["blend_weight"],
                "coverage": row["validation_metrics"]["coverage"],
                "accepted_sample_accuracy": row["validation_metrics"]["accepted_sample_accuracy"],
                "selection_score": row["validation_metrics"]["selection_score"],
                "utility": row["validation_metrics"]["utility"],
                "accepted_count": row["validation_metrics"]["accepted_count"],
                "continuation_accepted_accuracy": row["validation_metrics"]["continuation_accepted_accuracy"],
                "reversal_accepted_accuracy": row["validation_metrics"]["reversal_accepted_accuracy"],
                "accepted": row["accepted"],
            }
            for row in variant_results
        ]
    ).sort_values(["accepted_sample_accuracy", "selection_score"], ascending=False)
    summary_path = output_dir / "variant_summary.csv"
    summary.to_csv(summary_path, index=False)

    report = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "baseline_experiment": str(baseline_experiment),
        "follow_experiment": str(follow_experiment) if follow_experiment else None,
        "primary_metric": "validation accepted_sample_accuracy with coverage >= min_coverage",
        "mode": config["mode"],
        "objective": config["objective"],
        "threshold_search": config["threshold_search"],
        "train_window": _window_summary(train_frame),
        "validation_window": _window_summary(validation_frame),
        "baseline_validation_metrics": baseline_metrics,
        "best_variant": best_variant["variant"],
        "train_metrics": best_variant["train_metrics"],
        "validation_metrics": best_variant["validation_metrics"],
        "required_train_metrics": _required_metrics(best_variant["train_metrics"]),
        "required_validation_metrics": _required_metrics(best_variant["validation_metrics"]),
        "variant_results": variant_results,
        "variant_summary_path": str(summary_path),
        "accepted": bool(best_variant["accepted"]),
        "target_met": bool(best_variant["accepted"]),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run five reversal-oriented accepted-accuracy optimization variants.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run(args.config)
    metrics = report["validation_metrics"]
    print(
        json.dumps(
            {
                "report_path": str(Path(report["variant_summary_path"]).with_name("report.json")),
                "accepted": report["accepted"],
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
