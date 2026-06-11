from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
import yaml
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.reversal_hybrid import OBJECTIVE_METRIC_FIELDS, compute_decision_metrics, compute_reversal_continuation_metrics  # noqa: E402


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_catboost_continuation_side_coordinate_search.yaml")
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


def _regimes(predictions: pd.DataFrame) -> pd.Series:
    missing = {"timestamp", "first_minute_side"}.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing regime columns: {sorted(missing)}")
    timestamps = pd.to_datetime(predictions["timestamp"], utc=True)
    day = timestamps.dt.dayofweek
    hour = timestamps.dt.hour
    session = pd.cut(hour, bins=[-1, 7, 15, 23], labels=["asia", "europe", "us"]).astype(str)
    fm_side = predictions["first_minute_side"].astype("object")
    if not fm_side.isin(["YES", "NO"]).all():
        bad_values = sorted(str(value) for value in fm_side.loc[~fm_side.isin(["YES", "NO"])].dropna().unique())
        raise ValueError(f"first_minute_side has unsupported values: {bad_values}")
    values = ["d" + str(d) + "_" + s + "_fm_" + side.lower() for d, s, side in zip(day, session, fm_side)]
    return pd.Series(values, index=predictions.index)


def _decisions(p_up: pd.Series, predictions: pd.DataFrame, regimes: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    fm_side = predictions["first_minute_side"].astype("object")
    decisions = pd.Series("ABSTAIN", index=p_up.index, dtype="object")
    for regime, threshold in thresholds.items():
        mask = regimes == regime
        yes_mask = mask & (fm_side == "YES")
        no_mask = mask & (fm_side == "NO")
        decisions.loc[yes_mask & (p_up >= float(threshold))] = "UP"
        decisions.loc[no_mask & (p_up <= float(threshold))] = "DOWN"
    return decisions


def _selected_thresholds(thresholds: dict[str, float]) -> tuple[float, float]:
    yes_values = [value for key, value in thresholds.items() if key.endswith("_fm_yes")]
    no_values = [value for key, value in thresholds.items() if key.endswith("_fm_no")]
    return (
        float(sum(yes_values) / len(yes_values)) if yes_values else 0.0,
        float(sum(no_values) / len(no_values)) if no_values else 0.0,
    )


def _evaluate(
    predictions: pd.DataFrame,
    p_up: pd.Series,
    regimes: pd.Series,
    thresholds: dict[str, float],
    label: str,
) -> dict[str, Any]:
    decisions = _decisions(p_up, predictions, regimes, thresholds)
    selected_t_up, selected_t_down = _selected_thresholds(thresholds)
    metrics = compute_decision_metrics(
        predictions["target"],
        p_up,
        decisions,
        selected_t_up=selected_t_up,
        selected_t_down=selected_t_down,
    )
    metrics.update(compute_reversal_continuation_metrics(predictions, decisions))
    return {
        "step": label,
        "thresholds": json.dumps(thresholds, sort_keys=True),
        **metrics,
    }


def _better(candidate: dict[str, Any], incumbent: dict[str, Any], min_coverage: float) -> bool:
    if candidate["coverage"] < min_coverage or candidate["utility"] <= 0.0:
        return False
    return (
        candidate["accepted_sample_accuracy"],
        candidate["selection_score"],
        candidate["utility"],
        candidate["coverage"],
        candidate["accepted_count"],
    ) > (
        incumbent["accepted_sample_accuracy"],
        incumbent["selection_score"],
        incumbent["utility"],
        incumbent["coverage"],
        incumbent["accepted_count"],
    )


def _candidate_values(regime: str, config: dict[str, Any]) -> list[float]:
    search = config["threshold_search"]
    key = "yes_side_values" if regime.endswith("_fm_yes") else "no_side_values"
    return [float(value) for value in search[key]]


def _coordinate_search(
    predictions: pd.DataFrame,
    p_up: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float], pd.Series]:
    regimes = _regimes(predictions)
    regime_values = sorted(regimes.dropna().unique())
    search = config["threshold_search"]
    thresholds = {
        regime: float(search["initial_t_up"]) if regime.endswith("_fm_yes") else float(search["initial_t_down"])
        for regime in regime_values
    }
    min_coverage = float(config["objective"]["min_coverage"])
    records: list[dict[str, Any]] = []
    best = _evaluate(predictions, p_up, regimes, thresholds, "initial")
    records.append(best)
    for pass_index in range(int(search["passes"])):
        changed = False
        for regime in regime_values:
            local_best = best
            local_thresholds = dict(thresholds)
            for value in _candidate_values(regime, config):
                candidate_thresholds = dict(thresholds)
                candidate_thresholds[regime] = value
                row = _evaluate(predictions, p_up, regimes, candidate_thresholds, f"pass{pass_index}_{regime}")
                row["regime"] = regime
                row["candidate_threshold"] = value
                records.append(row)
                if _better(row, local_best, min_coverage):
                    local_best = row
                    local_thresholds = candidate_thresholds
            if local_best is not best:
                best = local_best
                thresholds = local_thresholds
                changed = True
        if not changed:
            break
    best["constraint_satisfied"] = bool(best["coverage"] >= min_coverage and best["utility"] > 0.0)
    best["objective"] = "accepted_sample_accuracy"
    best["hard_constraint"] = "coverage_only"
    return pd.DataFrame.from_records(records), best, thresholds, regimes


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
    model = CatBoostClassifier(**config["model"])
    model.fit(train_frame[features], train_frame["target"].astype(int))
    train_p_up = pd.Series(model.predict_proba(train_frame[features])[:, 1], index=train_frame.index).clip(0.0, 1.0)
    validation_p_up = pd.Series(
        model.predict_proba(validation_frame[features])[:, 1],
        index=validation_frame.index,
    ).clip(0.0, 1.0)
    frontier, best, thresholds, validation_regimes = _coordinate_search(validation_predictions, validation_p_up, config)
    train_regimes = _regimes(train_predictions)
    train_metrics = _evaluate(train_predictions, train_p_up, train_regimes, thresholds, "train_apply_validation_thresholds")
    frontier_path = output_dir / "continuation_side_coordinate_frontier.csv"
    frontier.to_csv(frontier_path, index=False)
    thresholds_path = output_dir / "thresholds.json"
    thresholds_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True), encoding="utf-8")
    regime_summary_path = output_dir / "regime_summary.csv"
    validation_regimes.value_counts().sort_index().rename_axis("regime").reset_index(name="sample_count").to_csv(
        regime_summary_path,
        index=False,
    )
    summary_path = output_dir / "variant_summary.csv"
    pd.DataFrame(
        [
            {
                "variant": "continuation_side_coordinate",
                "coverage": best["coverage"],
                "accepted_sample_accuracy": best["accepted_sample_accuracy"],
                "selection_score": best["selection_score"],
                "utility": best["utility"],
                "accepted_count": best["accepted_count"],
                "target_met": bool(
                    best["coverage"] >= float(config["objective"]["min_coverage"])
                    and best["accepted_sample_accuracy"] >= float(config["objective"]["target_accepted_sample_accuracy"])
                ),
            }
        ]
    ).to_csv(summary_path, index=False)
    report = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "primary_metric": "validation accepted_sample_accuracy with coverage >= min_coverage",
        "mode": config["mode"],
        "objective": config["objective"],
        "train_window": _window_summary(train_frame),
        "validation_window": _window_summary(validation_frame),
        "feature_count": len(features),
        "feature_set_patterns": list(config["feature_set"]["patterns"]),
        "best_variant": "continuation_side_coordinate",
        "train_metrics": train_metrics,
        "validation_metrics": best,
        "required_train_metrics": _required_metrics(train_metrics),
        "required_validation_metrics": _required_metrics(best),
        "frontier_path": str(frontier_path),
        "thresholds_path": str(thresholds_path),
        "regime_summary_path": str(regime_summary_path),
        "variant_summary_path": str(summary_path),
        "target_met": bool(
            best["coverage"] >= float(config["objective"]["min_coverage"])
            and best["accepted_sample_accuracy"] >= float(config["objective"]["target_accepted_sample_accuracy"])
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuation-side coordinate threshold search.")
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
