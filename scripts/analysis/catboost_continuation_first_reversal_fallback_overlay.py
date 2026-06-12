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
    apply_continuation_expert_decision,
    apply_reversal_only_decision,
    compute_decision_metrics,
    route_continuation_first_reversal_fallback,
)


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260612_continuation_first_reversal_weight8_abstain_overlay.yaml")
ACCEPTED_BASELINE_SELECTION_SCORE = 0.5748509217
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


def _selected_thresholds(thresholds: dict[str, float]) -> tuple[float, float]:
    yes_values = [value for key, value in thresholds.items() if key.endswith("_fm_yes")]
    no_values = [value for key, value in thresholds.items() if key.endswith("_fm_no")]
    return (
        float(sum(yes_values) / len(yes_values)) if yes_values else 0.0,
        float(sum(no_values) / len(no_values)) if no_values else 0.0,
    )


def _window_summary(frame: pd.DataFrame) -> dict[str, str | int]:
    column = "timestamp" if "timestamp" in frame.columns else "market_t0"
    timestamps = pd.to_datetime(frame[column], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamps.min()), "end": str(timestamps.max())}


def _required_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {field: float(metrics[field]) for field in OBJECTIVE_METRIC_FIELDS}


def _train_experts(
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    train_predictions: pd.DataFrame,
    features: list[str],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    continuation_model = CatBoostClassifier(**config["model"])
    continuation_model.fit(train_frame[features], train_frame["target"].astype(int))
    cont_train = pd.Series(continuation_model.predict_proba(train_frame[features])[:, 1], index=train_frame.index).clip(0.0, 1.0)
    cont_val = pd.Series(continuation_model.predict_proba(validation_frame[features])[:, 1], index=validation_frame.index).clip(0.0, 1.0)

    final_side = pd.Series(np.where(train_predictions["target"].astype(int) == 1, "YES", "NO"), index=train_predictions.index)
    reversal_mask = train_predictions["first_minute_side"].astype("object").isin(["YES", "NO"]) & (
        train_predictions["first_minute_side"].astype("object") != final_side
    )
    sample_weight = pd.Series(1.0, index=train_frame.index)
    sample_weight.loc[reversal_mask] = float(config["reversal_expert"]["sample_weight"])
    reversal_model = CatBoostClassifier(**config["model"])
    reversal_model.fit(train_frame[features], train_frame["target"].astype(int), sample_weight=sample_weight)
    rev_train = pd.Series(reversal_model.predict_proba(train_frame[features])[:, 1], index=train_frame.index).clip(0.0, 1.0)
    rev_val = pd.Series(reversal_model.predict_proba(validation_frame[features])[:, 1], index=validation_frame.index).clip(0.0, 1.0)
    return cont_train, cont_val, rev_train, rev_val


def _expert_decisions(
    predictions: pd.DataFrame,
    continuation_p_up: pd.Series,
    reversal_p_up: pd.Series,
    continuation_thresholds: dict[str, float],
    *,
    tau_rev: float,
) -> tuple[pd.Series, pd.Series]:
    cont_decision = apply_continuation_expert_decision(
        continuation_p_up,
        predictions["first_minute_side"],
        _regimes(predictions),
        continuation_thresholds,
    )
    rev_decision = apply_reversal_only_decision(
        reversal_p_up,
        predictions["first_minute_side"],
        t_up=float(tau_rev),
        t_down=float(1.0 - tau_rev),
    )
    return cont_decision, rev_decision


def _source_probability(
    continuation_p_up: pd.Series,
    reversal_p_up: pd.Series,
    final_source: pd.Series,
) -> pd.Series:
    probability = continuation_p_up.astype("float64").clip(0.0, 1.0).copy()
    fallback = final_source == "reversal_fallback"
    probability.loc[fallback] = reversal_p_up.loc[fallback].astype("float64").clip(0.0, 1.0)
    return probability


def _fallback_only_metrics(
    predictions: pd.DataFrame,
    decisions: pd.Series,
    *,
    residual_mask: pd.Series,
) -> dict[str, float]:
    fallback_mask = residual_mask & (decisions != "ABSTAIN")
    accepted_count = int(fallback_mask.sum())
    residual_count = int(residual_mask.sum())
    y = predictions["target"].astype(int)
    correct = ((decisions.loc[fallback_mask] == "UP") == (y.loc[fallback_mask] == 1)) if accepted_count else pd.Series(dtype="bool")
    accuracy = float(correct.mean()) if accepted_count else 0.0
    coverage_on_residual = float(accepted_count / residual_count) if residual_count else 0.0
    utility = float(coverage_on_residual * (2.0 * accuracy - 1.0))
    return {
        "reversal_fallback_residual_sample_count": float(residual_count),
        "reversal_fallback_accepted_count": float(accepted_count),
        "reversal_fallback_coverage_on_residual": coverage_on_residual,
        "reversal_fallback_accepted_accuracy": accuracy,
        "reversal_fallback_utility": utility,
    }


def _evaluate_overlay(
    predictions: pd.DataFrame,
    continuation_p_up: pd.Series,
    reversal_p_up: pd.Series,
    continuation_thresholds: dict[str, float],
    *,
    tau_rev: float,
) -> dict[str, Any]:
    cont_decision, rev_decision = _expert_decisions(
        predictions,
        continuation_p_up,
        reversal_p_up,
        continuation_thresholds,
        tau_rev=tau_rev,
    )
    routed = route_continuation_first_reversal_fallback(cont_decision, rev_decision)
    cont_t_up, cont_t_down = _selected_thresholds(continuation_thresholds)
    continuation_only = compute_decision_metrics(
        predictions["target"],
        continuation_p_up,
        cont_decision,
        selected_t_up=cont_t_up,
        selected_t_down=cont_t_down,
    )
    combined_probability = _source_probability(continuation_p_up, reversal_p_up, routed["final_source"])
    combined = compute_decision_metrics(
        predictions["target"],
        combined_probability,
        routed["final_decision"],
        selected_t_up=float(tau_rev),
        selected_t_down=float(1.0 - tau_rev),
    )
    fallback = _fallback_only_metrics(
        predictions,
        routed["final_decision"],
        residual_mask=~routed["continuation_accept"],
    )
    return {
        **combined,
        "tau_rev": float(tau_rev),
        "reversal_t_up": float(tau_rev),
        "reversal_t_down": float(1.0 - tau_rev),
        "continuation_coverage": float(continuation_only["coverage"]),
        "continuation_accepted_accuracy": float(continuation_only["accepted_sample_accuracy"]),
        "continuation_only_utility": float(continuation_only["utility"]),
        "continuation_only_accepted_count": float(continuation_only["accepted_count"]),
        "combined_coverage": float(combined["coverage"]),
        "combined_accepted_accuracy": float(combined["accepted_sample_accuracy"]),
        "combined_utility": float(combined["utility"]),
        "incremental_utility_from_reversal": float(combined["utility"] - continuation_only["utility"]),
        **fallback,
    }


def _success(row: dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(
        row["combined_coverage"] >= float(config["objective"]["min_coverage"])
        and row["combined_utility"] > row["continuation_only_utility"]
        and row["incremental_utility_from_reversal"] > 0.0
        and row["reversal_fallback_accepted_accuracy"] > 0.50
    )


def _best(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    passing = [row for row in records if row["success_criteria_satisfied"]]
    coverage_ok = [row for row in records if row["combined_coverage"] >= float(config["objective"]["min_coverage"])]
    pool = passing or coverage_ok or records
    best = max(
        pool,
        key=lambda row: (
            row["success_criteria_satisfied"],
            row["selection_score"],
            row["utility"],
            row["coverage"],
            row["accepted_count"],
            -abs(row["tau_rev"] - 0.65),
        ),
    )
    best["fallback_reason"] = None if passing else "no candidate satisfied all fallback success criteria"
    return best


def _make_audit(
    predictions: pd.DataFrame,
    continuation_p_up: pd.Series,
    reversal_p_up: pd.Series,
    continuation_thresholds: dict[str, float],
    *,
    tau_rev: float,
) -> pd.DataFrame:
    cont_decision, rev_decision = _expert_decisions(
        predictions,
        continuation_p_up,
        reversal_p_up,
        continuation_thresholds,
        tau_rev=tau_rev,
    )
    routed = route_continuation_first_reversal_fallback(cont_decision, rev_decision)
    y = predictions["target"].astype(int)
    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(predictions["timestamp"], utc=True).astype(str),
            "target": y,
            "first_minute_side": predictions["first_minute_side"].astype("object"),
            "continuation_p_up": continuation_p_up.astype("float64"),
            "continuation_decision": cont_decision.astype("object"),
            "continuation_accept": routed["continuation_accept"].astype(bool),
            "reversal_p_up": reversal_p_up.astype("float64"),
            "reversal_decision": rev_decision.astype("object"),
            "reversal_accept": routed["reversal_accept"].astype(bool),
            "reversal_fallback_accept": routed["reversal_fallback_accept"].astype(bool),
            "final_decision": routed["final_decision"].astype("object"),
            "final_source": routed["final_source"].astype("object"),
        }
    )
    final_accept = output["final_decision"] != "ABSTAIN"
    output["final_correct"] = ((output["final_decision"] == "UP") == (y == 1)) & final_accept
    return output


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_inputs(
        Path(config["baseline_experiment"])
    )
    features = _select_features(all_features, list(config["feature_set"]["patterns"]))
    leakage_features = sorted(set(features).intersection(LEAKAGE_BLOCKLIST))
    if leakage_features:
        raise ValueError(f"leakage features selected: {leakage_features}")

    cont_train_p, cont_val_p, rev_train_p, rev_val_p = _train_experts(
        config,
        train_frame,
        validation_frame,
        train_predictions,
        features,
    )
    continuation_thresholds = _read_json(Path(config["continuation_expert"]["thresholds_path"]))
    tau_values = [float(value) for value in config["threshold_search"]["tau_rev_grid"]]

    records: list[dict[str, Any]] = []
    for tau_rev in tau_values:
        row = _evaluate_overlay(
            validation_predictions,
            cont_val_p,
            rev_val_p,
            continuation_thresholds,
            tau_rev=tau_rev,
        )
        row["success_criteria_satisfied"] = _success(row, config)
        row["coverage_constraint_satisfied"] = bool(row["combined_coverage"] >= float(config["objective"]["min_coverage"]))
        row["official_acceptance_metric"] = "validation_selection_score_with_coverage_ge_0.70"
        records.append(row)

    frontier = pd.DataFrame.from_records(records)
    frontier_path = output_dir / "fallback_threshold_frontier.csv"
    frontier.to_csv(frontier_path, index=False)
    best = _best(records, config)

    train_metrics = _evaluate_overlay(
        train_predictions,
        cont_train_p,
        rev_train_p,
        continuation_thresholds,
        tau_rev=float(best["tau_rev"]),
    )
    train_metrics["success_criteria_satisfied"] = _success(train_metrics, config)
    train_metrics["coverage_constraint_satisfied"] = bool(train_metrics["combined_coverage"] >= float(config["objective"]["min_coverage"]))

    audit_path = output_dir / "validation_audit.csv"
    _make_audit(
        validation_predictions,
        cont_val_p,
        rev_val_p,
        continuation_thresholds,
        tau_rev=float(best["tau_rev"]),
    ).to_csv(audit_path, index=False)

    improved_over_accepted_baseline = bool(
        best["coverage_constraint_satisfied"] and best["selection_score"] > ACCEPTED_BASELINE_SELECTION_SCORE
    )
    report = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "primary_metric": "validation selection_score with coverage >= 0.70",
        "mode": config["mode"],
        "objective": config["objective"],
        "threshold_search": config["threshold_search"],
        "train_window": _window_summary(train_frame),
        "validation_window": _window_summary(validation_frame),
        "feature_count": len(features),
        "feature_set_patterns": list(config["feature_set"]["patterns"]),
        "continuation_expert_source": config["continuation_expert"]["source"],
        "continuation_expert_reason": "highest continuation_accepted_accuracy among current experiments",
        "reversal_expert_source": config["reversal_expert"]["source"],
        "reversal_expert_reason": "reversal_weight_8 requested for highest reversal_accepted_accuracy",
        "best_tau_rev": float(best["tau_rev"]),
        "train_metrics": {**_required_metrics(train_metrics), **{k: float(train_metrics[k]) for k in train_metrics if k not in OBJECTIVE_METRIC_FIELDS and isinstance(train_metrics[k], (int, float, bool))}},
        "validation_metrics": {**_required_metrics(best), **{k: float(best[k]) for k in best if k not in OBJECTIVE_METRIC_FIELDS and isinstance(best[k], (int, float, bool))}},
        "coverage_constraint_satisfied": bool(best["coverage_constraint_satisfied"]),
        "success_criteria_satisfied": bool(best["success_criteria_satisfied"]),
        "improved_over_accepted_baseline": improved_over_accepted_baseline,
        "accepted_baseline_selection_score": ACCEPTED_BASELINE_SELECTION_SCORE,
        "fallback_reason": best["fallback_reason"],
        "label_source": "polymarket_resolved",
        "label_version": "polymarket_resolved_gamma_v1",
        "deploy_training_mode": "not_regenerated",
        "offline_validation_metric_source": "validation fallback overlay evaluation only",
        "leakage_blocklist": sorted(LEAKAGE_BLOCKLIST),
        "leakage_features_selected": leakage_features,
        "output_files": {
            "fallback_threshold_frontier": str(frontier_path),
            "validation_audit": str(audit_path),
            "report": str(output_dir / "report.json"),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuation-first reversal fallback overlay.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run(args.config)
    metrics = report["validation_metrics"]
    print(
        json.dumps(
            {
                "report_path": report["output_files"]["report"],
                "best_tau_rev": report["best_tau_rev"],
                "coverage_constraint_satisfied": report["coverage_constraint_satisfied"],
                "success_criteria_satisfied": report["success_criteria_satisfied"],
                "selection_score": metrics["selection_score"],
                "utility": metrics["utility"],
                "accepted_sample_accuracy": metrics["accepted_sample_accuracy"],
                "coverage": metrics["coverage"],
                "incremental_utility_from_reversal": metrics["incremental_utility_from_reversal"],
                "reversal_fallback_accepted_accuracy": metrics["reversal_fallback_accepted_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
