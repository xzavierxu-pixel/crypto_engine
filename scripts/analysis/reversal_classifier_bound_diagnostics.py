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
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.reversal_auxiliary_ablation import (  # noqa: E402
    _apply_aux_gate,
    _fit_model,
    _load_baseline_inputs,
    _prepare_decision_frame,
    _reversal_target,
    _select_features,
)
from src.model.reversal_hybrid import compute_decision_metrics, compute_reversal_continuation_metrics  # noqa: E402


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_reversal_classifier_bound_diagnostics.yaml")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _first_minute_reversal_decision(frame: pd.DataFrame) -> pd.Series:
    fm_side = frame["first_minute_side"].astype("object")
    decisions = pd.Series("ABSTAIN", index=frame.index, dtype="object")
    decisions.loc[fm_side == "YES"] = "DOWN"
    decisions.loc[fm_side == "NO"] = "UP"
    return decisions


def _fit_reversal_classifier(
    train_frame: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
    config: dict[str, Any],
) -> lgb.LGBMClassifier:
    params = dict(config["model"])
    params.setdefault("objective", "binary")
    params.setdefault("verbosity", -1)
    positives = int(y_train.sum())
    negatives = int((y_train == 0).sum())
    if positives:
        params.setdefault("scale_pos_weight", negatives / positives)
    model = lgb.LGBMClassifier(**params)
    model.fit(train_frame[features], y_train.astype(int))
    return model


def _classifier_summary(y_true: pd.Series, risk: pd.Series) -> dict[str, Any]:
    y = y_true.astype(int)
    score = risk.astype("float64").clip(0.0, 1.0)
    top_precision = []
    for share in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        threshold = float(score.quantile(1.0 - share))
        mask = score >= threshold
        top_precision.append(
            {
                "top_share": float(share),
                "count": float(mask.sum()),
                "reversal_precision": float(y.loc[mask].mean()) if int(mask.sum()) else 0.0,
                "threshold": threshold,
            }
        )
    return {
        "reversal_roc_auc": float(roc_auc_score(y, score)) if y.nunique() == 2 else 0.0,
        "reversal_average_precision": float(average_precision_score(y, score)) if y.nunique() == 2 else 0.0,
        "top_precision": top_precision,
    }


def _search_override(
    validation_predictions: pd.DataFrame,
    base_decisions: pd.Series,
    risk: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    search = config["override_search"]
    min_coverage = float(config["objective"]["min_coverage"])
    reversal_decision = _first_minute_reversal_decision(validation_predictions)
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for threshold in search["risk_thresholds"]:
        for include_abstains in search["include_abstains"]:
            decisions = base_decisions.copy().astype("object")
            candidate_pool = (decisions != "ABSTAIN") | bool(include_abstains)
            override = candidate_pool & (risk >= float(threshold)) & reversal_decision.isin(["UP", "DOWN"])
            decisions.loc[override] = reversal_decision.loc[override]
            metrics = compute_decision_metrics(
                validation_predictions["target"],
                validation_predictions["p_up"],
                decisions,
                selected_t_up=float(threshold),
                selected_t_down=0.0,
            )
            metrics.update(compute_reversal_continuation_metrics(validation_predictions, decisions))
            row = {"risk_threshold": float(threshold), "include_abstains": bool(include_abstains), **metrics}
            records.append(row)
            if metrics["coverage"] >= min_coverage and metrics["utility"] > 0.0:
                eligible.append(row)
    pool = eligible if eligible else records
    best = max(
        pool,
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


def _target_requirement(metrics: dict[str, float], config: dict[str, Any]) -> dict[str, float]:
    sample_count = int(metrics["sample_count"])
    min_accepted = int(np.ceil(sample_count * float(config["objective"]["min_coverage"])))
    target_acc = float(config["objective"]["target_accepted_sample_accuracy"])
    current_accepted = int(metrics["accepted_count"])
    current_correct = int(round(metrics["accepted_sample_accuracy"] * current_accepted))
    return {
        "sample_count": float(sample_count),
        "min_accepted_for_coverage": float(min_accepted),
        "target_accuracy": target_acc,
        "correct_needed_at_min_coverage": float(np.ceil(min_accepted * target_acc)),
        "current_accepted": float(current_accepted),
        "current_correct_estimate": float(current_correct),
        "additional_correct_needed_at_min_coverage": float(np.ceil(min_accepted * target_acc) - current_correct),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_baseline_inputs(
        Path(config["baseline_experiment"])
    )
    gate_features = _select_features(all_features, {"mode": "keep_regex", **config["feature_sets"]["impulse"]})
    aux_model = _fit_model(
        train_frame,
        _reversal_target(train_frame),
        gate_features,
        {"model": {"n_estimators": 450, "learning_rate": 0.035, "num_leaves": 31, "min_child_samples": 80, "subsample": 0.85, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 8.0, "random_state": 20260611}},
    )
    aux_risk = pd.Series(aux_model.predict_proba(validation_frame[gate_features])[:, 1], index=validation_frame.index)
    current_frame = _prepare_decision_frame(validation_predictions, aux_risk)
    current_decisions = _apply_aux_gate(
        current_frame,
        reversal_risk_threshold=float(config["current_best_gate"]["reversal_risk_threshold"]),
        base_band=float(config["current_best_gate"]["base_band"]),
    )
    current_metrics = compute_decision_metrics(
        validation_predictions["target"],
        validation_predictions["p_up"],
        current_decisions,
        selected_t_up=float(config["current_best_gate"]["reversal_risk_threshold"]),
        selected_t_down=float(config["current_best_gate"]["base_band"]),
    )
    current_metrics.update(compute_reversal_continuation_metrics(validation_predictions, current_decisions))

    y_train_reversal = _reversal_target(train_frame)
    y_valid_reversal = _reversal_target(validation_frame)
    variant_results = []
    for name, feature_set in config["feature_sets"].items():
        features = _select_features(all_features, {"mode": "keep_regex", **feature_set})
        model = _fit_reversal_classifier(train_frame, y_train_reversal, features, config)
        risk = pd.Series(model.predict_proba(validation_frame[features])[:, 1], index=validation_frame.index)
        frontier, best_override = _search_override(validation_predictions, current_decisions, risk, config)
        frontier.to_csv(output_dir / f"{name}_override_frontier.csv", index=False)
        variant_results.append(
            {
                "variant": name,
                "feature_count": len(features),
                "classifier_metrics": _classifier_summary(y_valid_reversal, risk),
                "best_override_metrics": best_override,
            }
        )

    best_variant = max(
        variant_results,
        key=lambda row: (
            row["best_override_metrics"]["coverage"] >= float(config["objective"]["min_coverage"]),
            row["best_override_metrics"]["accepted_sample_accuracy"],
            row["best_override_metrics"]["selection_score"],
        ),
    )
    summary = pd.DataFrame(
        [
            {
                "variant": row["variant"],
                "feature_count": row["feature_count"],
                "reversal_roc_auc": row["classifier_metrics"]["reversal_roc_auc"],
                "reversal_average_precision": row["classifier_metrics"]["reversal_average_precision"],
                "coverage": row["best_override_metrics"]["coverage"],
                "accepted_sample_accuracy": row["best_override_metrics"]["accepted_sample_accuracy"],
                "selection_score": row["best_override_metrics"]["selection_score"],
                "utility": row["best_override_metrics"]["utility"],
                "accepted_count": row["best_override_metrics"]["accepted_count"],
                "target_met": bool(
                    row["best_override_metrics"]["coverage"] >= float(config["objective"]["min_coverage"])
                    and row["best_override_metrics"]["accepted_sample_accuracy"]
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
        "current_best_metrics": current_metrics,
        "target_requirement": _target_requirement(current_metrics, config),
        "best_variant": best_variant["variant"],
        "validation_metrics": best_variant["best_override_metrics"],
        "variant_results": variant_results,
        "variant_summary_path": str(summary_path),
        "target_met": bool(
            best_variant["best_override_metrics"]["coverage"] >= float(config["objective"]["min_coverage"])
            and best_variant["best_override_metrics"]["accepted_sample_accuracy"]
            >= float(config["objective"]["target_accepted_sample_accuracy"])
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose reversal classifier bounds and override search.")
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
