from __future__ import annotations

import argparse
import json
from pathlib import Path
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

from scripts.analysis.reversal_direction_meta_gate import _augment_meta_features  # noqa: E402
from scripts.analysis.reversal_precision_gate import (  # noqa: E402
    _load_baseline_inputs,
    _load_config,
    _reversal_target,
    _select_features,
    _window_summary,
)
from src.model.reversal_hybrid import compute_decision_metrics, compute_reversal_continuation_metrics  # noqa: E402


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_reversal_rescue_gate.yaml")


def _baseline_side(predictions: pd.DataFrame) -> pd.Series:
    return predictions["decision"].replace({"UP": "YES", "DOWN": "NO", "ABSTAIN": None})


def _baseline_continuation_mask(predictions: pd.DataFrame) -> pd.Series:
    side = _baseline_side(predictions)
    return (predictions["decision"] != "ABSTAIN") & (side == predictions["first_minute_side"])


def _baseline_reversal_mask(predictions: pd.DataFrame) -> pd.Series:
    side = _baseline_side(predictions)
    return (predictions["decision"] != "ABSTAIN") & side.isin(["YES", "NO"]) & (side != predictions["first_minute_side"])


def _reversal_decision_from_first_minute(predictions: pd.DataFrame) -> pd.Series:
    fm_side = predictions["first_minute_side"].astype("object")
    decision = pd.Series("ABSTAIN", index=predictions.index, dtype="object")
    decision.loc[fm_side == "YES"] = "DOWN"
    decision.loc[fm_side == "NO"] = "UP"
    return decision


def _fit_rescue_model(train_x: pd.DataFrame, y_train: pd.Series, config: dict[str, Any]):
    params = dict(config.get("model", {}))
    model_type = str(params.pop("type", "lightgbm_lgbmclassifier"))
    if model_type == "catboost_classifier":
        params.setdefault("loss_function", "Logloss")
        params.setdefault("verbose", False)
        model = CatBoostClassifier(**params)
    else:
        params.setdefault("objective", "binary")
        params.setdefault("verbosity", -1)
        positives = int(y_train.sum())
        negatives = int((y_train == 0).sum())
        if positives:
            params.setdefault("scale_pos_weight", negatives / positives)
        model = lgb.LGBMClassifier(**params)
    model.fit(train_x, y_train.astype(int))
    return model


def apply_rescue_gate(
    predictions: pd.DataFrame,
    rescue_probability: pd.Series,
    *,
    keep_threshold: float,
    reverse_threshold: float,
    keep_base_reversal: bool,
    use_abstains: bool,
) -> pd.Series:
    risk = rescue_probability.astype("float64").clip(0.0, 1.0)
    output = pd.Series("ABSTAIN", index=predictions.index, dtype="object")
    base_continuation = _baseline_continuation_mask(predictions)
    base_reversal = _baseline_reversal_mask(predictions)
    reversal_decision = _reversal_decision_from_first_minute(predictions)

    keep_continuation = base_continuation & (risk <= float(keep_threshold))
    reverse_continuation = base_continuation & (risk >= float(reverse_threshold))
    output.loc[keep_continuation] = predictions.loc[keep_continuation, "decision"]
    output.loc[reverse_continuation] = reversal_decision.loc[reverse_continuation]

    if keep_base_reversal:
        output.loc[base_reversal] = predictions.loc[base_reversal, "decision"]

    if use_abstains:
        abstained = predictions["decision"] == "ABSTAIN"
        follow_yes = abstained & (risk <= float(keep_threshold)) & (predictions["first_minute_side"] == "YES")
        follow_no = abstained & (risk <= float(keep_threshold)) & (predictions["first_minute_side"] == "NO")
        output.loc[follow_yes] = "UP"
        output.loc[follow_no] = "DOWN"
        reverse_abstain = abstained & (risk >= float(reverse_threshold))
        output.loc[reverse_abstain] = reversal_decision.loc[reverse_abstain]

    return output


def _search_gate(predictions: pd.DataFrame, p_rescue: pd.Series, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    objective = config["objective"]
    search = config["threshold_search"]
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for keep_threshold in search["keep_thresholds"]:
        for reverse_threshold in search["reverse_thresholds"]:
            if float(keep_threshold) >= float(reverse_threshold):
                continue
            for keep_base_reversal in search["keep_base_reversal"]:
                for use_abstains in search["use_abstains"]:
                    decisions = apply_rescue_gate(
                        predictions,
                        p_rescue,
                        keep_threshold=float(keep_threshold),
                        reverse_threshold=float(reverse_threshold),
                        keep_base_reversal=bool(keep_base_reversal),
                        use_abstains=bool(use_abstains),
                    )
                    metrics = compute_decision_metrics(
                        predictions["target"],
                        predictions["p_up"],
                        decisions,
                        selected_t_up=float(reverse_threshold),
                        selected_t_down=float(keep_threshold),
                    )
                    metrics.update(compute_reversal_continuation_metrics(predictions, decisions))
                    row = {
                        "keep_threshold": float(keep_threshold),
                        "reverse_threshold": float(reverse_threshold),
                        "keep_base_reversal": bool(keep_base_reversal),
                        "use_abstains": bool(use_abstains),
                        **metrics,
                    }
                    records.append(row)
                    if (
                        metrics["coverage"] > float(objective["min_coverage"])
                        and metrics["continuation_accepted_accuracy"] >= float(objective["min_continuation_accepted_accuracy"])
                        and metrics["reversal_accepted_accuracy"] > float(objective["min_reversal_accepted_accuracy"])
                    ):
                        eligible.append(row)
    pool = eligible if eligible else records
    best = max(
        pool,
        key=lambda row: (
            row["reversal_accepted_accuracy"],
            row["continuation_accepted_accuracy"] >= float(objective["min_continuation_accepted_accuracy"]),
            row["coverage"] > float(objective["min_coverage"]),
            row["selection_score"],
            row["coverage"],
        ),
    )
    return pd.DataFrame.from_records(records), {
        **best,
        "constraint_satisfied": bool(eligible),
        "objective": "maximize reversal_accepted_accuracy subject to coverage and continuation constraints",
        "hard_constraints": {
            "min_coverage": float(objective["min_coverage"]),
            "min_continuation_accepted_accuracy": float(objective["min_continuation_accepted_accuracy"]),
            "min_reversal_accepted_accuracy": float(objective["min_reversal_accepted_accuracy"]),
        },
        "fallback_reason": None if eligible else "no candidate satisfied all constraints",
    }


def _gap_summary(predictions: pd.DataFrame) -> dict[str, float]:
    reversal = _reversal_target(predictions).astype(bool)
    base_cont = _baseline_continuation_mask(predictions)
    base_rev = _baseline_reversal_mask(predictions)
    actual_cont = ~reversal
    wrong_reversal_accepts = int((base_cont & reversal).sum())
    correct_reversal_accepts = int((base_rev & reversal).sum())
    accepted_count = int((predictions["decision"] != "ABSTAIN").sum())
    required_correct_reversal_for_gt_half = wrong_reversal_accepts + 1
    return {
        "baseline_accepted_count": float(accepted_count),
        "baseline_actual_continuation_kept_correct": float((base_cont & actual_cont).sum()),
        "baseline_actual_reversal_kept_wrong_continuation": float(wrong_reversal_accepts),
        "baseline_actual_reversal_kept_correct_reversal": float(correct_reversal_accepts),
        "additional_correct_reversal_needed_for_gt_half_if_wrong_unchanged": float(
            max(required_correct_reversal_for_gt_half - correct_reversal_accepts, 0)
        ),
        "max_wrong_reversal_accepts_allowed_if_correct_unchanged": float(max(correct_reversal_accepts - 1, 0)),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    baseline_experiment = Path(config["baseline_experiment"])
    follow_experiment = Path(config["follow_experiment"]) if config.get("follow_experiment") else None
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_baseline_inputs(baseline_experiment)
    follow_train_predictions = pd.read_parquet(follow_experiment / "train_predictions.parquet") if follow_experiment else None
    follow_validation_predictions = pd.read_parquet(follow_experiment / "validation_predictions.parquet") if follow_experiment else None
    base_features = _select_features(all_features, config)
    train_x = _augment_meta_features(train_frame, train_predictions, follow_train_predictions, base_features)
    validation_x = _augment_meta_features(validation_frame, validation_predictions, follow_validation_predictions, base_features)

    train_scope = _baseline_continuation_mask(train_predictions)
    y_rescue = _reversal_target(train_predictions)
    model = _fit_rescue_model(train_x.loc[train_scope], y_rescue.loc[train_scope], config)
    train_rescue = pd.Series(model.predict_proba(train_x)[:, 1], index=train_frame.index)
    validation_rescue = pd.Series(model.predict_proba(validation_x)[:, 1], index=validation_frame.index)
    frontier, best = _search_gate(validation_predictions, validation_rescue, config)

    train_decisions = apply_rescue_gate(
        train_predictions,
        train_rescue,
        keep_threshold=float(best["keep_threshold"]),
        reverse_threshold=float(best["reverse_threshold"]),
        keep_base_reversal=bool(best["keep_base_reversal"]),
        use_abstains=bool(best["use_abstains"]),
    )
    validation_decisions = apply_rescue_gate(
        validation_predictions,
        validation_rescue,
        keep_threshold=float(best["keep_threshold"]),
        reverse_threshold=float(best["reverse_threshold"]),
        keep_base_reversal=bool(best["keep_base_reversal"]),
        use_abstains=bool(best["use_abstains"]),
    )
    train_metrics = compute_decision_metrics(
        train_predictions["target"],
        train_predictions["p_up"],
        train_decisions,
        selected_t_up=float(best["reverse_threshold"]),
        selected_t_down=float(best["keep_threshold"]),
    )
    train_metrics.update(compute_reversal_continuation_metrics(train_predictions, train_decisions))
    validation_metrics = compute_decision_metrics(
        validation_predictions["target"],
        validation_predictions["p_up"],
        validation_decisions,
        selected_t_up=float(best["reverse_threshold"]),
        selected_t_down=float(best["keep_threshold"]),
    )
    validation_metrics.update(compute_reversal_continuation_metrics(validation_predictions, validation_decisions))
    baseline_metrics = compute_decision_metrics(
        validation_predictions["target"],
        validation_predictions["p_up"],
        validation_predictions["decision"],
        selected_t_up=float(validation_predictions["selected_t_up"].iloc[0]),
        selected_t_down=float(validation_predictions["selected_t_down"].iloc[0]),
    )
    baseline_metrics.update(compute_reversal_continuation_metrics(validation_predictions, validation_predictions["decision"]))

    frontier_path = output_dir / "frontier.csv"
    report_path = output_dir / "report.json"
    frontier.to_csv(frontier_path, index=False)
    report = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "baseline_experiment": str(baseline_experiment),
        "follow_experiment": str(follow_experiment) if follow_experiment else None,
        "primary_metric": "validation reversal_accepted_accuracy with coverage and continuation constraints",
        "mode": config["mode"],
        "objective": config["objective"],
        "threshold_search": config["threshold_search"],
        "feature_variant": config["feature_variant"],
        "feature_count": int(train_x.shape[1]),
        "raw_impulse_feature_count": len(base_features),
        "train_window": _window_summary(train_frame),
        "validation_window": _window_summary(validation_frame),
        "baseline_gap_summary": _gap_summary(validation_predictions),
        "baseline_validation_metrics": baseline_metrics,
        "best_gate": best,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "frontier_path": str(frontier_path),
        "accepted": bool(best["constraint_satisfied"]),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline-continuation reversal rescue gate.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run(args.config)
    metrics = report["validation_metrics"]
    print(
        json.dumps(
            {
                "report_path": str(Path(report["frontier_path"]).with_name("report.json")),
                "accepted": report["accepted"],
                "coverage": metrics["coverage"],
                "continuation_accepted_accuracy": metrics["continuation_accepted_accuracy"],
                "reversal_accepted_accuracy": metrics["reversal_accepted_accuracy"],
                "selection_score": metrics["selection_score"],
                "accepted_count": metrics["accepted_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
