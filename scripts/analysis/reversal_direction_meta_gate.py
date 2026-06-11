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

from scripts.analysis.reversal_precision_gate import (  # noqa: E402
    _load_baseline_inputs,
    _load_config,
    _select_features,
    _window_summary,
)
from src.model.reversal_hybrid import (  # noqa: E402
    compute_decision_metrics,
    compute_reversal_continuation_metrics,
    p_follow_from_direction_probability,
)


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_reversal_direction_meta_gate.yaml")


def _augment_meta_features(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    follow_predictions: pd.DataFrame | None,
    base_features: list[str],
) -> pd.DataFrame:
    output = frame.loc[:, base_features].copy()
    p_base = pd.to_numeric(predictions["p_up"], errors="coerce").astype("float64").clip(0.0, 1.0)
    output["meta_base_p_up"] = p_base
    output["meta_base_confidence"] = (p_base - 0.5).abs()
    output["meta_base_accepted"] = (predictions["decision"] != "ABSTAIN").astype("float64")
    output["meta_base_up"] = (predictions["decision"] == "UP").astype("float64")
    output["meta_base_down"] = (predictions["decision"] == "DOWN").astype("float64")
    output["meta_first_minute_yes"] = (predictions["first_minute_side"] == "YES").astype("float64")
    if "first_minute_return" in predictions.columns:
        fm_ret = pd.to_numeric(predictions["first_minute_return"], errors="coerce").astype("float64")
        output["meta_first_minute_return"] = fm_ret
        output["meta_first_minute_abs_return"] = fm_ret.abs()
    if follow_predictions is not None:
        follow = follow_predictions.reindex(predictions.index)
        p_follow = p_follow_from_direction_probability(follow)
        output["meta_follow_p_follow"] = p_follow
        output["meta_follow_p_reversal"] = 1.0 - p_follow
        output["meta_follow_p_up"] = pd.to_numeric(follow["p_up"], errors="coerce").astype("float64").clip(0.0, 1.0)
    return output


def _fit_direction_model(
    train_x: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> lgb.LGBMClassifier:
    params = dict(config.get("model", {}))
    model_type = str(params.pop("type", "lightgbm_lgbmclassifier"))
    if model_type == "catboost_classifier":
        params.setdefault("loss_function", "Logloss")
        params.setdefault("verbose", False)
        model = CatBoostClassifier(**params)
    else:
        params.setdefault("objective", "binary")
        params.setdefault("verbosity", -1)
        model = lgb.LGBMClassifier(**params)
    model.fit(train_x, y_train.astype(int))
    return model


def _training_target(frame: pd.DataFrame, predictions: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    target_mode = str(config.get("model_target", "final_direction"))
    if target_mode == "final_direction":
        return frame["target"].astype(int)
    if target_mode == "follow_relative":
        fm_yes = predictions["first_minute_side"].astype("object") == "YES"
        final_yes = frame["target"].astype(int) == 1
        return (fm_yes == final_yes).astype(int)
    raise ValueError(f"unsupported model_target: {target_mode}")


def _final_p_up_from_model_probability(
    frame: pd.DataFrame,
    model_probability: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    target_mode = str(config.get("model_target", "final_direction"))
    probability = model_probability.astype("float64").clip(0.0, 1.0)
    if target_mode == "final_direction":
        return probability
    if target_mode == "follow_relative":
        fm_yes = frame["first_minute_side"].astype("object") == "YES"
        return probability.where(fm_yes, 1.0 - probability).astype("float64").clip(0.0, 1.0)
    raise ValueError(f"unsupported model_target: {target_mode}")


def _decisions_from_probability(p_up: pd.Series, *, t_up: float, t_down: float) -> pd.Series:
    decisions = pd.Series("ABSTAIN", index=p_up.index, dtype="object")
    decisions.loc[p_up >= float(t_up)] = "UP"
    decisions.loc[p_up <= float(t_down)] = "DOWN"
    return decisions


def _decisions_from_first_minute_relative_probability(
    frame: pd.DataFrame,
    p_up: pd.Series,
    *,
    t_follow: float,
    t_reversal: float,
) -> pd.Series:
    fm_side = frame["first_minute_side"].astype("object")
    p_follow = p_up.where(fm_side == "YES", 1.0 - p_up).astype("float64").clip(0.0, 1.0)
    decisions = pd.Series("ABSTAIN", index=p_up.index, dtype="object")
    decisions.loc[(fm_side == "YES") & (p_follow >= float(t_follow))] = "UP"
    decisions.loc[(fm_side == "NO") & (p_follow >= float(t_follow))] = "DOWN"
    decisions.loc[(fm_side == "YES") & (p_follow <= float(t_reversal))] = "DOWN"
    decisions.loc[(fm_side == "NO") & (p_follow <= float(t_reversal))] = "UP"
    return decisions


def _search_thresholds(frame: pd.DataFrame, p_up: pd.Series, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    objective = config["objective"]
    search = config["threshold_search"]
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    decision_mode = str(search.get("decision_mode", "absolute_up_down"))
    if decision_mode == "first_minute_relative":
        up_values = search["t_follow_values"]
        down_values = search["t_reversal_values"]
    else:
        up_values = search["t_up_values"]
        down_values = search["t_down_values"]
    for t_up in up_values:
        for t_down in down_values:
            if decision_mode == "absolute_up_down" and float(t_down) >= float(t_up):
                continue
            if decision_mode == "first_minute_relative" and float(t_down) >= float(t_up):
                continue
            if decision_mode == "first_minute_relative":
                decisions = _decisions_from_first_minute_relative_probability(
                    frame,
                    p_up,
                    t_follow=float(t_up),
                    t_reversal=float(t_down),
                )
            else:
                decisions = _decisions_from_probability(p_up, t_up=float(t_up), t_down=float(t_down))
            metrics = compute_decision_metrics(
                frame["target"],
                p_up,
                decisions,
                selected_t_up=float(t_up),
                selected_t_down=float(t_down),
            )
            metrics.update(compute_reversal_continuation_metrics(frame, decisions))
            row = {"decision_mode": decision_mode, "t_up": float(t_up), "t_down": float(t_down), **metrics}
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

    model = _fit_direction_model(train_x, _training_target(train_frame, train_predictions, config), config)
    train_model_probability = pd.Series(model.predict_proba(train_x)[:, 1], index=train_frame.index)
    validation_model_probability = pd.Series(model.predict_proba(validation_x)[:, 1], index=validation_frame.index)
    train_p_up = _final_p_up_from_model_probability(train_predictions, train_model_probability, config)
    validation_p_up = _final_p_up_from_model_probability(validation_predictions, validation_model_probability, config)
    frontier, best = _search_thresholds(validation_predictions, validation_p_up, config)

    if str(config["threshold_search"].get("decision_mode", "absolute_up_down")) == "first_minute_relative":
        train_decisions = _decisions_from_first_minute_relative_probability(
            train_predictions,
            train_p_up,
            t_follow=float(best["t_up"]),
            t_reversal=float(best["t_down"]),
        )
        validation_decisions = _decisions_from_first_minute_relative_probability(
            validation_predictions,
            validation_p_up,
            t_follow=float(best["t_up"]),
            t_reversal=float(best["t_down"]),
        )
    else:
        train_decisions = _decisions_from_probability(
            train_p_up,
            t_up=float(best["t_up"]),
            t_down=float(best["t_down"]),
        )
        validation_decisions = _decisions_from_probability(
            validation_p_up,
            t_up=float(best["t_up"]),
            t_down=float(best["t_down"]),
        )
    train_metrics = compute_decision_metrics(
        train_predictions["target"],
        train_p_up,
        train_decisions,
        selected_t_up=float(best["t_up"]),
        selected_t_down=float(best["t_down"]),
    )
    train_metrics.update(compute_reversal_continuation_metrics(train_predictions, train_decisions))
    validation_metrics = compute_decision_metrics(
        validation_predictions["target"],
        validation_p_up,
        validation_decisions,
        selected_t_up=float(best["t_up"]),
        selected_t_down=float(best["t_down"]),
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
        "model_target": config.get("model_target", "final_direction"),
        "objective": config["objective"],
        "threshold_search": config["threshold_search"],
        "feature_variant": config["feature_variant"],
        "feature_count": int(train_x.shape[1]),
        "raw_impulse_feature_count": len(base_features),
        "train_window": _window_summary(train_frame),
        "validation_window": _window_summary(validation_frame),
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
    parser = argparse.ArgumentParser(description="Train an impulse-flow direction meta model and search selective thresholds.")
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
