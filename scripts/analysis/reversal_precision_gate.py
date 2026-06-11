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
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.reversal_hybrid import (  # noqa: E402
    compute_decision_metrics,
    compute_reversal_continuation_metrics,
    p_follow_from_direction_probability,
)


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_reversal_precision_gate.yaml")
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


def _load_baseline_inputs(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    manifest = _read_json(experiment_dir / "artifact_manifest.json")
    feature_columns = list(manifest["feature_columns"])
    train_frame = pd.read_parquet(experiment_dir / "development_frame.parquet")
    validation_frame = pd.read_parquet(experiment_dir / "validation_frame.parquet")
    train_predictions = pd.read_parquet(experiment_dir / "train_predictions.parquet")
    validation_predictions = pd.read_parquet(experiment_dir / "validation_predictions.parquet")
    return train_frame, validation_frame, train_predictions, validation_predictions, feature_columns


def _first_minute_up(frame: pd.DataFrame) -> pd.Series:
    if "fm_ret" in frame.columns:
        ret = pd.to_numeric(frame["fm_ret"], errors="coerce")
    elif "first_minute_return" in frame.columns:
        ret = pd.to_numeric(frame["first_minute_return"], errors="coerce")
    elif "ret_1" in frame.columns:
        ret = pd.to_numeric(frame["ret_1"], errors="coerce")
    else:
        raise ValueError("reversal target requires fm_ret, first_minute_return, or ret_1.")
    return ret >= 0.0


def _reversal_target(frame: pd.DataFrame) -> pd.Series:
    return (_first_minute_up(frame) != (frame["target"].astype(int) == 1)).astype(int)


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def _select_features(all_features: list[str], config: dict[str, Any]) -> list[str]:
    safe = [name for name in all_features if name not in LEAKAGE_BLOCKLIST and "target" not in name.lower()]
    variant = config["feature_variant"]
    patterns = _compile_patterns(list(variant.get("patterns", [])))
    selected = [name for name in safe if any(pattern.search(name) for pattern in patterns)]
    if not selected:
        raise ValueError("feature_variant selected no features.")
    return selected


def _fit_model(train_frame: pd.DataFrame, y_train: pd.Series, features: list[str], config: dict[str, Any]) -> lgb.LGBMClassifier:
    params = dict(config.get("model", {}))
    params.pop("type", None)
    params.setdefault("objective", "binary")
    params.setdefault("verbosity", -1)
    positives = int(y_train.sum())
    negatives = int((y_train == 0).sum())
    if positives:
        params.setdefault("scale_pos_weight", negatives / positives)
    model = lgb.LGBMClassifier(**params)
    model.fit(train_frame[features], y_train)
    return model


def _prepare_decision_frame(
    predictions: pd.DataFrame,
    reversal_risk: pd.Series,
    follow_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["p_base"] = pd.to_numeric(frame["p_up"], errors="coerce").astype("float64").clip(0.0, 1.0)
    frame["base_decision"] = frame["decision"].astype("object")
    frame["reversal_risk"] = reversal_risk.reindex(frame.index).astype("float64").clip(0.0, 1.0)
    frame["base_confidence"] = (frame["p_base"] - 0.5).abs()
    if "first_minute_return" in frame.columns:
        frame["first_minute_abs_return"] = pd.to_numeric(frame["first_minute_return"], errors="coerce").abs()
    else:
        frame["first_minute_abs_return"] = np.nan
    if follow_predictions is not None:
        follow = follow_predictions.reindex(frame.index)
        frame["follow_reversal_probability"] = 1.0 - p_follow_from_direction_probability(follow)
    else:
        frame["follow_reversal_probability"] = 0.0
    return frame


def _reversal_decision_from_first_minute(frame: pd.DataFrame) -> pd.Series:
    fm_side = frame["first_minute_side"].astype("object")
    decision = pd.Series("ABSTAIN", index=frame.index, dtype="object")
    decision.loc[fm_side == "YES"] = "DOWN"
    decision.loc[fm_side == "NO"] = "UP"
    return decision


def apply_reversal_precision_gate(
    frame: pd.DataFrame,
    *,
    override_threshold: float,
    max_base_confidence: float,
    min_first_minute_abs_return: float,
    min_follow_reversal_probability: float,
    low_risk_filter_threshold: float,
    low_follow_filter_threshold: float,
    max_continuation_risk: float,
    max_continuation_follow_reversal: float,
    include_abstains: bool,
) -> pd.Series:
    required = {"base_decision", "p_base", "reversal_risk", "first_minute_side"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"precision gate frame is missing required columns: {missing}")

    base_decision = frame["base_decision"].astype("object")
    decision = base_decision.copy()
    risk = frame["reversal_risk"].astype("float64")
    follow_reversal = frame["follow_reversal_probability"].astype("float64")
    confidence = frame["base_confidence"].astype("float64")
    fm_abs_return = frame["first_minute_abs_return"].astype("float64").fillna(0.0)
    reversal_decision = _reversal_decision_from_first_minute(frame)

    base_accepted = base_decision != "ABSTAIN"
    candidate_pool = base_accepted | bool(include_abstains)
    override = (
        candidate_pool
        & reversal_decision.isin(["UP", "DOWN"])
        & (risk >= float(override_threshold))
        & (follow_reversal >= float(min_follow_reversal_probability))
        & (confidence <= float(max_base_confidence))
        & (fm_abs_return >= float(min_first_minute_abs_return))
    )
    decision.loc[override] = reversal_decision.loc[override]

    if low_risk_filter_threshold > 0.0:
        fm_side = frame["first_minute_side"].astype("object")
        side = decision.replace({"UP": "YES", "DOWN": "NO", "ABSTAIN": None})
        reversal_signal = (decision != "ABSTAIN") & side.isin(["YES", "NO"]) & (side != fm_side)
        low_reversal_evidence = (risk < float(low_risk_filter_threshold)) & (
            follow_reversal < float(low_follow_filter_threshold)
        )
        decision.loc[reversal_signal & low_reversal_evidence] = "ABSTAIN"

    if max_continuation_risk < 1.0 or max_continuation_follow_reversal < 1.0:
        fm_side = frame["first_minute_side"].astype("object")
        side = decision.replace({"UP": "YES", "DOWN": "NO", "ABSTAIN": None})
        continuation_signal = (decision != "ABSTAIN") & side.isin(["YES", "NO"]) & (side == fm_side)
        weak_continuation_evidence = (risk > float(max_continuation_risk)) | (
            follow_reversal > float(max_continuation_follow_reversal)
        )
        decision.loc[continuation_signal & weak_continuation_evidence] = "ABSTAIN"

    return decision


def _search_gate(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    search = config["threshold_search"]
    objective = config["objective"]
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for override_threshold in search["override_thresholds"]:
        for max_base_confidence in search["max_base_confidences"]:
            for min_fm_abs_return in search["min_first_minute_abs_returns"]:
                for min_follow_reversal_probability in search["min_follow_reversal_probabilities"]:
                    for low_risk_filter_threshold in search["low_risk_filter_thresholds"]:
                        for low_follow_filter_threshold in search["low_follow_filter_thresholds"]:
                            for max_continuation_risk in search["max_continuation_risks"]:
                                for max_continuation_follow_reversal in search["max_continuation_follow_reversal_probabilities"]:
                                    for include_abstains in search["include_abstains"]:
                                        decisions = apply_reversal_precision_gate(
                                            frame,
                                            override_threshold=float(override_threshold),
                                            max_base_confidence=float(max_base_confidence),
                                            min_first_minute_abs_return=float(min_fm_abs_return),
                                            min_follow_reversal_probability=float(min_follow_reversal_probability),
                                            low_risk_filter_threshold=float(low_risk_filter_threshold),
                                            low_follow_filter_threshold=float(low_follow_filter_threshold),
                                            max_continuation_risk=float(max_continuation_risk),
                                            max_continuation_follow_reversal=float(max_continuation_follow_reversal),
                                            include_abstains=bool(include_abstains),
                                        )
                                        metrics = compute_decision_metrics(
                                            frame["target"],
                                            frame["p_base"],
                                            decisions,
                                            selected_t_up=float(override_threshold),
                                            selected_t_down=float(low_risk_filter_threshold),
                                        )
                                        metrics.update(compute_reversal_continuation_metrics(frame, decisions))
                                        row = {
                                            "override_threshold": float(override_threshold),
                                            "max_base_confidence": float(max_base_confidence),
                                            "min_first_minute_abs_return": float(min_fm_abs_return),
                                            "min_follow_reversal_probability": float(min_follow_reversal_probability),
                                            "low_risk_filter_threshold": float(low_risk_filter_threshold),
                                            "low_follow_filter_threshold": float(low_follow_filter_threshold),
                                            "max_continuation_risk": float(max_continuation_risk),
                                            "max_continuation_follow_reversal": float(max_continuation_follow_reversal),
                                            "include_abstains": bool(include_abstains),
                                            **metrics,
                                        }
                                        records.append(row)
                                        if (
                                            metrics["coverage"] >= float(objective["min_coverage"])
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
            row["coverage"] >= float(objective["min_coverage"]),
            row["selection_score"],
            row["coverage"],
        ),
    )
    best = {
        **best,
        "constraint_satisfied": bool(eligible),
        "objective": "maximize reversal_accepted_accuracy subject to coverage and continuation constraints",
        "hard_constraints": {
            "min_coverage": float(objective["min_coverage"]),
            "min_continuation_accepted_accuracy": float(objective["min_continuation_accepted_accuracy"]),
            "min_reversal_accepted_accuracy": float(objective["min_reversal_accepted_accuracy"]),
        },
        "fallback_reason": None if eligible else "no candidate satisfied all reversal precision constraints",
    }
    return pd.DataFrame.from_records(records), best


def _alert_metrics(y_reversal: pd.Series, reversal_risk: pd.Series, threshold: float) -> dict[str, float]:
    y = y_reversal.astype(int)
    risk = reversal_risk.astype("float64").clip(0.0, 1.0)
    alerts = risk >= threshold
    alert_count = int(alerts.sum())
    false_alerts = int(((y == 0) & alerts).sum())
    continuation_count = int((y == 0).sum())
    return {
        "reversal_alert_count": float(alert_count),
        "reversal_alert_precision": float(precision_score(y, alerts.astype(int), zero_division=0)),
        "reversal_alert_recall": float(recall_score(y, alerts.astype(int), zero_division=0)),
        "continuation_false_alert_rate": float(false_alerts / continuation_count) if continuation_count else 0.0,
        "alert_reversal_share": float(y.loc[alerts].mean()) if alert_count else 0.0,
        "reversal_risk_roc_auc": float(roc_auc_score(y, risk)) if y.nunique() == 2 else 0.0,
        "reversal_risk_average_precision": float(average_precision_score(y, risk)) if y.nunique() == 2 else 0.0,
    }


def _window_summary(frame: pd.DataFrame) -> dict[str, str | int]:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True) if "timestamp" in frame.columns else pd.to_datetime(frame["market_t0"], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamps.min()), "end": str(timestamps.max())}


def _required_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "sample_count",
        "coverage",
        "precision_up",
        "precision_down",
        "balanced_precision",
        "all_sample_accuracy",
        "accepted_sample_accuracy",
        "share_up_predictions",
        "share_down_predictions",
        "selected_t_up",
        "selected_t_down",
        "accepted_count",
        "up_prediction_count",
        "down_prediction_count",
        "roc_auc",
        "brier_score",
        "log_loss",
        "utility",
        "downside_risk",
        "selection_score",
        "up_signal_count",
        "down_signal_count",
        "total_signal_count",
        "signal_coverage",
        "overall_signal_accuracy",
    ]
    return {key: metrics[key] for key in keys}


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    baseline_experiment = Path(config["baseline_experiment"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_baseline_inputs(baseline_experiment)
    follow_train_predictions = None
    follow_validation_predictions = None
    if config.get("follow_experiment"):
        follow_experiment = Path(config["follow_experiment"])
        follow_train_predictions = pd.read_parquet(follow_experiment / "train_predictions.parquet")
        follow_validation_predictions = pd.read_parquet(follow_experiment / "validation_predictions.parquet")
    features = _select_features(all_features, config)
    y_train_reversal = _reversal_target(train_frame)
    y_valid_reversal = _reversal_target(validation_frame)
    train_scope = str(config.get("model", {}).get("train_scope", "all"))
    fit_frame = train_frame
    fit_target = y_train_reversal
    if train_scope == "base_continuation":
        train_side = train_predictions["decision"].replace({"UP": "YES", "DOWN": "NO", "ABSTAIN": None})
        fit_mask = (train_predictions["decision"] != "ABSTAIN") & (train_side == train_predictions["first_minute_side"])
        fit_frame = train_frame.loc[fit_mask]
        fit_target = y_train_reversal.loc[fit_mask]
    elif train_scope != "all":
        raise ValueError(f"unsupported model.train_scope: {train_scope}")
    model = _fit_model(fit_frame, fit_target, features, config)
    train_risk = pd.Series(model.predict_proba(train_frame[features])[:, 1], index=train_frame.index)
    valid_risk = pd.Series(model.predict_proba(validation_frame[features])[:, 1], index=validation_frame.index)

    train_decision_frame = _prepare_decision_frame(train_predictions, train_risk, follow_train_predictions)
    validation_decision_frame = _prepare_decision_frame(validation_predictions, valid_risk, follow_validation_predictions)
    frontier, best = _search_gate(validation_decision_frame, config)

    train_decisions = apply_reversal_precision_gate(
        train_decision_frame,
        override_threshold=float(best["override_threshold"]),
        max_base_confidence=float(best["max_base_confidence"]),
        min_first_minute_abs_return=float(best["min_first_minute_abs_return"]),
        min_follow_reversal_probability=float(best["min_follow_reversal_probability"]),
        low_risk_filter_threshold=float(best["low_risk_filter_threshold"]),
        low_follow_filter_threshold=float(best["low_follow_filter_threshold"]),
        max_continuation_risk=float(best["max_continuation_risk"]),
        max_continuation_follow_reversal=float(best["max_continuation_follow_reversal"]),
        include_abstains=bool(best["include_abstains"]),
    )
    validation_decisions = apply_reversal_precision_gate(
        validation_decision_frame,
        override_threshold=float(best["override_threshold"]),
        max_base_confidence=float(best["max_base_confidence"]),
        min_first_minute_abs_return=float(best["min_first_minute_abs_return"]),
        min_follow_reversal_probability=float(best["min_follow_reversal_probability"]),
        low_risk_filter_threshold=float(best["low_risk_filter_threshold"]),
        low_follow_filter_threshold=float(best["low_follow_filter_threshold"]),
        max_continuation_risk=float(best["max_continuation_risk"]),
        max_continuation_follow_reversal=float(best["max_continuation_follow_reversal"]),
        include_abstains=bool(best["include_abstains"]),
    )
    train_metrics = compute_decision_metrics(
        train_decision_frame["target"],
        train_decision_frame["p_base"],
        train_decisions,
        selected_t_up=float(best["override_threshold"]),
        selected_t_down=float(best["low_risk_filter_threshold"]),
    )
    train_metrics.update(compute_reversal_continuation_metrics(train_decision_frame, train_decisions))
    validation_metrics = compute_decision_metrics(
        validation_decision_frame["target"],
        validation_decision_frame["p_base"],
        validation_decisions,
        selected_t_up=float(best["override_threshold"]),
        selected_t_down=float(best["low_risk_filter_threshold"]),
    )
    validation_metrics.update(compute_reversal_continuation_metrics(validation_decision_frame, validation_decisions))
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
        "primary_metric": "validation reversal_accepted_accuracy with coverage and continuation constraints",
        "mode": config["mode"],
        "objective": config["objective"],
        "threshold_search": config["threshold_search"],
        "feature_variant": config["feature_variant"],
        "feature_count": len(features),
        "train_window": _window_summary(train_frame),
        "validation_window": _window_summary(validation_frame),
        "baseline_validation_metrics": baseline_metrics,
        "best_gate": best,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "required_train_metrics": _required_metric_subset(train_metrics),
        "required_validation_metrics": _required_metric_subset(validation_metrics),
        "alert_metrics": _alert_metrics(y_valid_reversal, valid_risk, float(best["override_threshold"])),
        "frontier_path": str(frontier_path),
        "accepted": bool(best["constraint_satisfied"]),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a high-precision reversal override gate.")
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
