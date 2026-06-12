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
    compute_reversal_continuation_metrics,
    route_conflict_margin_hybrid,
)


DEFAULT_CONFIG_PATH = Path("experiments/configs/20260612_catboost_continuation_reversal_hybrid_router.yaml")
BASELINE_SELECTION_SCORE = 0.5748509217
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


def _actual_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    y = predictions["target"].astype(int)
    actual_direction = pd.Series(np.where(y == 1, "UP", "DOWN"), index=predictions.index, dtype="object")
    actual_side = pd.Series(np.where(y == 1, "YES", "NO"), index=predictions.index, dtype="object")
    fm_side = predictions["first_minute_side"].astype("object")
    actual_regime = pd.Series(
        np.where(fm_side == actual_side, "continuation", "reversal"),
        index=predictions.index,
        dtype="object",
    )
    timestamps = pd.to_datetime(predictions["timestamp"], utc=True)
    session = pd.cut(timestamps.dt.hour, bins=[-1, 7, 15, 23], labels=["asia", "europe", "us"]).astype(str)
    return pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "day": timestamps.dt.dayofweek.astype(int),
            "session": session,
            "first_minute_side": fm_side,
            "actual_direction": actual_direction,
            "actual_regime": actual_regime,
        },
        index=predictions.index,
    )


def _make_audit(
    predictions: pd.DataFrame,
    *,
    continuation_p_up: pd.Series,
    continuation_decision: pd.Series,
    reversal_p_up: pd.Series,
    reversal_decision: pd.Series,
    routed: pd.DataFrame,
    p_continuation: pd.Series | None = None,
) -> pd.DataFrame:
    y = predictions["target"].astype(int)
    final_decision = routed["final_decision"].astype("object")
    final_accept = final_decision != "ABSTAIN"
    final_correct = ((final_decision == "UP") == (y == 1)) & final_accept
    audit = _actual_columns(predictions)
    audit["continuation_p_up"] = continuation_p_up.astype("float64")
    audit["continuation_decision"] = continuation_decision.astype("object")
    audit["continuation_accept"] = continuation_decision.astype("object") != "ABSTAIN"
    audit["continuation_confidence"] = (continuation_p_up.astype("float64").clip(0.0, 1.0) - 0.5).abs()
    audit["reversal_p_up"] = reversal_p_up.astype("float64")
    audit["reversal_decision"] = reversal_decision.astype("object")
    audit["reversal_accept"] = reversal_decision.astype("object") != "ABSTAIN"
    audit["reversal_confidence"] = (reversal_p_up.astype("float64").clip(0.0, 1.0) - 0.5).abs()
    if p_continuation is not None:
        audit["p_continuation"] = p_continuation.astype("float64").clip(0.0, 1.0)
        audit["p_reversal"] = 1.0 - audit["p_continuation"]
    audit["final_decision"] = final_decision
    audit["final_accept"] = final_accept
    audit["final_correct"] = final_correct
    audit["used_expert"] = routed["used_expert"].astype("object")
    audit["routing_reason"] = routed["routing_reason"].astype("object")
    return audit


def _final_probability(
    continuation_p_up: pd.Series,
    reversal_p_up: pd.Series,
    used_expert: pd.Series,
) -> pd.Series:
    probability = ((continuation_p_up.astype("float64") + reversal_p_up.astype("float64")) / 2.0).clip(0.0, 1.0)
    probability.loc[used_expert == "continuation"] = continuation_p_up.loc[used_expert == "continuation"]
    probability.loc[used_expert == "reversal"] = reversal_p_up.loc[used_expert == "reversal"]
    return probability.clip(0.0, 1.0)


def _router_metrics(
    predictions: pd.DataFrame,
    continuation_p_up: pd.Series,
    reversal_p_up: pd.Series,
    routed: pd.DataFrame,
    *,
    selected_t_up: float,
    selected_t_down: float,
) -> dict[str, float]:
    final_probability = _final_probability(continuation_p_up, reversal_p_up, routed["used_expert"])
    decisions = routed["final_decision"].astype("object")
    metrics = compute_decision_metrics(
        predictions["target"],
        final_probability,
        decisions,
        selected_t_up=selected_t_up,
        selected_t_down=selected_t_down,
    )
    metrics.update(compute_reversal_continuation_metrics(predictions, decisions))
    reason_counts = routed["routing_reason"].value_counts()
    used_counts = routed["used_expert"].value_counts()
    for field in [
        "conflict_abstain_count",
        "both_abstain_count",
        "conflict_cont_win_count",
        "conflict_rev_win_count",
        "cont_only_count",
        "rev_only_count",
    ]:
        reason = field.removesuffix("_count")
        metrics[field] = float(reason_counts.get(reason, 0))
    metrics["cont_model_used_count"] = float(used_counts.get("continuation", 0))
    metrics["rev_model_used_count"] = float(used_counts.get("reversal", 0))
    metrics["abstain_count"] = float(used_counts.get("abstain", 0))
    metrics["balanced_score"] = float(
        metrics["accepted_sample_accuracy"]
        - 0.4 * abs(metrics["continuation_accepted_accuracy"] - metrics["reversal_accepted_accuracy"])
        - 0.3 * metrics["downside_risk"]
    )
    metrics["router_score"] = float(
        metrics["utility"]
        - 0.5 * max(0.0, 0.60 - metrics["continuation_accepted_accuracy"])
        - 0.7 * max(0.0, 0.55 - metrics["reversal_accepted_accuracy"])
        - 0.3 * metrics["downside_risk"]
    )
    return metrics


def _constraint_satisfied(metrics: dict[str, float], config: dict[str, Any]) -> bool:
    return bool(
        metrics["coverage"] >= float(config["objective"]["min_coverage"])
        and metrics["accepted_count"] >= float(config["objective"]["minimum_accepted_count"])
        and metrics["continuation_accepted_count"] > 0.0
        and metrics["reversal_accepted_count"] > 0.0
        and metrics["utility"] > 0.0
        and metrics["accepted_sample_accuracy"] > 0.50
    )


def _best(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    constrained = [row for row in records if row["coverage_constraint_satisfied"]]
    pool = constrained if constrained else records
    best = max(
        pool,
        key=lambda row: (
            row["selection_score"],
            row["utility"],
            row["coverage"],
            row["accepted_count"],
            row["balanced_score"],
        ),
    )
    best["fallback_reason"] = None if constrained else "no candidate satisfied coverage/router constraints"
    return best


def _gate_features(frame: pd.DataFrame, predictions: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    timestamps = pd.to_datetime(predictions["timestamp"], utc=True)
    output = frame[base_features].copy()
    output["gate_first_minute_side_yes"] = (predictions["first_minute_side"].astype("object") == "YES").astype(int)
    output["gate_dayofweek"] = timestamps.dt.dayofweek.astype(int)
    output["gate_hour"] = timestamps.dt.hour.astype(int)
    output["gate_session_id"] = pd.cut(timestamps.dt.hour, bins=[-1, 7, 15, 23], labels=[0, 1, 2]).astype(int)
    return output


def _gate_target(predictions: pd.DataFrame) -> pd.Series:
    fm_direction = (predictions["first_minute_side"].astype("object") == "YES").astype(int)
    actual_direction = predictions["target"].astype(int)
    return (fm_direction == actual_direction).astype(int)


def _gate_route(
    p_continuation: pd.Series,
    continuation_decision: pd.Series,
    reversal_decision: pd.Series,
    *,
    tau_cont: float,
    tau_rev: float,
) -> pd.DataFrame:
    p_cont = p_continuation.astype("float64").clip(0.0, 1.0)
    p_rev = 1.0 - p_cont
    final_decision = pd.Series("ABSTAIN", index=p_cont.index, dtype="object")
    used_expert = pd.Series("abstain", index=p_cont.index, dtype="object")
    routing_reason = pd.Series("gate_abstain", index=p_cont.index, dtype="object")
    use_cont = p_cont >= float(tau_cont)
    use_rev = (~use_cont) & (p_rev >= float(tau_rev))
    cont_accept = continuation_decision.astype("object") != "ABSTAIN"
    rev_accept = reversal_decision.astype("object") != "ABSTAIN"

    cont_take = use_cont & cont_accept
    rev_take = use_rev & rev_accept
    final_decision.loc[cont_take] = continuation_decision.loc[cont_take]
    used_expert.loc[cont_take] = "continuation"
    routing_reason.loc[cont_take] = "gate_cont"
    final_decision.loc[rev_take] = reversal_decision.loc[rev_take]
    used_expert.loc[rev_take] = "reversal"
    routing_reason.loc[rev_take] = "gate_rev"
    return pd.DataFrame({"final_decision": final_decision, "used_expert": used_expert, "routing_reason": routing_reason})


def _window_summary(frame: pd.DataFrame) -> dict[str, str | int]:
    column = "timestamp" if "timestamp" in frame.columns else "market_t0"
    timestamps = pd.to_datetime(frame[column], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamps.min()), "end": str(timestamps.max())}


def _train_experts(
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    train_predictions: pd.DataFrame,
    features: list[str],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    continuation_model = CatBoostClassifier(**config["expert_model"])
    continuation_model.fit(train_frame[features], train_frame["target"].astype(int))
    cont_train = pd.Series(continuation_model.predict_proba(train_frame[features])[:, 1], index=train_frame.index).clip(0.0, 1.0)
    cont_val = pd.Series(continuation_model.predict_proba(validation_frame[features])[:, 1], index=validation_frame.index).clip(0.0, 1.0)

    final_side = pd.Series(np.where(train_predictions["target"].astype(int) == 1, "YES", "NO"), index=train_predictions.index)
    reversal_mask = train_predictions["first_minute_side"].astype("object").isin(["YES", "NO"]) & (
        train_predictions["first_minute_side"].astype("object") != final_side
    )
    sample_weight = pd.Series(1.0, index=train_frame.index)
    sample_weight.loc[reversal_mask] = float(config["reversal_expert"]["sample_weight"])
    reversal_model = CatBoostClassifier(**config["expert_model"])
    reversal_model.fit(train_frame[features], train_frame["target"].astype(int), sample_weight=sample_weight)
    rev_train = pd.Series(reversal_model.predict_proba(train_frame[features])[:, 1], index=train_frame.index).clip(0.0, 1.0)
    rev_val = pd.Series(reversal_model.predict_proba(validation_frame[features])[:, 1], index=validation_frame.index).clip(0.0, 1.0)
    return cont_train, cont_val, rev_train, rev_val


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, validation_frame, train_predictions, validation_predictions, all_features = _load_inputs(
        Path(config["baseline_experiment"])
    )
    features = _select_features(all_features, list(config["feature_set"]["patterns"]))
    gate_base_features = _select_features(all_features, list(config["gate_feature_set"]["patterns"]))
    leakage_features = sorted(set(features + gate_base_features).intersection(LEAKAGE_BLOCKLIST))
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
    cont_t_up, cont_t_down = _selected_thresholds(continuation_thresholds)
    rev_t_up = float(config["reversal_expert"]["t_up"])
    rev_t_down = float(config["reversal_expert"]["t_down"])

    train_regimes = _regimes(train_predictions)
    validation_regimes = _regimes(validation_predictions)
    cont_train_decision = apply_continuation_expert_decision(
        cont_train_p,
        train_predictions["first_minute_side"],
        train_regimes,
        continuation_thresholds,
    )
    cont_val_decision = apply_continuation_expert_decision(
        cont_val_p,
        validation_predictions["first_minute_side"],
        validation_regimes,
        continuation_thresholds,
    )
    rev_train_decision = apply_reversal_only_decision(
        rev_train_p,
        train_predictions["first_minute_side"],
        t_up=rev_t_up,
        t_down=rev_t_down,
    )
    rev_val_decision = apply_reversal_only_decision(
        rev_val_p,
        validation_predictions["first_minute_side"],
        t_up=rev_t_up,
        t_down=rev_t_down,
    )

    conflict_records: list[dict[str, Any]] = []
    conflict_audits: dict[float, pd.DataFrame] = {}
    for margin in [float(value) for value in config["conflict_margin_values"]]:
        routed = route_conflict_margin_hybrid(
            cont_val_p,
            cont_val_decision,
            rev_val_p,
            rev_val_decision,
            conflict_margin=margin,
        )
        metrics = _router_metrics(
            validation_predictions,
            cont_val_p,
            rev_val_p,
            routed,
            selected_t_up=margin,
            selected_t_down=margin,
        )
        row = {"experiment": "conflict_margin", "conflict_margin": margin, **metrics}
        row["coverage_constraint_satisfied"] = _constraint_satisfied(metrics, config)
        row["official_acceptance_metric"] = "validation_selection_score_with_coverage_ge_0.70"
        conflict_records.append(row)
        conflict_audits[margin] = _make_audit(
            validation_predictions,
            continuation_p_up=cont_val_p,
            continuation_decision=cont_val_decision,
            reversal_p_up=rev_val_p,
            reversal_decision=rev_val_decision,
            routed=routed,
        )
    conflict_summary = pd.DataFrame.from_records(conflict_records)
    conflict_summary.to_csv(output_dir / "experiment3_conflict_margin_summary.csv", index=False)
    conflict_best = _best(conflict_records, config)
    conflict_best["baseline_selection_score"] = BASELINE_SELECTION_SCORE
    conflict_best["improved_over_accepted_baseline"] = bool(
        conflict_best["coverage_constraint_satisfied"] and conflict_best["selection_score"] > BASELINE_SELECTION_SCORE
    )
    (output_dir / "experiment3_conflict_margin_best_metrics.json").write_text(
        json.dumps(conflict_best, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    conflict_audits[float(conflict_best["conflict_margin"])].to_csv(output_dir / "experiment3_conflict_margin_audit.csv", index=False)

    gate_model = CatBoostClassifier(**config["gate_model"])
    gate_train_x = _gate_features(train_frame, train_predictions, gate_base_features)
    gate_validation_x = _gate_features(validation_frame, validation_predictions, gate_base_features)
    gate_model.fit(gate_train_x, _gate_target(train_predictions))
    gate_train_p_cont = pd.Series(gate_model.predict_proba(gate_train_x)[:, 1], index=train_frame.index).clip(0.0, 1.0)
    gate_val_p_cont = pd.Series(gate_model.predict_proba(gate_validation_x)[:, 1], index=validation_frame.index).clip(0.0, 1.0)

    gate_records: list[dict[str, Any]] = []
    gate_audits: dict[tuple[float, float], pd.DataFrame] = {}
    for tau_cont in [float(value) for value in config["gate_search"]["tau_cont_values"]]:
        for tau_rev in [float(value) for value in config["gate_search"]["tau_rev_values"]]:
            routed = _gate_route(
                gate_val_p_cont,
                cont_val_decision,
                rev_val_decision,
                tau_cont=tau_cont,
                tau_rev=tau_rev,
            )
            metrics = _router_metrics(
                validation_predictions,
                cont_val_p,
                rev_val_p,
                routed,
                selected_t_up=tau_cont,
                selected_t_down=tau_rev,
            )
            row = {"experiment": "gate_hybrid", "tau_cont": tau_cont, "tau_rev": tau_rev, **metrics}
            row["coverage_constraint_satisfied"] = _constraint_satisfied(metrics, config)
            row["official_acceptance_metric"] = "validation_selection_score_with_coverage_ge_0.70"
            gate_records.append(row)
            gate_audits[(tau_cont, tau_rev)] = _make_audit(
                validation_predictions,
                continuation_p_up=cont_val_p,
                continuation_decision=cont_val_decision,
                reversal_p_up=rev_val_p,
                reversal_decision=rev_val_decision,
                routed=routed,
                p_continuation=gate_val_p_cont,
            )
    gate_summary = pd.DataFrame.from_records(gate_records)
    gate_summary.to_csv(output_dir / "experiment4_gate_hybrid_summary.csv", index=False)
    gate_best = _best(gate_records, config)
    gate_best["baseline_selection_score"] = BASELINE_SELECTION_SCORE
    gate_best["improved_over_accepted_baseline"] = bool(
        gate_best["coverage_constraint_satisfied"] and gate_best["selection_score"] > BASELINE_SELECTION_SCORE
    )
    (output_dir / "experiment4_gate_hybrid_best_metrics.json").write_text(
        json.dumps(gate_best, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    gate_audits[(float(gate_best["tau_cont"]), float(gate_best["tau_rev"]))].to_csv(
        output_dir / "experiment4_gate_hybrid_audit.csv",
        index=False,
    )

    best_validation = max(
        [conflict_best, gate_best],
        key=lambda row: (
            row["coverage_constraint_satisfied"],
            row["selection_score"],
            row["utility"],
            row["coverage"],
        ),
    )
    if best_validation["experiment"] == "conflict_margin":
        train_routed = route_conflict_margin_hybrid(
            cont_train_p,
            cont_train_decision,
            rev_train_p,
            rev_train_decision,
            conflict_margin=float(best_validation["conflict_margin"]),
        )
        train_selected_t_up = float(best_validation["conflict_margin"])
        train_selected_t_down = float(best_validation["conflict_margin"])
    else:
        train_routed = _gate_route(
            gate_train_p_cont,
            cont_train_decision,
            rev_train_decision,
            tau_cont=float(best_validation["tau_cont"]),
            tau_rev=float(best_validation["tau_rev"]),
        )
        train_selected_t_up = float(best_validation["tau_cont"])
        train_selected_t_down = float(best_validation["tau_rev"])
    train_metrics = _router_metrics(
        train_predictions,
        cont_train_p,
        rev_train_p,
        train_routed,
        selected_t_up=train_selected_t_up,
        selected_t_down=train_selected_t_down,
    )
    improved_over_baseline = bool(
        best_validation["coverage_constraint_satisfied"] and best_validation["selection_score"] > BASELINE_SELECTION_SCORE
    )
    metadata = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "continuation_expert_source": config["continuation_expert"]["source"],
        "continuation_expert_reason": "highest continuation_accepted_accuracy among current experiments",
        "reversal_expert_source": config["reversal_expert"]["source"],
        "reversal_expert_reason": "highest reversal_accepted_accuracy among current reversal experiments",
        "hybrid_goal": "combine continuation expert and reversal expert using routing instead of probability averaging",
        "primary_metric": "validation selection_score with coverage >= 0.70",
        "objective": config["objective"],
        "baseline_selection_score": BASELINE_SELECTION_SCORE,
        "best_experiment": best_validation["experiment"],
        "coverage_constraint_satisfied": bool(best_validation["coverage_constraint_satisfied"]),
        "improved_over_accepted_baseline": improved_over_baseline,
        "train_window": _window_summary(train_frame),
        "validation_window": _window_summary(validation_frame),
        "feature_count": len(features),
        "gate_feature_count": len(gate_base_features) + 4,
        "feature_set_patterns": list(config["feature_set"]["patterns"]),
        "gate_feature_set_patterns": list(config["gate_feature_set"]["patterns"]),
        "leakage_blocklist": sorted(LEAKAGE_BLOCKLIST),
        "leakage_features_selected": leakage_features,
        "label_source": "polymarket_resolved",
        "label_version": "polymarket_resolved_gamma_v1",
        "deploy_training_mode": "not_regenerated",
        "offline_validation_metric_source": "validation hybrid router evaluation only",
        "train_metrics": {field: float(train_metrics[field]) for field in OBJECTIVE_METRIC_FIELDS},
        "validation_metrics": {field: float(best_validation[field]) for field in OBJECTIVE_METRIC_FIELDS},
        "diagnostic_scores": {
            "router_score": float(best_validation["router_score"]),
            "balanced_score": float(best_validation["balanced_score"]),
        },
    }
    report = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "primary_metric": "validation selection_score with coverage >= 0.70",
        "mode": config["mode"],
        "objective": config["objective"],
        "train_window": metadata["train_window"],
        "validation_window": metadata["validation_window"],
        "train_metrics": metadata["train_metrics"],
        "validation_metrics": metadata["validation_metrics"],
        "best_experiment": metadata["best_experiment"],
        "coverage_constraint_satisfied": metadata["coverage_constraint_satisfied"],
        "improved_over_accepted_baseline": metadata["improved_over_accepted_baseline"],
        "baseline_selection_score": BASELINE_SELECTION_SCORE,
        "diagnostic_scores": metadata["diagnostic_scores"],
        "output_files": {
            "experiment3_conflict_margin_summary": str(output_dir / "experiment3_conflict_margin_summary.csv"),
            "experiment3_conflict_margin_best_metrics": str(output_dir / "experiment3_conflict_margin_best_metrics.json"),
            "experiment3_conflict_margin_audit": str(output_dir / "experiment3_conflict_margin_audit.csv"),
            "experiment4_gate_hybrid_summary": str(output_dir / "experiment4_gate_hybrid_summary.csv"),
            "experiment4_gate_hybrid_best_metrics": str(output_dir / "experiment4_gate_hybrid_best_metrics.json"),
            "experiment4_gate_hybrid_audit": str(output_dir / "experiment4_gate_hybrid_audit.csv"),
            "hybrid_experiment_metadata": str(output_dir / "hybrid_experiment_metadata.json"),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "hybrid_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuation + reversal hybrid router experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    metadata = run(args.config)
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "best_experiment": metadata["best_experiment"],
                "coverage_constraint_satisfied": metadata["coverage_constraint_satisfied"],
                "selection_score": metadata["validation_metrics"]["selection_score"],
                "utility": metadata["validation_metrics"]["utility"],
                "accepted_sample_accuracy": metadata["validation_metrics"]["accepted_sample_accuracy"],
                "coverage": metadata["validation_metrics"]["coverage"],
                "improved_over_accepted_baseline": metadata["improved_over_accepted_baseline"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
