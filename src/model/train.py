from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.calibration.base import CalibrationPlugin
from src.calibration.none import NoCalibration
from src.calibration.registry import create_calibration_plugin
from src.core.config import Settings
from src.core.constants import (
    DEFAULT_ABS_RETURN_COLUMN,
    DEFAULT_STAGE1_PROBABILITY_COLUMN,
    DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN,
    DEFAULT_STAGE2_PREDICTED_RETURN_COLUMN,
    DEFAULT_STAGE2_TARGET_COLUMN,
    DEFAULT_TARGET_COLUMN,
)
from src.data.dataset_builder import TrainingFrame
from src.model.base import ModelPlugin
from src.model.evaluation import (
    WalkForwardFoldResult,
    compute_binary_classification_metrics,
    compute_ks_distance,
    compute_return_direction_metrics,
    compute_stage1_coverage,
    compute_two_stage_end_to_end_metrics,
    purged_chronological_time_window_split,
    summarize_walk_forward,
)
from src.model.registry import create_model_plugin


@dataclass(frozen=True)
class TwoStageTrainingArtifacts:
    stage1_model: ModelPlugin
    stage1_calibrator: CalibrationPlugin
    stage2_model: ModelPlugin
    stage2_calibrator: CalibrationPlugin
    feature_columns: list[str]
    stage2_feature_columns: list[str]
    stage1_threshold: float
    up_threshold: float
    down_threshold: float
    margin_threshold: float
    train_metrics: dict[str, dict[str, Any]]
    train_window: dict[str, Any]
    validation_window: dict[str, Any]
    validation_metrics: dict[str, dict[str, Any]]
    walk_forward_summary: dict[str, float | int]
    walk_forward_results: list[WalkForwardFoldResult]
    walk_forward_fold_details: list[dict[str, Any]]
    base_rate: float
    threshold_search: dict[str, Any]
    stage1_probability_summary: dict[str, dict[str, Any]]
    stage1_probability_reference: dict[str, Any]
    stage2_direction_reference: dict[str, Any]


def _slice_training_frame(training: TrainingFrame, frame_slice: slice) -> TrainingFrame:
    return TrainingFrame(
        frame=training.frame.iloc[frame_slice].reset_index(drop=True),
        feature_columns=training.feature_columns,
        target_column=training.target_column,
        sample_weight_column=training.sample_weight_column,
    )


def _make_training_frame_from_cached_split(frame: pd.DataFrame) -> TrainingFrame:
    feature_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "asset",
            "horizon",
            "grid_id",
            "grid_t0",
            "is_grid_t0",
            "feature_version",
            "label_version",
            DEFAULT_TARGET_COLUMN,
            DEFAULT_STAGE2_TARGET_COLUMN,
            DEFAULT_ABS_RETURN_COLUMN,
            "signed_return",
            DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN,
        }
    ]
    return TrainingFrame(
        frame=frame.reset_index(drop=True),
        feature_columns=feature_columns,
        target_column=DEFAULT_TARGET_COLUMN,
        sample_weight_column=None,
    )


def load_cached_training_split(
    *,
    development_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
) -> tuple[TrainingFrame, TrainingFrame]:
    return (
        _make_training_frame_from_cached_split(development_frame),
        _make_training_frame_from_cached_split(validation_frame),
    )


def split_training_frame(
    training: TrainingFrame,
    *,
    validation_window_days: int = 30,
    purge_rows: int = 1,
) -> tuple[TrainingFrame, TrainingFrame]:
    _, _, _, _, split = purged_chronological_time_window_split(
        training,
        validation_window_days=validation_window_days,
        purge_rows=purge_rows,
    )
    return (
        _slice_training_frame(training, split.train_slice),
        _slice_training_frame(training, split.valid_slice),
    )


def _replace_training_columns(
    training: TrainingFrame,
    *,
    frame: pd.DataFrame | None = None,
    target_column: str | None = None,
    sample_weight_column: str | None = None,
    feature_columns: list[str] | None = None,
    reset_index: bool = True,
) -> TrainingFrame:
    base_frame = frame if frame is not None else training.frame
    return TrainingFrame(
        frame=base_frame.reset_index(drop=True) if reset_index else base_frame.copy(),
        feature_columns=feature_columns or training.feature_columns,
        target_column=target_column or training.target_column,
        sample_weight_column=sample_weight_column,
    )


def _build_stage1_training_frame(training: TrainingFrame, settings: Settings) -> TrainingFrame:
    tau = float(settings.labels.two_stage.active_return_threshold)
    frame = training.frame.copy()
    frame["stage1_target"] = (frame[DEFAULT_ABS_RETURN_COLUMN] > tau).astype(int)
    if DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN not in frame.columns:
        frame[DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN] = 1.0
    return _replace_training_columns(
        training,
        frame=frame,
        target_column="stage1_target",
        sample_weight_column=DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN,
    )


def _build_stage2_training_frame(
    training: TrainingFrame,
    stage1_probabilities: pd.Series,
    *,
    stage1_threshold: float,
) -> TrainingFrame:
    frame = training.frame.copy()
    frame[DEFAULT_STAGE1_PROBABILITY_COLUMN] = stage1_probabilities.reindex(frame.index)
    selected = frame.loc[frame[DEFAULT_STAGE1_PROBABILITY_COLUMN] >= stage1_threshold].copy()
    return _replace_training_columns(
        training,
        frame=selected,
        target_column=DEFAULT_STAGE2_TARGET_COLUMN,
        sample_weight_column=None,
        feature_columns=[*training.feature_columns, DEFAULT_STAGE1_PROBABILITY_COLUMN],
        reset_index=False,
    )


def _compute_stage1_scale_pos_weight(training: TrainingFrame) -> float:
    class_counts = training.y.astype(int).value_counts()
    positive_count = int(class_counts.get(1, 0))
    negative_count = int(class_counts.get(0, 0))
    if positive_count <= 0 or negative_count <= 0:
        return 1.0
    return float(negative_count / positive_count)


def _resolve_model_plugin_params(
    training: TrainingFrame,
    settings: Settings,
    *,
    stage: str,
) -> dict[str, Any] | None:
    plugin_name = settings.model.resolve_plugin(stage=stage)
    if not plugin_name.startswith("lightgbm"):
        return None
    if stage == "stage1":
        return {"scale_pos_weight": _compute_stage1_scale_pos_weight(training), "objective": "binary"}
    return {
        "objective": "quantile",
        "alpha": 0.5,
    }


def _select_stage1_calibrator(
    settings: Settings,
    raw_probabilities: pd.Series,
    y_true: pd.Series,
) -> CalibrationPlugin:
    plugin_name = settings.calibration.resolve_plugin(stage="stage1")
    if plugin_name == "none":
        return NoCalibration()
    if raw_probabilities.empty or len(raw_probabilities) < 100 or y_true.nunique() < 2:
        return NoCalibration()

    base_metrics = compute_binary_classification_metrics(y_true, raw_probabilities)
    calibrator = create_calibration_plugin(settings, stage="stage1")
    calibrator.fit(raw_probabilities, y_true)
    calibrated_probabilities = calibrator.transform(raw_probabilities)
    calibrated_metrics = compute_binary_classification_metrics(y_true, calibrated_probabilities)

    if (
        calibrated_metrics["log_loss"] < base_metrics["log_loss"]
        and calibrated_metrics["brier_score"] <= base_metrics["brier_score"]
    ):
        return calibrator
    return NoCalibration()


def _fit_model(
    training: TrainingFrame,
    settings: Settings,
    *,
    stage: str,
    validation: TrainingFrame | None = None,
) -> ModelPlugin:
    plugin_params = _resolve_model_plugin_params(training, settings, stage=stage)
    model = create_model_plugin(settings, stage=stage, plugin_params=plugin_params)
    model.fit(
        training.X,
        training.y.astype(int) if stage == "stage1" else training.y.astype(float),
        X_valid=validation.X if validation is not None and not validation.frame.empty else None,
        y_valid=(
            validation.y.astype(int) if stage == "stage1" else validation.y.astype(float)
        ) if validation is not None and not validation.frame.empty else None,
        sample_weight=training.sample_weight,
        sample_weight_valid=validation.sample_weight if validation is not None and not validation.frame.empty else None,
    )
    return model


def _summarize_probability_series(probabilities: pd.Series) -> dict[str, float]:
    return {
        "mean": float(probabilities.mean()),
        "std": float(probabilities.std(ddof=0)),
        "p10": float(probabilities.quantile(0.10)),
        "p50": float(probabilities.quantile(0.50)),
        "p90": float(probabilities.quantile(0.90)),
    }


def _serialize_probability_reference(probabilities: pd.Series, max_points: int = 4096) -> dict[str, Any]:
    cleaned = probabilities.dropna().astype("float64")
    if cleaned.empty:
        return {"sample_count": 0, "sample": []}
    if len(cleaned) <= max_points:
        sample = cleaned
    else:
        sample_indices = np.linspace(0, len(cleaned) - 1, num=max_points, dtype=int)
        sample = cleaned.iloc[sample_indices]
    return {
        "sample_count": int(len(cleaned)),
        "sample": [float(value) for value in sample.to_list()],
    }


def _get_threshold_search_settings(settings: Settings) -> dict[str, float]:
    config = settings.dataset.threshold_search
    return {
        "stage1_coverage_min": float(config.stage1_coverage_min),
        "stage1_coverage_max": float(config.stage1_coverage_max),
        "min_active_samples": int(config.min_active_samples),
        "min_end_to_end_coverage": float(config.min_end_to_end_coverage),
    }


def _round4(value: float) -> float:
    return round(float(value), 4)


def _tune_stage1_filter_threshold(
    *,
    y_true: pd.Series,
    probabilities: pd.Series,
    coverage_min: float,
    coverage_max: float,
) -> tuple[float, dict[str, Any]]:
    candidates = [round(float(threshold), 4) for threshold in np.arange(0.10, 0.901, 0.02)]
    records: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    fallback_reason: str | None = None

    for threshold in candidates:
        metrics = compute_binary_classification_metrics(y_true, probabilities, threshold=threshold)
        record = {
            "threshold": threshold,
            "coverage": compute_stage1_coverage(probabilities, threshold),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "sample_count": float(len(y_true)),
        }
        records.append(record)
        if coverage_min <= record["coverage"] <= coverage_max:
            eligible.append(record)

    if eligible:
        best = max(eligible, key=lambda item: (_round4(item["precision"]), item["threshold"]))
        constraint_satisfied = True
    else:
        constraint_satisfied = False
        fallback_reason = "no threshold satisfied stage1 coverage bounds"
        best = max(
            records,
            key=lambda item: (
                -min(abs(item["coverage"] - coverage_min), abs(item["coverage"] - coverage_max)),
                _round4(item["precision"]),
                item["threshold"],
            ),
        )

    return best["threshold"], {
        "selection_data": "validation",
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "constraint_satisfied": constraint_satisfied,
        "fallback_reason": fallback_reason,
        "records": records,
        "best": best,
    }


def _empty_stage2_prediction_series(index: pd.Index) -> pd.Series:
    return pd.Series(float("nan"), index=index, dtype="float64", name=DEFAULT_STAGE2_PREDICTED_RETURN_COLUMN)


def _fill_stage2_predictions(full_index: pd.Index, subset: TrainingFrame, predictions: pd.Series) -> pd.Series:
    full = _empty_stage2_prediction_series(full_index)
    if not subset.frame.empty:
        full.loc[subset.frame.index] = predictions.to_numpy()
    return full


def _summarize_stage2_sign_decisions(
    *,
    full_validation_target: pd.Series,
    stage1_validation_probabilities: pd.Series,
    full_validation_stage2_predictions: pd.Series,
    stage1_threshold: float,
    min_active_samples: int,
    min_end_to_end_coverage: float,
) -> dict[str, Any]:
    end_to_end = compute_two_stage_end_to_end_metrics(
        full_validation_target,
        stage1_validation_probabilities,
        full_validation_stage2_predictions,
        stage1_threshold=stage1_threshold,
    )
    record = {
        "decision_rule": "sign(predicted_median_return)",
        "trade_precision_up": end_to_end["trade_precision_up"],
        "trade_precision_down": end_to_end["trade_precision_down"],
        "coverage_end_to_end": end_to_end["coverage_end_to_end"],
        "stage2_trade_count": end_to_end["stage2_trade_count"],
        "pnl_per_sample": end_to_end["trade_pnl.pnl_per_sample"],
    }
    constraint_satisfied = (
        record["stage2_trade_count"] >= min_active_samples
        and record["coverage_end_to_end"] >= min_end_to_end_coverage
    )
    return {
        "selection_data": "stage2_validation_sign_rule",
        "min_active_samples": min_active_samples,
        "min_end_to_end_coverage": min_end_to_end_coverage,
        "constraint_satisfied": constraint_satisfied,
        "fallback_reason": None if constraint_satisfied else "sign rule did not satisfy stage2 hard constraints",
        "records": [record],
        "best": record,
    }


def _compute_stage1_window_stats(
    frame: pd.DataFrame,
    stage1_probabilities: pd.Series,
    stage1_threshold: float,
    tau: float,
) -> dict[str, Any]:
    stage1_target = (frame[DEFAULT_ABS_RETURN_COLUMN] > tau).astype(int)
    metrics = compute_binary_classification_metrics(stage1_target, stage1_probabilities, threshold=stage1_threshold)
    selected_mask = stage1_probabilities >= stage1_threshold
    return {
        "stage1_class_ratio": {
            "inactive": float((stage1_target == 0).mean()) if len(stage1_target) else 0.0,
            "active": float((stage1_target == 1).mean()) if len(stage1_target) else 0.0,
        },
        "stage2_selected_ratio": float(selected_mask.mean()) if len(selected_mask) else 0.0,
        "stage1_filter_purity": float(stage1_target.loc[selected_mask].mean()) if selected_mask.any() else 0.0,
        "stage1_precision": metrics["precision"],
        "stage1_recall": metrics["recall"],
    }


def _compute_stage2_window_stats(selected_frame: pd.DataFrame) -> dict[str, Any]:
    if selected_frame.empty:
        return {
            "stage2_row_count": 0,
            "stage2_return_direction_ratio": {"down": 0.0, "flat": 0.0, "up": 0.0},
        }
    target = selected_frame[DEFAULT_STAGE2_TARGET_COLUMN].astype(float)
    return {
        "stage2_row_count": int(len(selected_frame)),
        "stage2_return_direction_ratio": {
            "down": float((target < 0.0).mean()),
            "flat": float((target == 0.0).mean()),
            "up": float((target > 0.0).mean()),
        },
    }


def _train_two_stage_for_split(
    development: TrainingFrame,
    validation: TrainingFrame,
    settings: Settings,
) -> tuple[
    ModelPlugin,
    CalibrationPlugin,
    ModelPlugin,
    CalibrationPlugin,
    float,
    float,
    float,
    float,
    dict[str, Any],
    dict[str, dict[str, float]],
    dict[str, Any],
    dict[str, dict[str, float]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    threshold_settings = _get_threshold_search_settings(settings)
    tau = float(settings.labels.two_stage.active_return_threshold)

    stage1_training = _build_stage1_training_frame(development, settings)
    stage1_validation = _build_stage1_training_frame(validation, settings)
    stage1_model = _fit_model(stage1_training, settings, stage="stage1", validation=stage1_validation)
    stage1_dev_raw = stage1_model.predict_proba(stage1_training.X)
    stage1_valid_raw = stage1_model.predict_proba(stage1_validation.X)
    stage1_calibrator = _select_stage1_calibrator(settings, stage1_valid_raw, stage1_validation.y.astype(int))
    stage1_dev_proba = stage1_calibrator.transform(stage1_dev_raw)
    stage1_valid_proba = stage1_calibrator.transform(stage1_valid_raw)

    stage1_threshold, stage1_threshold_search = _tune_stage1_filter_threshold(
        y_true=stage1_validation.y.astype(int),
        probabilities=stage1_valid_proba,
        coverage_min=threshold_settings["stage1_coverage_min"],
        coverage_max=threshold_settings["stage1_coverage_max"],
    )

    stage2_training = _build_stage2_training_frame(
        development,
        stage1_dev_proba,
        stage1_threshold=stage1_threshold,
    )
    stage2_validation = _build_stage2_training_frame(
        validation,
        stage1_valid_proba,
        stage1_threshold=stage1_threshold,
    )
    if stage2_training.frame.empty:
        raise ValueError("No samples passed the Stage 1 filter for Stage 2 training.")

    stage2_model = _fit_model(
        stage2_training,
        settings,
        stage="stage2",
        validation=stage2_validation if not stage2_validation.frame.empty else None,
    )
    stage2_calibrator = NoCalibration()
    stage2_dev_predictions = stage2_model.predict(stage2_training.X)
    stage2_valid_predictions = (
        stage2_model.predict(stage2_validation.X)
        if not stage2_validation.frame.empty
        else _empty_stage2_prediction_series(stage2_validation.frame.index)
    )

    full_stage2_dev_predictions = _fill_stage2_predictions(development.frame.index, stage2_training, stage2_dev_predictions)
    full_stage2_valid_predictions = _fill_stage2_predictions(validation.frame.index, stage2_validation, stage2_valid_predictions)

    up_threshold = 0.0
    down_threshold = 0.0
    margin_threshold = 0.0
    stage2_threshold_search = _summarize_stage2_sign_decisions(
        full_validation_target=validation.frame[DEFAULT_STAGE2_TARGET_COLUMN].astype(float),
        stage1_validation_probabilities=stage1_valid_proba,
        full_validation_stage2_predictions=full_stage2_valid_predictions,
        stage1_threshold=stage1_threshold,
        min_active_samples=threshold_settings["min_active_samples"],
        min_end_to_end_coverage=threshold_settings["min_end_to_end_coverage"],
    )

    train_metrics = {
        "stage1": compute_binary_classification_metrics(stage1_training.y.astype(int), stage1_dev_proba, threshold=stage1_threshold),
        "stage2": compute_return_direction_metrics(stage2_training.y.astype(float), stage2_dev_predictions),
        "end_to_end": compute_two_stage_end_to_end_metrics(
            development.frame[DEFAULT_STAGE2_TARGET_COLUMN].astype(float),
            stage1_dev_proba,
            full_stage2_dev_predictions,
            stage1_threshold=stage1_threshold,
        ),
    }
    validation_metrics = {
        "stage1": compute_binary_classification_metrics(stage1_validation.y.astype(int), stage1_valid_proba, threshold=stage1_threshold),
        "stage2": compute_return_direction_metrics(stage2_validation.y.astype(float), stage2_valid_predictions)
        if not stage2_validation.frame.empty
        else {"sample_count": 0.0},
        "end_to_end": compute_two_stage_end_to_end_metrics(
            validation.frame[DEFAULT_STAGE2_TARGET_COLUMN].astype(float),
            stage1_valid_proba,
            full_stage2_valid_predictions,
            stage1_threshold=stage1_threshold,
        ),
    }
    probability_summary = {
        "stage1_prob_train": _summarize_probability_series(stage1_dev_proba),
        "stage1_prob_validation": _summarize_probability_series(stage1_valid_proba),
        "stage1_prob_ks": {
            "train_vs_validation": float(compute_ks_distance(stage1_dev_proba, stage1_valid_proba)),
        },
    }
    probability_reference = {
        "stage1_prob_train": _serialize_probability_reference(stage1_dev_proba),
        "stage1_prob_validation": _serialize_probability_reference(stage1_valid_proba),
    }
    stage2_direction_reference = {
        "stage2_direction_train": _serialize_probability_reference(stage2_dev_predictions),
        "stage2_direction_validation": (
            _serialize_probability_reference(stage2_valid_predictions)
            if not stage2_valid_predictions.empty
            else {"sample_count": 0, "sample": []}
        ),
    }
    train_window = {
        "row_count": len(development.frame),
        "start": str(development.frame["timestamp"].min()) if not development.frame.empty else None,
        "end": str(development.frame["timestamp"].max()) if not development.frame.empty else None,
        **_compute_stage1_window_stats(development.frame, stage1_dev_proba, stage1_threshold, tau),
        **_compute_stage2_window_stats(stage2_training.frame),
    }
    validation_window = {
        "row_count": len(validation.frame),
        "start": str(validation.frame["timestamp"].min()) if not validation.frame.empty else None,
        "end": str(validation.frame["timestamp"].max()) if not validation.frame.empty else None,
        **_compute_stage1_window_stats(validation.frame, stage1_valid_proba, stage1_threshold, tau),
        **_compute_stage2_window_stats(stage2_validation.frame),
    }

    return (
        stage1_model,
        stage1_calibrator,
        stage2_model,
        stage2_calibrator,
        stage1_threshold,
        up_threshold,
        down_threshold,
        margin_threshold,
        {
            "stage1_threshold_search": stage1_threshold_search,
            "stage2_threshold_search": stage2_threshold_search,
        },
        probability_summary,
        probability_reference,
        {"train": train_metrics, "validation": validation_metrics},
        train_window,
        validation_window,
        stage2_direction_reference,
    )


def train_two_stage_model(
    training: TrainingFrame,
    settings: Settings,
    validation_window_days: int = 30,
    purge_rows: int = 1,
) -> TwoStageTrainingArtifacts:
    development, validation = split_training_frame(
        training,
        validation_window_days=validation_window_days,
        purge_rows=purge_rows,
    )
    return train_two_stage_model_from_split(
        development=development,
        validation=validation,
        settings=settings,
    )


def train_two_stage_model_from_split(
    *,
    development: TrainingFrame,
    validation: TrainingFrame,
    settings: Settings,
) -> TwoStageTrainingArtifacts:
    (
        stage1_model,
        stage1_calibrator,
        stage2_model,
        stage2_calibrator,
        stage1_threshold,
        up_threshold,
        down_threshold,
        margin_threshold,
        threshold_search,
        probability_summary,
        probability_reference,
        metrics,
        train_window,
        validation_window,
        stage2_direction_reference,
    ) = _train_two_stage_for_split(development, validation, settings)

    walk_forward_enabled = bool(settings.dataset.walk_forward.get("enabled", False))
    walk_forward_results: list[WalkForwardFoldResult] = []
    walk_forward_fold_details: list[dict[str, Any]] = []
    base_rate = float((development.frame[DEFAULT_STAGE2_TARGET_COLUMN] > 0.0).mean()) if not development.frame.empty else 0.0
    stage2_feature_columns = [*development.feature_columns, DEFAULT_STAGE1_PROBABILITY_COLUMN]

    return TwoStageTrainingArtifacts(
        stage1_model=stage1_model,
        stage1_calibrator=stage1_calibrator,
        stage2_model=stage2_model,
        stage2_calibrator=stage2_calibrator,
        feature_columns=development.feature_columns,
        stage2_feature_columns=stage2_feature_columns,
        stage1_threshold=stage1_threshold,
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        margin_threshold=margin_threshold,
        train_metrics=metrics["train"],
        train_window=train_window,
        validation_window=validation_window,
        validation_metrics=metrics["validation"],
        walk_forward_summary=(
            summarize_walk_forward(walk_forward_results)
            if walk_forward_enabled
            else {"enabled": False, "fold_count": 0}
        ),
        walk_forward_results=walk_forward_results,
        walk_forward_fold_details=walk_forward_fold_details,
        base_rate=base_rate,
        threshold_search=threshold_search,
        stage1_probability_summary=probability_summary,
        stage1_probability_reference=probability_reference,
        stage2_direction_reference=stage2_direction_reference,
    )
