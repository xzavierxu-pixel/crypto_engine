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
    DEFAULT_SAMPLE_WEIGHT_COLUMN,
    DEFAULT_STAGE1_PROBABILITY_COLUMN,
    DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN,
    DEFAULT_TARGET_COLUMN,
)
from src.data.dataset_builder import TrainingFrame
from src.model.base import ModelPlugin
from src.model.evaluation import (
    WalkForwardFoldResult,
    compute_classification_metrics,
    compute_pnl_metrics,
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
    buy_threshold: float
    train_metrics: dict[str, dict[str, float]]
    train_window: dict[str, str | int | None]
    validation_window: dict[str, str | int | None]
    validation_metrics: dict[str, dict[str, float]]
    walk_forward_summary: dict[str, float | int]
    walk_forward_results: list[WalkForwardFoldResult]
    walk_forward_fold_details: list[dict[str, Any]]
    base_rate: float
    threshold_search: dict[str, Any]
    stage1_probability_summary: dict[str, dict[str, float]]
    stage1_probability_reference: dict[str, Any]


def _slice_training_frame(training: TrainingFrame, frame_slice: slice) -> TrainingFrame:
    return TrainingFrame(
        frame=training.frame.iloc[frame_slice].reset_index(drop=True),
        feature_columns=training.feature_columns,
        target_column=training.target_column,
        sample_weight_column=training.sample_weight_column,
    )


def _make_training_frame_from_cached_split(frame: pd.DataFrame) -> TrainingFrame:
    feature_columns = [
        column for column in frame.columns
        if column not in {
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
            DEFAULT_ABS_RETURN_COLUMN,
            DEFAULT_SAMPLE_WEIGHT_COLUMN,
            DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN,
        }
    ]
    return TrainingFrame(
        frame=frame.reset_index(drop=True),
        feature_columns=feature_columns,
        target_column=DEFAULT_TARGET_COLUMN,
        sample_weight_column=DEFAULT_SAMPLE_WEIGHT_COLUMN if DEFAULT_SAMPLE_WEIGHT_COLUMN in frame.columns else None,
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
    return TrainingFrame(
        frame=(frame if frame is not None else training.frame).reset_index(drop=True)
        if reset_index
        else (frame if frame is not None else training.frame).copy(),
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
    settings: Settings,
) -> TrainingFrame:
    tau = float(settings.labels.two_stage.active_return_threshold)
    frame = training.frame.copy()
    frame[DEFAULT_STAGE1_PROBABILITY_COLUMN] = stage1_probabilities.reindex(frame.index)
    active_frame = frame.loc[
        (frame[DEFAULT_ABS_RETURN_COLUMN] > tau)
        & frame[DEFAULT_STAGE1_PROBABILITY_COLUMN].notna()
    ].copy()
    return _replace_training_columns(
        training,
        frame=active_frame,
        target_column=DEFAULT_TARGET_COLUMN,
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
    return {"scale_pos_weight": _compute_stage1_scale_pos_weight(training)}


def _select_calibrator(
    settings: Settings,
    raw_probabilities: pd.Series,
    y_true: pd.Series,
    *,
    stage: str,
) -> CalibrationPlugin:
    plugin_name = settings.calibration.resolve_plugin(stage=stage)
    if plugin_name == "none":
        return NoCalibration()
    if raw_probabilities.empty or len(raw_probabilities) < 100 or y_true.nunique() < 2:
        return NoCalibration()

    base_metrics = compute_classification_metrics(y_true, raw_probabilities)
    calibrator = create_calibration_plugin(settings, stage=stage)
    calibrator.fit(raw_probabilities, y_true)
    calibrated_probabilities = calibrator.transform(raw_probabilities)
    calibrated_metrics = compute_classification_metrics(y_true, calibrated_probabilities)

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
        training.y.astype(int),
        X_valid=validation.X if validation is not None and not validation.frame.empty else None,
        y_valid=validation.y.astype(int) if validation is not None and not validation.frame.empty else None,
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


def _summarize_threshold_records(records: list[dict[str, float]]) -> dict[str, dict[str, float | None]]:
    metric_columns = [
        "coverage",
        "trade_accuracy",
        "pnl_per_trade",
        "pnl_per_sample",
        "active_sample_count",
        "stage1_precision",
        "stage1_recall",
    ]
    if not records:
        return {
            metric: {"max": None, "min": None, "mean": None, "median": None}
            for metric in metric_columns
        }

    frame = pd.DataFrame(records)
    summary: dict[str, dict[str, float | None]] = {}
    for metric in metric_columns:
        series = frame[metric].astype("float64")
        summary[metric] = {
            "max": float(series.max()),
            "min": float(series.min()),
            "mean": float(series.mean()),
            "median": float(series.median()),
        }
    return summary


def _get_threshold_search_settings(settings: Settings) -> dict[str, float]:
    config = settings.dataset.threshold_search
    return {
        "min_stage1_coverage": float(config.get("min_stage1_coverage", 0.60)),
        "min_active_samples": int(config.get("min_active_samples", 25)),
    }


def _tune_stage1_threshold(
    *,
    stage1_y_true: pd.Series,
    stage1_probabilities: pd.Series,
    end_to_end_y_true: pd.Series,
    stage2_probabilities: pd.Series,
    min_stage1_coverage: float,
    min_active_samples: int,
) -> tuple[float, float, dict[str, Any]]:
    stage1_candidates = [round(float(threshold), 4) for threshold in np.arange(0.10, 0.901, 0.02)]
    buy_candidates = [round(float(threshold), 4) for threshold in np.arange(0.10, 0.901, 0.02)]
    all_records: list[dict[str, float]] = []
    eligible_records: list[dict[str, float]] = []
    best_stage1_threshold = 0.5
    best_buy_threshold = 0.5
    best_rank_key = (float("-inf"), float("-inf"), float("-inf"))
    best_coverage_fallback_key = (float("-inf"), float("-inf"), float("-inf"))
    used_coverage_fallback = False

    stage1_y = stage1_y_true.astype(int)
    end_to_end_y = end_to_end_y_true.astype(int)

    for stage1_threshold in stage1_candidates:
        stage1_metrics = compute_classification_metrics(
            stage1_y,
            stage1_probabilities,
            threshold=stage1_threshold,
        )
        for buy_threshold in buy_candidates:
            metrics = compute_pnl_metrics(
                y_true=end_to_end_y,
                stage1_probabilities=stage1_probabilities,
                stage2_probabilities=stage2_probabilities,
                stage1_threshold=stage1_threshold,
                buy_threshold=buy_threshold,
            )
            record = {
                "stage1_threshold": stage1_threshold,
                "buy_threshold": buy_threshold,
                "coverage": metrics["coverage"],
                "trade_accuracy": metrics["trade_accuracy"],
                "pnl_per_trade": metrics["pnl_per_trade"],
                "pnl_per_sample": metrics["pnl_per_sample"],
                "active_sample_count": metrics["active_sample_count"],
                "stage1_precision": stage1_metrics["precision"],
                "stage1_recall": stage1_metrics["recall"],
            }
            all_records.append(record)
            if record["active_sample_count"] < min_active_samples:
                continue
            coverage_fallback_key = (
                record["coverage"],
                record["pnl_per_sample"],
                record["trade_accuracy"],
            )
            if coverage_fallback_key > best_coverage_fallback_key:
                best_coverage_fallback_key = coverage_fallback_key
                best_stage1_threshold = stage1_threshold
                best_buy_threshold = buy_threshold

            if record["coverage"] < min_stage1_coverage:
                continue
            eligible_records.append(record)
            rank_key = (
                record["pnl_per_sample"],
                record["trade_accuracy"],
                record["coverage"],
            )
            if rank_key > best_rank_key:
                best_rank_key = rank_key
                best_stage1_threshold = stage1_threshold
                best_buy_threshold = buy_threshold

    constraint_satisfied = best_rank_key[0] != float("-inf")
    if not constraint_satisfied:
        used_coverage_fallback = True

    best_record = next(
        (
            record
            for record in all_records
            if record["stage1_threshold"] == best_stage1_threshold and record["buy_threshold"] == best_buy_threshold
        ),
        None,
    )

    return (
        best_stage1_threshold,
        best_buy_threshold,
        {
            "selection_data": "validation",
            "stage1_threshold_candidates": stage1_candidates,
            "buy_threshold_candidates": buy_candidates,
            "ranking_priority": ["pnl_per_sample", "trade_accuracy", "coverage"],
            "stage1_coverage_constraint": min_stage1_coverage,
            "constraint_applied": True,
            "constraint_satisfied": constraint_satisfied,
            "fallback_reason": (
                f"no threshold combination satisfied validation stage1 coverage >= {min_stage1_coverage:.4f}"
                if used_coverage_fallback
                else None
            ),
            "min_active_sample_count": min_active_samples,
            "total_record_count": len(all_records),
            "eligible_record_count": len(eligible_records),
            "records": eligible_records,
            "record_metric_summary": _summarize_threshold_records(eligible_records),
            "best": {
                "stage1_threshold": best_stage1_threshold,
                "buy_threshold": best_buy_threshold,
                "ranking_key": {
                    "pnl_per_sample": best_rank_key[0] if constraint_satisfied else best_coverage_fallback_key[1],
                    "trade_accuracy": best_rank_key[1] if constraint_satisfied else best_coverage_fallback_key[2],
                    "coverage": best_rank_key[2] if constraint_satisfied else best_coverage_fallback_key[0],
                },
                "stage1_precision": None if best_record is None else best_record["stage1_precision"],
                "stage1_recall": None if best_record is None else best_record["stage1_recall"],
                "coverage": None if best_record is None else best_record["coverage"],
                "active_sample_count": None if best_record is None else best_record["active_sample_count"],
            },
        },
    )


def _train_two_stage_for_split(
    development: TrainingFrame,
    validation: TrainingFrame,
    settings: Settings,
) -> tuple[
    ModelPlugin,
    CalibrationPlugin,
    ModelPlugin,
    CalibrationPlugin,
    pd.Series,
    pd.Series,
    float,
    float,
    float,
    dict[str, Any],
    dict[str, dict[str, float]],
    dict[str, Any],
    dict[str, dict[str, float]],
]:
    threshold_settings = _get_threshold_search_settings(settings)
    stage1_training = _build_stage1_training_frame(development, settings)
    stage1_validation = _build_stage1_training_frame(validation, settings)
    stage1_model = _fit_model(stage1_training, settings, stage="stage1", validation=stage1_validation)
    stage1_dev_raw = stage1_model.predict_proba(stage1_training.X)
    stage1_valid_raw = stage1_model.predict_proba(stage1_validation.X)
    stage1_calibrator = _select_calibrator(
        settings,
        stage1_valid_raw,
        stage1_validation.y.astype(int),
        stage="stage1",
    )
    stage1_dev_proba = stage1_calibrator.transform(stage1_dev_raw)
    stage1_valid_proba = stage1_calibrator.transform(stage1_valid_raw)

    stage2_training = _build_stage2_training_frame(development, stage1_dev_proba, settings)
    if stage2_training.frame.empty:
        raise ValueError("No active samples available for Stage 2 training.")
    base_rate = float(stage2_training.y.mean())
    stage2_validation = _build_stage2_training_frame(validation, stage1_valid_proba, settings)
    stage2_model = _fit_model(
        stage2_training,
        settings,
        stage="stage2",
        validation=stage2_validation if not stage2_validation.frame.empty else None,
    )
    stage2_dev_raw = stage2_model.predict_proba(stage2_training.X)
    stage2_valid_raw = (
        stage2_model.predict_proba(stage2_validation.X)
        if not stage2_validation.frame.empty
        else pd.Series(dtype="float64")
    )
    stage2_calibrator = _select_calibrator(
        settings,
        stage2_valid_raw,
        stage2_validation.y.astype(int) if not stage2_validation.frame.empty else pd.Series(dtype="int64"),
        stage="stage2",
    )
    stage2_dev_proba = stage2_calibrator.transform(stage2_dev_raw)
    stage2_valid_proba = (
        stage2_calibrator.transform(stage2_valid_raw)
        if not stage2_valid_raw.empty
        else pd.Series(dtype="float64")
    )
    stage2_dev_input = _build_stage2_training_frame(development, stage1_dev_proba, settings)
    stage2_valid_input = stage2_validation
    full_stage2_dev_proba = pd.Series(base_rate, index=development.frame.index, dtype="float64")
    full_stage2_valid_proba = pd.Series(base_rate, index=validation.frame.index, dtype="float64")
    full_stage2_dev_proba.loc[stage2_dev_input.frame.index] = stage2_dev_proba.to_numpy()
    if not stage2_valid_input.frame.empty:
        full_stage2_valid_proba.loc[stage2_valid_input.frame.index] = stage2_valid_proba.to_numpy()
    stage1_threshold, buy_threshold, threshold_search = _tune_stage1_threshold(
        stage1_y_true=stage1_validation.y.astype(int),
        stage1_probabilities=stage1_valid_proba,
        end_to_end_y_true=validation.y.astype(int),
        stage2_probabilities=full_stage2_valid_proba,
        min_stage1_coverage=threshold_settings["min_stage1_coverage"],
        min_active_samples=threshold_settings["min_active_samples"],
    )

    train_metrics = {
        "stage1": compute_classification_metrics(stage1_training.y.astype(int), stage1_dev_proba, threshold=stage1_threshold),
        "stage2": compute_classification_metrics(stage2_dev_input.y.astype(int), stage2_dev_proba, threshold=buy_threshold),
        "end_to_end": compute_pnl_metrics(
            y_true=development.y.astype(int),
            stage1_probabilities=stage1_dev_proba,
            stage2_probabilities=full_stage2_dev_proba,
            stage1_threshold=stage1_threshold,
            buy_threshold=buy_threshold,
        ),
    }
    validation_metrics = {
        "stage1": compute_classification_metrics(
            _build_stage1_training_frame(validation, settings).y.astype(int),
            stage1_valid_proba,
            threshold=stage1_threshold,
        ),
        "stage2": (
            compute_classification_metrics(stage2_valid_input.y.astype(int), stage2_valid_proba, threshold=buy_threshold)
            if not stage2_valid_input.frame.empty
            else {"sample_count": 0.0}
        ),
        "end_to_end": (
            compute_pnl_metrics(
                y_true=validation.y.astype(int),
                stage1_probabilities=stage1_valid_proba,
                stage2_probabilities=full_stage2_valid_proba,
                stage1_threshold=stage1_threshold,
                buy_threshold=buy_threshold,
            )
            if not validation.frame.empty
            else {"sample_count": 0.0, "pnl_per_sample": 0.0, "coverage": 0.0}
        ),
    }
    probability_summary = {
        "stage1_prob_train": _summarize_probability_series(stage1_dev_proba),
        "stage1_prob_validation": _summarize_probability_series(stage1_valid_proba),
    }
    probability_reference = {
        "stage1_prob_train": _serialize_probability_reference(stage1_dev_proba),
    }

    return (
        stage1_model,
        stage1_calibrator,
        stage2_model,
        stage2_calibrator,
        stage1_valid_proba,
        stage2_valid_proba,
        stage1_threshold,
        buy_threshold,
        base_rate,
        threshold_search,
        probability_summary,
        probability_reference,
        {"train": train_metrics, "validation": validation_metrics},
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
        _stage1_valid_proba,
        _stage2_valid_proba,
        stage1_threshold,
        buy_threshold,
        base_rate,
        threshold_search,
        probability_summary,
        probability_reference,
        metrics,
    ) = _train_two_stage_for_split(development, validation, settings)

    walk_forward_enabled = bool(settings.dataset.walk_forward.get("enabled", False))
    walk_forward_results: list[WalkForwardFoldResult] = []
    walk_forward_fold_details: list[dict[str, Any]] = []
    # walk_forward is retained as a compatibility/reporting switch only.

    development_frame = development.frame
    validation_frame = validation.frame
    stage2_feature_columns = [*development.feature_columns, DEFAULT_STAGE1_PROBABILITY_COLUMN]

    return TwoStageTrainingArtifacts(
        stage1_model=stage1_model,
        stage1_calibrator=stage1_calibrator,
        stage2_model=stage2_model,
        stage2_calibrator=stage2_calibrator,
        feature_columns=development.feature_columns,
        stage2_feature_columns=stage2_feature_columns,
        stage1_threshold=stage1_threshold,
        buy_threshold=buy_threshold,
        train_metrics=metrics["train"],
        train_window={
            "row_count": len(development_frame),
            "start": str(development_frame["timestamp"].min()) if not development_frame.empty else None,
            "end": str(development_frame["timestamp"].max()) if not development_frame.empty else None,
        },
        validation_window={
            "row_count": len(validation_frame),
            "start": str(validation_frame["timestamp"].min()) if not validation_frame.empty else None,
            "end": str(validation_frame["timestamp"].max()) if not validation_frame.empty else None,
        },
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
    )
