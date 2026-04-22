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
    build_walk_forward_splits,
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
    stage1_threshold_scan: list[dict[str, float]]
    stage1_probability_summary: dict[str, dict[str, float]]
    stage1_probability_reference: dict[str, Any]


def _slice_training_frame(training: TrainingFrame, frame_slice: slice) -> TrainingFrame:
    return TrainingFrame(
        frame=training.frame.iloc[frame_slice].reset_index(drop=True),
        feature_columns=training.feature_columns,
        target_column=training.target_column,
        sample_weight_column=training.sample_weight_column,
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
    return _replace_training_columns(
        training,
        frame=frame,
        target_column="stage1_target",
        sample_weight_column=DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN
        if DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN in frame.columns
        else DEFAULT_SAMPLE_WEIGHT_COLUMN,
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
        sample_weight_column=DEFAULT_SAMPLE_WEIGHT_COLUMN if DEFAULT_SAMPLE_WEIGHT_COLUMN in active_frame.columns else None,
        feature_columns=[*training.feature_columns, DEFAULT_STAGE1_PROBABILITY_COLUMN],
        reset_index=False,
    )


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
) -> ModelPlugin:
    model = create_model_plugin(settings, stage=stage)
    model.fit(training.X, training.y.astype(int), sample_weight=training.sample_weight)
    return model


def _build_oof_predictions(
    training: TrainingFrame,
    settings: Settings,
    *,
    stage: str,
    step_size: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    minimum_rows = 60
    if len(training.frame) < minimum_rows:
        return pd.Series(dtype="float64"), pd.Series(dtype="int64")

    validation_size = max(len(training.frame) // 5, 1)
    min_train_size = max(validation_size * 2, minimum_rows)
    walk_splits = build_walk_forward_splits(
        training,
        min_train_size=min_train_size,
        validation_size=validation_size,
        step_size=step_size or max(validation_size // 2, 1),
        purge_rows=1,
    )
    if not walk_splits:
        return pd.Series(dtype="float64"), pd.Series(dtype="int64")

    oof_probabilities: list[pd.Series] = []
    oof_targets: list[pd.Series] = []
    for split in walk_splits:
        fold_training = _slice_training_frame(training, split.train_slice)
        X_valid = training.X.iloc[split.valid_slice]
        y_valid = training.y.astype(int).iloc[split.valid_slice]
        fold_model = create_model_plugin(settings, stage=stage)
        fold_model.fit(
            fold_training.X,
            fold_training.y.astype(int),
            sample_weight=fold_training.sample_weight,
        )
        oof_probabilities.append(fold_model.predict_proba(X_valid))
        oof_targets.append(y_valid)

    probability_series = pd.concat(oof_probabilities).sort_index()
    target_series = pd.concat(oof_targets).sort_index()
    if probability_series.index.has_duplicates:
        probability_series = probability_series.groupby(level=0).mean()
    if target_series.index.has_duplicates:
        target_series = target_series.groupby(level=0).last()
    return probability_series, target_series


def _require_strict_oof_predictions(
    training: TrainingFrame,
    oof_probabilities: pd.Series,
    *,
    stage: str,
) -> tuple[pd.Series, pd.Series]:
    if oof_probabilities.empty:
        raise ValueError(f"{stage} strict OOF predictions are unavailable for the current split.")

    aligned = oof_probabilities.reindex(training.frame.index).astype("float64")
    covered = aligned.dropna()
    if covered.empty:
        raise ValueError(f"{stage} strict OOF predictions are unavailable for the current split.")
    return aligned, covered


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


def _tune_stage1_threshold(
    stage1_probabilities: pd.Series,
    stage2_probabilities: pd.Series,
    y_true: pd.Series,
    buy_threshold: float,
) -> tuple[float, list[dict[str, float]]]:
    records: list[dict[str, float]] = []
    best_threshold = 0.5
    best_score = float("-inf")

    for threshold in np.arange(0.30, 0.80, 0.02):
        metrics = compute_pnl_metrics(
            y_true=y_true,
            stage1_probabilities=stage1_probabilities,
            stage2_probabilities=stage2_probabilities,
            stage1_threshold=float(threshold),
            buy_threshold=buy_threshold,
        )
        if metrics["active_sample_count"] < 25:
            continue
        record = {
            "threshold": float(threshold),
            "coverage": metrics["coverage"],
            "trade_accuracy": metrics["trade_accuracy"],
            "pnl_per_trade": metrics["pnl_per_trade"],
            "pnl_per_sample": metrics["pnl_per_sample"],
        }
        records.append(record)
        if record["pnl_per_sample"] > best_score:
            best_score = record["pnl_per_sample"]
            best_threshold = float(threshold)

    return best_threshold, records


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
    list[dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, Any],
    dict[str, dict[str, float]],
]:
    stage1_training = _build_stage1_training_frame(development, settings)
    stage1_oof_raw, _ = _build_oof_predictions(stage1_training, settings, stage="stage1")
    stage1_oof_aligned, stage1_oof = _require_strict_oof_predictions(
        stage1_training,
        stage1_oof_raw,
        stage="Stage 1",
    )
    stage1_oof_targets = stage1_training.y.astype(int).loc[stage1_oof.index]
    stage1_model = _fit_model(stage1_training, settings, stage="stage1")
    stage1_calibrator = _select_calibrator(settings, stage1_oof, stage1_oof_targets, stage="stage1")

    stage2_training = _build_stage2_training_frame(development, stage1_oof_aligned, settings)
    if stage2_training.frame.empty:
        raise ValueError("No active samples available for Stage 2 training.")
    base_rate = float(stage2_training.y.mean())
    stage2_oof_raw, _ = _build_oof_predictions(stage2_training, settings, stage="stage2")
    stage2_model = _fit_model(stage2_training, settings, stage="stage2")
    _, stage2_oof_raw = _require_strict_oof_predictions(
        stage2_training,
        stage2_oof_raw,
        stage="Stage 2",
    )
    stage2_oof_targets = stage2_training.y.astype(int).loc[stage2_oof_raw.index]
    stage2_calibrator = _select_calibrator(settings, stage2_oof_raw, stage2_oof_targets, stage="stage2")
    stage2_oof = stage2_calibrator.transform(stage2_oof_raw)
    stage1_active_prob = stage2_training.frame.loc[stage2_oof.index, DEFAULT_STAGE1_PROBABILITY_COLUMN].astype("float64")
    stage1_threshold, threshold_scan = _tune_stage1_threshold(
        stage1_probabilities=stage1_active_prob,
        stage2_probabilities=stage2_oof,
        y_true=stage2_oof_targets,
        buy_threshold=base_rate,
    )

    stage1_dev_proba = stage1_calibrator.transform(stage1_model.predict_proba(stage1_training.X))
    stage1_valid_proba = stage1_calibrator.transform(stage1_model.predict_proba(validation.X))
    stage2_dev_input = _build_stage2_training_frame(development, stage1_dev_proba, settings)
    stage2_valid_input = _build_stage2_training_frame(validation, stage1_valid_proba, settings)
    stage2_dev_proba = stage2_calibrator.transform(stage2_model.predict_proba(stage2_dev_input.X))
    stage2_valid_proba = (
        stage2_calibrator.transform(stage2_model.predict_proba(stage2_valid_input.X))
        if not stage2_valid_input.frame.empty
        else pd.Series(dtype="float64")
    )
    full_stage2_dev_proba = pd.Series(base_rate, index=development.frame.index, dtype="float64")
    full_stage2_valid_proba = pd.Series(base_rate, index=validation.frame.index, dtype="float64")
    full_stage2_dev_proba.loc[stage2_dev_input.frame.index] = stage2_dev_proba.to_numpy()
    if not stage2_valid_input.frame.empty:
        full_stage2_valid_proba.loc[stage2_valid_input.frame.index] = stage2_valid_proba.to_numpy()

    train_metrics = {
        "stage1": compute_classification_metrics(stage1_training.y.astype(int), stage1_dev_proba, threshold=stage1_threshold),
        "stage2": compute_classification_metrics(stage2_dev_input.y.astype(int), stage2_dev_proba, threshold=base_rate),
        "end_to_end": compute_pnl_metrics(
            y_true=development.y.astype(int),
            stage1_probabilities=stage1_dev_proba,
            stage2_probabilities=full_stage2_dev_proba,
            stage1_threshold=stage1_threshold,
            buy_threshold=base_rate,
        ),
    }
    validation_metrics = {
        "stage1": compute_classification_metrics(
            _build_stage1_training_frame(validation, settings).y.astype(int),
            stage1_valid_proba,
            threshold=stage1_threshold,
        ),
        "stage2": (
            compute_classification_metrics(stage2_valid_input.y.astype(int), stage2_valid_proba, threshold=base_rate)
            if not stage2_valid_input.frame.empty
            else {"sample_count": 0.0}
        ),
        "end_to_end": (
            compute_pnl_metrics(
                y_true=validation.y.astype(int),
                stage1_probabilities=stage1_valid_proba,
                stage2_probabilities=full_stage2_valid_proba,
                stage1_threshold=stage1_threshold,
                buy_threshold=base_rate,
            )
            if not validation.frame.empty
            else {"sample_count": 0.0, "pnl_per_sample": 0.0, "coverage": 0.0}
        ),
    }
    probability_summary = {
        "stage1_prob_oof_train": _summarize_probability_series(stage1_oof),
        "stage1_prob_validation": _summarize_probability_series(stage1_valid_proba),
    }
    probability_reference = {
        "stage1_prob_oof_train": _serialize_probability_reference(stage1_oof),
    }

    return (
        stage1_model,
        stage1_calibrator,
        stage2_model,
        stage2_calibrator,
        stage1_valid_proba,
        stage2_valid_proba,
        stage1_threshold,
        base_rate,
        threshold_scan,
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
    _, _, _, _, split = purged_chronological_time_window_split(
        training,
        validation_window_days=validation_window_days,
        purge_rows=purge_rows,
    )

    development = _slice_training_frame(training, split.train_slice)
    validation = _slice_training_frame(training, split.valid_slice)
    (
        stage1_model,
        stage1_calibrator,
        stage2_model,
        stage2_calibrator,
        _stage1_valid_proba,
        _stage2_valid_proba,
        stage1_threshold,
        buy_threshold,
        threshold_scan,
        probability_summary,
        probability_reference,
        metrics,
    ) = _train_two_stage_for_split(development, validation, settings)

    train_rows = split.train_end - split.train_start
    valid_rows = split.valid_end - split.valid_start
    walk_forward_results: list[WalkForwardFoldResult] = []
    walk_forward_fold_details: list[dict[str, Any]] = []
    walk_forward_splits = build_walk_forward_splits(
        training,
        min_train_size=max(train_rows, 1),
        validation_size=max(valid_rows, 1),
        step_size=max(valid_rows // 2, 1),
        purge_rows=purge_rows,
    )
    for fold_index, fold_split in enumerate(walk_forward_splits, start=1):
        fold_train = _slice_training_frame(training, fold_split.train_slice)
        fold_valid = _slice_training_frame(training, fold_split.valid_slice)
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _fold_stage1_threshold,
            _fold_buy_threshold,
            _fold_threshold_scan,
            _fold_probability_summary,
            _fold_probability_reference,
            fold_metrics,
        ) = _train_two_stage_for_split(fold_train, fold_valid, settings)
        walk_forward_results.append(
            WalkForwardFoldResult(
                fold_index=fold_index,
                split=fold_split,
                metrics=fold_metrics["validation"]["end_to_end"],
                validation_probabilities=pd.Series(dtype="float64"),
            )
        )
        walk_forward_fold_details.append(
            {
                "fold_index": fold_index,
                "train_start": fold_split.train_start,
                "train_end": fold_split.train_end,
                "valid_start": fold_split.valid_start,
                "valid_end": fold_split.valid_end,
                "purge_rows": fold_split.purge_rows,
                "stage1_threshold": _fold_stage1_threshold,
                "buy_threshold": _fold_buy_threshold,
                "base_rate": _fold_buy_threshold,
                "metrics": fold_metrics["validation"],
            }
        )

    development_frame = development.frame
    validation_frame = validation.frame
    stage2_feature_columns = [*training.feature_columns, DEFAULT_STAGE1_PROBABILITY_COLUMN]

    return TwoStageTrainingArtifacts(
        stage1_model=stage1_model,
        stage1_calibrator=stage1_calibrator,
        stage2_model=stage2_model,
        stage2_calibrator=stage2_calibrator,
        feature_columns=training.feature_columns,
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
        walk_forward_summary=summarize_walk_forward(walk_forward_results),
        walk_forward_results=walk_forward_results,
        walk_forward_fold_details=walk_forward_fold_details,
        base_rate=buy_threshold,
        stage1_threshold_scan=threshold_scan,
        stage1_probability_summary=probability_summary,
        stage1_probability_reference=probability_reference,
    )
