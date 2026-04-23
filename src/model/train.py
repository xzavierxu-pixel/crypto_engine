from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.calibration.base import CalibrationPlugin
from src.calibration.none import NoCalibration
from src.calibration.registry import create_calibration_plugin
from src.core.config import Settings
from src.data.dataset_builder import TrainingFrame
from src.model.base import ModelPlugin
from src.model.evaluation import (
    WalkForwardFoldResult,
    build_walk_forward_splits,
    compute_classification_metrics,
    purged_chronological_split,
    purged_chronological_time_window_split,
    summarize_walk_forward,
)
from src.model.registry import create_model_plugin


@dataclass(frozen=True)
class TrainingArtifacts:
    model: ModelPlugin
    calibrator: CalibrationPlugin
    feature_columns: list[str]
    train_metrics: dict[str, float]
    train_window: dict[str, str | int | None]
    validation_window: dict[str, str | int | None]
    raw_validation_probabilities: pd.Series
    validation_probabilities: pd.Series
    raw_validation_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    walk_forward_summary: dict[str, float | int]
    walk_forward_results: list[WalkForwardFoldResult]


def _slice_training_frame(training: TrainingFrame, frame_slice: slice) -> TrainingFrame:
    return TrainingFrame(
        frame=training.frame.iloc[frame_slice].reset_index(drop=True),
        feature_columns=training.feature_columns,
        target_column=training.target_column,
        sample_weight_column=training.sample_weight_column,
    )


def _build_calibration_oof_predictions(
    training: TrainingFrame,
    settings: Settings,
    calibration_fraction: float,
    purge_rows: int,
    model_plugin_name: str | None = None,
    model_plugin_params: dict | None = None,
) -> tuple[pd.Series, pd.Series]:
    if len(training.frame) < 200:
        return pd.Series(dtype="float64"), pd.Series(dtype="int64")

    validation_size = max(int(len(training.frame) * calibration_fraction), 1)
    min_train_size = max(validation_size * 2, 200)
    splits = build_walk_forward_splits(
        training,
        min_train_size=min_train_size,
        validation_size=validation_size,
        step_size=validation_size,
        purge_rows=purge_rows,
    )
    if not splits:
        return pd.Series(dtype="float64"), pd.Series(dtype="int64")

    oof_probabilities: list[pd.Series] = []
    oof_targets: list[pd.Series] = []
    for split in splits:
        fold_training = _slice_training_frame(training, split.train_slice)
        X_fold_valid = training.X.iloc[split.valid_slice]
        y_fold_valid = training.y.astype(int).iloc[split.valid_slice]
        fold_model = create_model_plugin(
            settings,
            plugin_name=model_plugin_name,
            plugin_params=model_plugin_params,
        )
        fold_model.fit(
            fold_training.X,
            fold_training.y.astype(int),
            X_valid=X_fold_valid,
            y_valid=y_fold_valid,
            sample_weight=fold_training.sample_weight,
            sample_weight_valid=training.sample_weight.iloc[split.valid_slice]
            if training.sample_weight is not None
            else None,
        )
        oof_probabilities.append(fold_model.predict_proba(X_fold_valid))
        oof_targets.append(y_fold_valid)

    if not oof_probabilities:
        return pd.Series(dtype="float64"), pd.Series(dtype="int64")

    return pd.concat(oof_probabilities).sort_index(), pd.concat(oof_targets).sort_index()


def _select_calibrator(
    settings: Settings,
    raw_probabilities: pd.Series,
    y_true: pd.Series,
) -> CalibrationPlugin:
    if raw_probabilities.empty or len(raw_probabilities) < 100 or y_true.nunique() < 2:
        return NoCalibration()

    base_metrics = compute_classification_metrics(y_true, raw_probabilities)
    calibrator = create_calibration_plugin(settings)
    calibrator.fit(raw_probabilities, y_true)
    calibrated_probabilities = calibrator.transform(raw_probabilities)
    calibrated_metrics = compute_classification_metrics(y_true, calibrated_probabilities)

    if (
        calibrated_metrics["log_loss"] < base_metrics["log_loss"]
        and calibrated_metrics["brier_score"] <= base_metrics["brier_score"]
    ):
        return calibrator

    return NoCalibration()


def _fit_model_and_calibrator(
    training: TrainingFrame,
    settings: Settings,
    calibration_fraction: float,
    purge_rows: int,
    model_plugin_name: str | None = None,
    model_plugin_params: dict | None = None,
) -> tuple[ModelPlugin, CalibrationPlugin]:
    model = create_model_plugin(
        settings,
        plugin_name=model_plugin_name,
        plugin_params=model_plugin_params,
    )
    model.fit(training.X, training.y.astype(int), sample_weight=training.sample_weight)
    raw_calibration_proba, calibration_targets = _build_calibration_oof_predictions(
        training,
        settings,
        calibration_fraction=calibration_fraction,
        purge_rows=purge_rows,
        model_plugin_name=model_plugin_name,
        model_plugin_params=model_plugin_params,
    )
    calibrator = _select_calibrator(settings, raw_calibration_proba, calibration_targets)
    return model, calibrator


def train_model(
    training: TrainingFrame,
    settings: Settings,
    validation_window_days: int = 30,
    validation_fraction: float | None = None,
    calibration_fraction: float = 0.15,
    purge_rows: int = 1,
    threshold: float = 0.5,
    model_plugin_name: str | None = None,
    model_plugin_params: dict | None = None,
) -> TrainingArtifacts:
    if validation_fraction is not None:
        _, X_valid, _, y_valid, split = purged_chronological_split(
            training,
            validation_fraction=validation_fraction,
            purge_rows=purge_rows,
        )
    else:
        _, X_valid, _, y_valid, split = purged_chronological_time_window_split(
            training,
            validation_window_days=validation_window_days,
            purge_rows=purge_rows,
        )

    development = _slice_training_frame(training, split.train_slice)
    model, calibrator = _fit_model_and_calibrator(
        development,
        settings,
        calibration_fraction=calibration_fraction,
        purge_rows=purge_rows,
        model_plugin_name=model_plugin_name,
        model_plugin_params=model_plugin_params,
    )

    raw_valid_proba = model.predict_proba(X_valid)
    calibrated_valid_proba = calibrator.transform(raw_valid_proba)
    development_probabilities = calibrator.transform(model.predict_proba(development.X))
    train_metrics = compute_classification_metrics(
        development.y.astype(int),
        development_probabilities,
        threshold=threshold,
    )
    raw_validation_metrics = compute_classification_metrics(y_valid, raw_valid_proba, threshold=threshold)
    validation_metrics = compute_classification_metrics(y_valid, calibrated_valid_proba, threshold=threshold)
    development_frame = development.frame
    validation_frame = training.frame.iloc[split.valid_slice].reset_index(drop=True)

    train_rows = split.train_end - split.train_start
    valid_rows = split.valid_end - split.valid_start
    walk_forward_results: list[WalkForwardFoldResult] = []
    walk_forward_splits = build_walk_forward_splits(
        training,
        min_train_size=max(train_rows, 1),
        validation_size=max(valid_rows, 1),
        step_size=valid_rows,
        purge_rows=purge_rows,
    )

    for fold_index, fold_split in enumerate(walk_forward_splits, start=1):
        fold_training = _slice_training_frame(training, fold_split.train_slice)
        X_fold_valid = training.X.iloc[fold_split.valid_slice]
        y_fold_valid = training.y.astype(int).iloc[fold_split.valid_slice]
        fold_valid_weights = (
            training.sample_weight.iloc[fold_split.valid_slice] if training.sample_weight is not None else None
        )

        fold_model, fold_calibrator = _fit_model_and_calibrator(
            fold_training,
            settings,
            calibration_fraction=calibration_fraction,
            purge_rows=purge_rows,
            model_plugin_name=model_plugin_name,
            model_plugin_params=model_plugin_params,
        )
        raw_fold_proba = fold_model.predict_proba(X_fold_valid)
        calibrated_fold_proba = fold_calibrator.transform(raw_fold_proba)
        walk_forward_results.append(
            WalkForwardFoldResult(
                fold_index=fold_index,
                split=fold_split,
                metrics=compute_classification_metrics(
                    y_fold_valid,
                    calibrated_fold_proba,
                    threshold=threshold,
                ),
                validation_probabilities=calibrated_fold_proba,
            )
        )

    return TrainingArtifacts(
        model=model,
        calibrator=calibrator,
        feature_columns=training.feature_columns,
        train_metrics=train_metrics,
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
        raw_validation_probabilities=raw_valid_proba,
        validation_probabilities=calibrated_valid_proba,
        raw_validation_metrics=raw_validation_metrics,
        validation_metrics=validation_metrics,
        walk_forward_summary=summarize_walk_forward(walk_forward_results),
        walk_forward_results=walk_forward_results,
    )
