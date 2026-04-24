from __future__ import annotations

import json
import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd

from scripts import train_model as train_model_script
from src.core.config import load_settings
from src.core.constants import DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN, DEFAULT_STAGE2_TARGET_COLUMN
from src.data.dataset_builder import build_training_frame
from src.model.infer import predict_frame, predict_frame_multiclass
from src.model.train import (
    _build_stage1_training_frame,
    _build_stage2_training_frame,
    _fit_model,
    _tune_stage1_filter_threshold,
    train_two_stage_model,
)


def _build_frame(length: int = 2500) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=length, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(length)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(length)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(length)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(length)],
            "volume": [10 + index for index in range(length)],
        }
    )


def test_train_model_pipeline_and_roundtrip() -> None:
    settings = load_settings()
    training = build_training_frame(_build_frame(), settings, horizon_name="5m")
    artifacts = train_two_stage_model(training, settings, validation_window_days=1)
    assert "stage1" in artifacts.train_metrics
    assert "stage2" in artifacts.train_metrics
    assert "end_to_end" in artifacts.validation_metrics
    assert "macro_f1" in artifacts.train_metrics["stage2"]
    assert "class_pnl.up" in artifacts.validation_metrics["stage2"]
    assert "trade_pnl.pnl_per_sample" in artifacts.validation_metrics["stage2"]
    assert "trade_pnl.pnl_per_sample" in artifacts.validation_metrics["end_to_end"]
    assert artifacts.train_window["row_count"] > 0
    assert artifacts.validation_window["row_count"] > 0
    assert "stage1_threshold_search" in artifacts.threshold_search
    assert "stage2_threshold_search" in artifacts.threshold_search
    assert artifacts.threshold_search["stage1_threshold_search"]["best"]["threshold"] == artifacts.stage1_threshold
    assert artifacts.threshold_search["stage2_threshold_search"]["best"]["up_threshold"] == artifacts.up_threshold
    assert "train_vs_validation" in artifacts.stage1_probability_summary["stage1_prob_ks"]
    assert "stage2_direction_train" in artifacts.stage2_direction_reference

    output_dir = Path("artifacts/test_model_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_model_path = output_dir / "stage1.pkl"
    stage2_model_path = output_dir / "stage2.pkl"
    artifacts.stage1_model.save(stage1_model_path)
    artifacts.stage2_model.save(stage2_model_path)
    loaded_stage1 = artifacts.stage1_model.load(stage1_model_path)
    loaded_stage2 = artifacts.stage2_model.load(stage2_model_path)
    stage1_predictions = predict_frame(
        training.frame.tail(5),
        loaded_stage1,
        calibrator=artifacts.stage1_calibrator,
        feature_columns=training.feature_columns,
    )
    stage2_frame = training.frame.tail(5).copy()
    stage2_frame["stage1_prob"] = stage1_predictions
    stage2_predictions = predict_frame_multiclass(
        stage2_frame,
        loaded_stage2,
        feature_columns=artifacts.stage2_feature_columns,
    )
    assert list(stage2_predictions.columns) == ["p_down", "p_flat", "p_up"]
    assert len(stage2_predictions) == 5


def test_stage1_training_frame_uses_boundary_weight_only() -> None:
    settings = load_settings()
    training = build_training_frame(_build_frame(500), settings, horizon_name="5m")
    stage1_training = _build_stage1_training_frame(training, settings)
    assert stage1_training.sample_weight_column == DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN
    assert stage1_training.sample_weight is not None
    assert stage1_training.sample_weight.equals(stage1_training.frame[DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN])


def test_fit_model_uses_stage_specific_lightgbm_params(monkeypatch) -> None:
    settings = load_settings()
    captured: list[dict] = []

    class DummyModel:
        def fit(
            self,
            X_train,
            y_train,
            X_valid=None,
            y_valid=None,
            sample_weight=None,
            sample_weight_valid=None,
        ):
            return self

    def fake_create_model_plugin(_settings, plugin_name=None, plugin_params=None, stage=None):
        captured.append({"stage": stage, "plugin_params": plugin_params})
        return DummyModel()

    monkeypatch.setattr("src.model.train.create_model_plugin", fake_create_model_plugin)
    training_frame = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "target": [0, 1, 0, 0], "weight": [1.0] * 4})
    from src.data.dataset_builder import TrainingFrame
    training = TrainingFrame(frame=training_frame, feature_columns=["f1"], target_column="target", sample_weight_column="weight")
    _fit_model(training, settings, stage="stage1")
    multiclass = training_frame.copy()
    multiclass["target"] = [0, 1, 2, 1]
    multi_training = TrainingFrame(frame=multiclass, feature_columns=["f1"], target_column="target")
    _fit_model(multi_training, settings, stage="stage2")
    assert captured[0]["plugin_params"]["scale_pos_weight"] == 3.0
    assert captured[1]["plugin_params"]["objective"] == "multiclass"
    assert captured[1]["plugin_params"]["num_class"] == 3


def test_stage2_training_frame_filters_only_on_stage1_threshold() -> None:
    settings = load_settings()
    training = build_training_frame(_build_frame(500), settings, horizon_name="5m")
    stage2_training = _build_stage2_training_frame(
        training,
        pd.Series(0.8, index=training.frame.index, dtype="float64"),
        stage1_threshold=0.5,
    )
    assert stage2_training.sample_weight_column is None
    assert stage2_training.sample_weight is None
    assert stage2_training.target_column == DEFAULT_STAGE2_TARGET_COLUMN
    assert "stage1_prob" in stage2_training.feature_columns


def test_train_model_script_writes_new_threshold_and_window_fields(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "output"
    _build_frame().to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--validation-window-days",
            "1",
        ],
    )
    train_model_script.main()
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["up_threshold"] >= 0.3
    assert manifest["down_threshold"] >= 0.3
    assert "buy_threshold" not in manifest
    assert "stage1_class_ratio" in manifest["train_window"]
    assert "stage2_class_ratio" in manifest["validation_window"]
    assert "threshold_search_path" in manifest
    assert "stage2_direction_reference_path" in manifest
    assert (output_dir / "data_quality" / "dqc_summary.txt").exists()
    assert not (output_dir / "data_quality" / "dqc_report.json").exists()


def test_tune_stage1_filter_threshold_respects_coverage_band() -> None:
    y_true = pd.Series([1, 1, 1, 0, 0, 0], dtype="int64")
    probabilities = pd.Series([0.9, 0.85, 0.8, 0.7, 0.4, 0.2], dtype="float64")
    stage1_threshold, threshold_search = _tune_stage1_filter_threshold(
        y_true=y_true,
        probabilities=probabilities,
        coverage_min=0.50,
        coverage_max=0.70,
    )
    assert threshold_search["constraint_satisfied"] is True
    assert threshold_search["best"]["coverage"] >= 0.50
    assert threshold_search["best"]["coverage"] <= 0.70
    assert stage1_threshold == threshold_search["best"]["threshold"]
