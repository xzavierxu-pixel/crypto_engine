from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import train_model as train_model_script
from src.calibration.registry import load_calibration_plugin
from src.core.config import load_settings
from src.core.constants import DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN
from src.data.dataset_builder import build_training_frame
from src.model.infer import predict_frame
from src.model.registry import load_model_plugin
from src.model.train import (
    _build_stage2_training_frame,
    _build_stage1_training_frame,
    _fit_model,
    _tune_stage1_threshold,
    train_two_stage_model,
)


def test_train_model_pipeline_and_roundtrip() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=5000, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(5000)],
            "volume": [10 + index for index in range(5000)],
        }
    )

    training = build_training_frame(frame, settings, horizon_name="5m")
    artifacts = train_two_stage_model(training, settings, validation_window_days=1)

    assert "stage1" in artifacts.train_metrics
    assert "stage2" in artifacts.train_metrics
    assert "end_to_end" in artifacts.validation_metrics
    assert "accuracy" in artifacts.train_metrics["stage1"]
    assert "precision" in artifacts.train_metrics["stage1"]
    assert "recall" in artifacts.train_metrics["stage1"]
    assert "precision" in artifacts.train_metrics["stage2"]
    assert "recall" in artifacts.train_metrics["stage2"]
    assert "accuracy" in artifacts.validation_metrics["stage1"]
    assert "precision" in artifacts.validation_metrics["stage1"]
    assert "recall" in artifacts.validation_metrics["stage1"]
    assert "precision" in artifacts.validation_metrics["stage2"]
    assert "recall" in artifacts.validation_metrics["stage2"]
    assert artifacts.train_window["row_count"] > 0
    assert artifacts.validation_window["row_count"] > 0
    assert artifacts.walk_forward_summary == {"enabled": False, "fold_count": 0}
    assert artifacts.threshold_search["selection_data"] == "validation"
    assert artifacts.threshold_search["constraint_applied"] is True
    assert artifacts.threshold_search["best"]["stage1_threshold"] == artifacts.stage1_threshold
    assert artifacts.threshold_search["best"]["buy_threshold"] == artifacts.buy_threshold

    output_dir = Path("artifacts/test_model_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_model_path = output_dir / "stage1.pkl"
    stage2_model_path = output_dir / "stage2.pkl"
    stage1_calibrator_path = output_dir / "stage1_calibrator.pkl"
    stage2_calibrator_path = output_dir / "stage2_calibrator.pkl"
    artifacts.stage1_model.save(stage1_model_path)
    artifacts.stage2_model.save(stage2_model_path)
    artifacts.stage1_calibrator.save(stage1_calibrator_path)
    artifacts.stage2_calibrator.save(stage2_calibrator_path)

    loaded_stage1 = load_model_plugin(settings.model.resolve_plugin(stage="stage1"), str(stage1_model_path))
    loaded_stage2 = load_model_plugin(settings.model.resolve_plugin(stage="stage2"), str(stage2_model_path))
    loaded_stage1_calibrator = load_calibration_plugin(artifacts.stage1_calibrator.name, str(stage1_calibrator_path))
    loaded_stage2_calibrator = load_calibration_plugin(artifacts.stage2_calibrator.name, str(stage2_calibrator_path))
    stage1_predictions = predict_frame(
        training.frame.tail(5),
        loaded_stage1,
        calibrator=loaded_stage1_calibrator,
        feature_columns=training.feature_columns,
    )
    stage2_frame = training.frame.tail(5).copy()
    stage2_frame["stage1_prob"] = stage1_predictions
    stage2_predictions = predict_frame(
        stage2_frame,
        loaded_stage2,
        calibrator=loaded_stage2_calibrator,
        feature_columns=artifacts.stage2_feature_columns,
    )

    assert len(stage2_predictions) == 5
    assert stage2_predictions.between(0.0, 1.0).all()


def test_stage1_training_frame_uses_boundary_weight_only() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=500, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(500)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(500)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(500)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(500)],
            "volume": [10 + index for index in range(500)],
        }
    )
    training = build_training_frame(frame, settings, horizon_name="5m")
    stage1_training = _build_stage1_training_frame(training, settings)

    assert stage1_training.sample_weight_column == DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN
    assert stage1_training.sample_weight is not None
    assert stage1_training.sample_weight.equals(stage1_training.frame[DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN])


def test_fit_model_applies_scale_pos_weight_for_lightgbm_stages(monkeypatch) -> None:
    settings = load_settings()
    settings.model.active_plugins["stage1"] = "lightgbm_stage1"
    settings.model.active_plugins["stage2"] = "lightgbm_stage2"
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
        captured.append({"stage": stage, "plugin_name": plugin_name, "plugin_params": plugin_params})
        return DummyModel()

    monkeypatch.setattr("src.model.train.create_model_plugin", fake_create_model_plugin)

    training_frame = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 1, 0, 0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    from src.data.dataset_builder import TrainingFrame
    training = TrainingFrame(
        frame=training_frame,
        feature_columns=["f1"],
        target_column="target",
        sample_weight_column="weight",
    )

    _fit_model(training, settings, stage="stage1")
    _fit_model(training, settings, stage="stage2")

    assert captured[0]["plugin_params"] == {"scale_pos_weight": 3.0}
    assert captured[1]["plugin_params"] == {"scale_pos_weight": 3.0}


def test_stage2_training_frame_uses_unit_weights_by_omission() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=500, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(500)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(500)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(500)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(500)],
            "volume": [10 + index for index in range(500)],
        }
    )
    training = build_training_frame(frame, settings, horizon_name="5m")
    stage2_training = _build_stage2_training_frame(
        training,
        pd.Series(0.8, index=training.frame.index, dtype="float64"),
        settings,
    )

    assert stage2_training.sample_weight_column is None
    assert stage2_training.sample_weight is None


def test_train_model_script_writes_train_metrics_into_training_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=5000, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(5000)],
            "volume": [10 + index for index in range(5000)],
        }
    )
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "output"
    frame.to_csv(input_path, index=False)

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

    report = json.loads((output_dir / "training_report.json").read_text(encoding="utf-8"))
    assert (output_dir / "development_frame.parquet").exists()
    assert (output_dir / "validation_frame.parquet").exists()
    assert report["model_plugins"]["stage1"] == "lightgbm_stage1"
    assert report["model_plugins"]["stage2"] == "lightgbm_stage2"
    assert "feature_columns" not in report
    assert "feature_count" not in report
    assert "stage2_feature_columns" in report
    assert report["feature_counts"]["stage1"] > 0
    assert report["feature_counts"]["stage2"] == len(report["stage2_feature_columns"])
    assert report["feature_counts"]["stage2"] == report["feature_counts"]["stage1"] + 1
    assert report["train_metrics"]["stage1"]["sample_count"] > 0
    assert "accuracy" in report["train_metrics"]["stage1"]
    assert "precision" in report["train_metrics"]["stage1"]
    assert "recall" in report["train_metrics"]["stage1"]
    assert "precision" in report["train_metrics"]["stage2"]
    assert "recall" in report["train_metrics"]["stage2"]
    assert "precision" in report["validation_metrics"]["stage1"]
    assert "recall" in report["validation_metrics"]["stage1"]
    assert "precision" in report["validation_metrics"]["stage2"]
    assert "recall" in report["validation_metrics"]["stage2"]
    assert report["train_window"]["row_count"] > 0
    assert report["validation_window"]["row_count"] > 0
    assert "stage1_probability_reference" not in report
    assert "stage1_threshold_scan" not in report
    assert report["walk_forward_summary"] == {"enabled": False, "fold_count": 0}
    assert report["walk_forward_folds"] == []
    threshold_search = json.loads((output_dir / report["threshold_search_path"]).read_text(encoding="utf-8"))
    reference = json.loads((output_dir / report["stage1_probability_reference_path"]).read_text(encoding="utf-8"))
    assert report["threshold_selection_data"] == "validation"
    assert report["threshold_search_constraints"]["constraint_applied"] is True
    assert report["threshold_search_constraints"]["stage1_coverage_constraint"] == 0.6
    assert threshold_search["selection_data"] == "validation"
    assert threshold_search["best"]["stage1_threshold"] == report["stage1_threshold"]
    assert threshold_search["best"]["buy_threshold"] == report["buy_threshold"]
    assert "stage1_precision" in threshold_search["best"]
    assert "coverage" in threshold_search["best"]
    assert threshold_search["eligible_record_count"] == len(threshold_search["records"])
    assert all(record["coverage"] >= 0.6 for record in threshold_search["records"])
    assert "record_metric_summary" in threshold_search
    assert "coverage" in threshold_search["record_metric_summary"]
    assert "median" in threshold_search["record_metric_summary"]["coverage"]
    assert reference["stage1_prob_train"]["sample_count"] > 0
    assert reference["stage1_prob_train"]["sample"]


def test_train_model_script_can_reuse_cached_split_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=5000, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(5000)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(5000)],
            "volume": [10 + index for index in range(5000)],
        }
    )
    input_path = tmp_path / "input.csv"
    initial_output_dir = tmp_path / "initial_output"
    reused_output_dir = tmp_path / "reused_output"
    frame.to_csv(input_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(initial_output_dir),
            "--validation-window-days",
            "1",
        ],
    )
    train_model_script.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "--cached-split-dir",
            str(initial_output_dir),
            "--output-dir",
            str(reused_output_dir),
            "--validation-window-days",
            "1",
        ],
    )
    train_model_script.main()

    reused_report = json.loads((reused_output_dir / "training_report.json").read_text(encoding="utf-8"))
    reused_threshold_search = json.loads(
        (reused_output_dir / reused_report["threshold_search_path"]).read_text(encoding="utf-8")
    )
    assert reused_report["train_row_count"] > 0
    assert reused_report["train_window"]["row_count"] > 0
    assert reused_report["validation_window"]["row_count"] > 0
    assert reused_threshold_search["selection_data"] == "validation"
    assert reused_threshold_search["best"]["stage1_threshold"] == reused_report["stage1_threshold"]
    assert reused_threshold_search["best"]["buy_threshold"] == reused_report["buy_threshold"]


def test_tune_stage1_threshold_respects_coverage_constraint() -> None:
    stage1_y_true = pd.Series([1, 1, 1, 0, 0, 0], dtype="int64")
    stage1_probabilities = pd.Series([0.9, 0.85, 0.8, 0.7, 0.4, 0.2], dtype="float64")
    end_to_end_y_true = pd.Series([1, 1, 1, 0, 0, 0], dtype="int64")
    stage2_probabilities = pd.Series([0.8, 0.75, 0.7, 0.45, 0.4, 0.35], dtype="float64")

    stage1_threshold, buy_threshold, threshold_search = _tune_stage1_threshold(
        stage1_y_true=stage1_y_true,
        stage1_probabilities=stage1_probabilities,
        end_to_end_y_true=end_to_end_y_true,
        stage2_probabilities=stage2_probabilities,
        min_stage1_coverage=0.60,
        min_active_samples=1,
    )

    assert threshold_search["selection_data"] == "validation"
    assert threshold_search["constraint_applied"] is True
    assert threshold_search["constraint_satisfied"] is True
    assert threshold_search["fallback_reason"] is None
    assert threshold_search["best"]["coverage"] >= 0.60
    assert threshold_search["eligible_record_count"] == len(threshold_search["records"])
    assert all(record["coverage"] >= 0.60 for record in threshold_search["records"])
    assert threshold_search["record_metric_summary"]["coverage"]["min"] >= 0.60
    assert buy_threshold in threshold_search["buy_threshold_candidates"]


def test_tune_stage1_threshold_falls_back_to_best_coverage_when_constraint_fails() -> None:
    stage1_y_true = pd.Series([1, 1, 0, 0], dtype="int64")
    stage1_probabilities = pd.Series([0.9, 0.8, 0.7, 0.6], dtype="float64")
    end_to_end_y_true = pd.Series([1, 0, 1, 0], dtype="int64")
    stage2_probabilities = pd.Series([0.9, 0.4, 0.8, 0.3], dtype="float64")

    stage1_threshold, buy_threshold, threshold_search = _tune_stage1_threshold(
        stage1_y_true=stage1_y_true,
        stage1_probabilities=stage1_probabilities,
        end_to_end_y_true=end_to_end_y_true,
        stage2_probabilities=stage2_probabilities,
        min_stage1_coverage=1.1,
        min_active_samples=1,
    )

    assert threshold_search["constraint_satisfied"] is False
    assert threshold_search["fallback_reason"] is not None
    assert threshold_search["best"]["coverage"] == 1.0
    assert threshold_search["records"] == []
    assert threshold_search["record_metric_summary"]["coverage"]["max"] is None
    assert stage1_threshold in threshold_search["stage1_threshold_candidates"]
    assert buy_threshold in threshold_search["buy_threshold_candidates"]
