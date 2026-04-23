from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import train_model as train_model_script
from src.calibration.registry import load_calibration_plugin
from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.model.infer import predict_frame
from src.model.registry import load_model_plugin
from src.model.train import train_two_stage_model


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
    assert report["model_plugins"]["stage1"] == "lightgbm_stage1"
    assert report["model_plugins"]["stage2"] == "lightgbm_stage2"
    assert report["feature_counts"]["stage1"] == len(report["feature_columns"])
    assert report["feature_counts"]["stage2"] == len(report["stage2_feature_columns"])
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
    assert threshold_search["best"]["stage1_threshold"] == report["stage1_threshold"]
    assert threshold_search["best"]["buy_threshold"] == report["buy_threshold"]
    assert reference["stage1_prob_oof_train"]["sample_count"] > 0
    assert reference["stage1_prob_oof_train"]["sample"]
