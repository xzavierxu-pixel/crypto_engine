from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.registry import load_calibration_plugin
from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.model.infer import predict_frame
from src.model.registry import load_model_plugin
from src.model.train import train_model


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
    artifacts = train_model(training, settings, validation_window_days=1)

    assert not artifacts.validation_probabilities.empty
    assert artifacts.validation_probabilities.between(0.0, 1.0).all()
    assert "accuracy" in artifacts.validation_metrics
    assert "fold_count" in artifacts.walk_forward_summary

    output_dir = Path("artifacts/test_model_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"
    calibrator_path = output_dir / "calibrator.pkl"
    artifacts.model.save(model_path)
    artifacts.calibrator.save(calibrator_path)

    loaded_model = load_model_plugin(settings.model.active_plugin, str(model_path))
    loaded_calibrator = load_calibration_plugin(artifacts.calibrator.name, str(calibrator_path))
    predictions = predict_frame(
        training.frame.tail(5),
        loaded_model,
        calibrator=loaded_calibrator,
        feature_columns=training.feature_columns,
    )

    assert len(predictions) == 5
    assert predictions.between(0.0, 1.0).all()


def test_train_model_pipeline_with_logistic_plugin_override() -> None:
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
    artifacts = train_model(
        training,
        settings,
        validation_window_days=1,
        model_plugin_name="logistic",
    )

    assert not artifacts.validation_probabilities.empty
    assert artifacts.validation_probabilities.between(0.0, 1.0).all()

    output_dir = Path("artifacts/test_model_pipeline_logistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"
    calibrator_path = output_dir / "calibrator.pkl"
    artifacts.model.save(model_path)
    artifacts.calibrator.save(calibrator_path)

    loaded_model = load_model_plugin("logistic", str(model_path))
    loaded_calibrator = load_calibration_plugin(artifacts.calibrator.name, str(calibrator_path))
    predictions = predict_frame(
        training.frame.tail(5),
        loaded_model,
        calibrator=loaded_calibrator,
        feature_columns=training.feature_columns,
    )

    assert len(predictions) == 5
    assert predictions.between(0.0, 1.0).all()
