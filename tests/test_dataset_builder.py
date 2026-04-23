from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.core.config import DatasetConfig, load_settings
from src.data.dataset_builder import TrainingFrame, build_training_frame
from src.model.train import _build_stage1_training_frame


def test_build_training_frame_drops_incomplete_rows_and_exposes_feature_columns() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=360, freq="1min"),
            "open": [100 + index for index in range(360)],
            "high": [101 + index for index in range(360)],
            "low": [99 + index for index in range(360)],
            "close": [100 + index for index in range(360)],
            "volume": [10 + index for index in range(360)],
        }
    )

    training = build_training_frame(frame, settings, horizon_name="5m")

    assert not training.frame.empty
    assert training.target_column == "target"
    assert "ret_1" in training.feature_columns
    assert "regime_vol_ratio" in training.feature_columns
    assert "compression_score" in training.feature_columns
    assert training.frame["target"].notna().all()
    assert training.X.columns.tolist() == training.feature_columns
    assert len(training.X) == len(training.y)
    assert training.sample_weight is not None
    assert training.sample_weight.between(0.5, 3.0).all()
    assert training.frame["timestamp"].min() >= pd.Timestamp("2024-01-01T13:40:00Z")


def test_build_training_frame_respects_dataset_timerange() -> None:
    settings = load_settings()
    custom_dataset = DatasetConfig(
        train_start="2024-01-01T14:00:00Z",
        train_end="2024-01-01T15:00:00Z",
        strict_grid_only=True,
        drop_incomplete_candles=True,
    )
    scoped_settings = replace(settings, dataset=custom_dataset)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=360, freq="1min"),
            "open": [100 + index for index in range(360)],
            "high": [101 + index for index in range(360)],
            "low": [99 + index for index in range(360)],
            "close": [100 + index for index in range(360)],
            "volume": [10 + index for index in range(360)],
        }
    )

    training = build_training_frame(frame, scoped_settings, horizon_name="5m")

    assert not training.frame.empty
    assert training.frame["timestamp"].min() >= pd.Timestamp("2024-01-01T14:00:00Z")
    assert training.frame["timestamp"].max() <= pd.Timestamp("2024-01-01T15:00:00Z")
    assert training.frame["target"].eq(1.0).all()


def test_stage1_training_frame_uses_boundary_weight_only() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=6, freq="1min"),
            "open": [100.0] * 6,
            "high": [101.0] * 6,
            "low": [99.0] * 6,
            "close": [100.0] * 6,
            "volume": [10.0] * 6,
            "abs_return": [0.0004, 0.0005, 0.0006, 0.0007, 0.0001, 0.0002],
            "target": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "sample_weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "stage1_sample_weight": [0.4, 0.5, 0.6, 0.7, 0.1, 0.2],
        }
    )
    custom_training = TrainingFrame(
        frame=frame,
        feature_columns=[],
        target_column="target",
        sample_weight_column="sample_weight",
    )

    stage1_training = _build_stage1_training_frame(custom_training, settings)

    assert stage1_training.frame["stage1_sample_weight"].tolist() == frame["stage1_sample_weight"].tolist()
