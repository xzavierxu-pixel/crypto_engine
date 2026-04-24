from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.model.artifacts import discover_latest_artifact_dir, load_two_stage_artifacts
from src.model.train import train_two_stage_model


def _train_artifacts():
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=2500, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(2500)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(2500)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(2500)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(2500)],
            "volume": [10 + index for index in range(2500)],
        }
    )
    training = build_training_frame(frame, settings, horizon_name="5m")
    return settings, train_two_stage_model(training, settings, validation_window_days=1)


def test_load_two_stage_artifacts_from_manifest_and_directory(tmp_path: Path) -> None:
    settings, artifacts = _train_artifacts()
    artifact_dir = tmp_path / "bundle"
    artifact_dir.mkdir()
    artifacts.stage1_model.save(artifact_dir / "lightgbm_stage1.stage1.pkl")
    artifacts.stage2_model.save(artifact_dir / "lightgbm_stage2.stage2.pkl")
    artifacts.stage1_calibrator.save(artifact_dir / f"{artifacts.stage1_calibrator.name}.stage1.pkl")
    artifacts.stage2_calibrator.save(artifact_dir / f"{artifacts.stage2_calibrator.name}.stage2.pkl")
    manifest = {
        "model_plugins": {"stage1": "lightgbm_stage1", "stage2": "lightgbm_stage2"},
        "calibration_plugins": {
            "stage1": artifacts.stage1_calibrator.name,
            "stage2": artifacts.stage2_calibrator.name,
        },
        "stage2_feature_columns": artifacts.stage2_feature_columns,
        "stage1_threshold": artifacts.stage1_threshold,
        "up_threshold": artifacts.up_threshold,
        "down_threshold": artifacts.down_threshold,
        "margin_threshold": artifacts.margin_threshold,
        "base_rate": artifacts.base_rate,
        "stage1_probability_reference_path": "stage1_probability_reference.json",
        "stage2_direction_reference_path": "stage2_direction_reference.json",
    }
    (artifact_dir / "stage1_probability_reference.json").write_text(
        json.dumps(artifacts.stage1_probability_reference),
        encoding="utf-8",
    )
    (artifact_dir / "stage2_direction_reference.json").write_text(
        json.dumps(artifacts.stage2_direction_reference),
        encoding="utf-8",
    )
    (artifact_dir / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = load_two_stage_artifacts(settings, artifact_dir=artifact_dir)
    assert bundle.stage1_threshold == artifacts.stage1_threshold
    assert bundle.up_threshold == artifacts.up_threshold
    assert bundle.down_threshold == artifacts.down_threshold
    assert bundle.margin_threshold == artifacts.margin_threshold
    assert bundle.feature_columns == artifacts.feature_columns
    assert bundle.stage2_feature_columns == artifacts.stage2_feature_columns
    assert bundle.stage1_reference_probabilities
    assert bundle.stage2_direction_reference


def test_discover_latest_artifact_dir_returns_most_recent_manifest_directory(tmp_path: Path) -> None:
    older_dir = tmp_path / "older"
    newer_dir = tmp_path / "newer"
    older_dir.mkdir()
    newer_dir.mkdir()
    older_manifest = older_dir / "artifact_manifest.json"
    newer_manifest = newer_dir / "artifact_manifest.json"
    older_manifest.write_text("{}", encoding="utf-8")
    newer_manifest.write_text("{}", encoding="utf-8")
    older_timestamp = older_manifest.stat().st_mtime - 10
    newer_timestamp = newer_manifest.stat().st_mtime + 10
    older_manifest.touch()
    newer_manifest.touch()
    import os
    os.utime(older_manifest, (older_timestamp, older_timestamp))
    os.utime(newer_manifest, (newer_timestamp, newer_timestamp))
    assert discover_latest_artifact_dir(tmp_path) == newer_dir
