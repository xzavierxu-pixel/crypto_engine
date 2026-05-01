from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.model.artifacts import discover_latest_artifact_dir, load_binary_selective_artifacts
from src.model.train import train_binary_selective_model


def _train_artifacts():
    settings = load_settings()
    settings = replace(settings, derivatives=replace(settings.derivatives, enabled=False))
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=3500, freq="1min"),
            "open": [100 + np.sin(index / 4.0) * 3 for index in range(3500)],
            "high": [101 + np.sin(index / 4.0) * 3 for index in range(3500)],
            "low": [99 + np.sin(index / 4.0) * 3 for index in range(3500)],
            "close": [100 + np.sin(index / 3.0) * 4 for index in range(3500)],
            "volume": [10 + index for index in range(3500)],
        }
    )
    training = build_training_frame(frame, settings, horizon_name="5m")
    return settings, train_binary_selective_model(training, settings, train_days=1, validation_days=1)


def test_load_binary_selective_artifacts_from_manifest_and_directory(tmp_path: Path) -> None:
    settings, artifacts = _train_artifacts()
    artifact_dir = tmp_path / "bundle"
    artifact_dir.mkdir()
    artifacts.model.save(artifact_dir / "lightgbm.binary.pkl")
    artifacts.calibrator.save(artifact_dir / f"{artifacts.calibrator.name}.binary.pkl")
    manifest = {
        "model_plugin": "lightgbm",
        "calibration_plugin": artifacts.calibrator.name,
        "feature_columns": artifacts.feature_columns,
        "t_up": artifacts.t_up,
        "t_down": artifacts.t_down,
        "base_rate": artifacts.base_rate,
        "probability_reference_path": "probability_reference.json",
    }
    (artifact_dir / "probability_reference.json").write_text(
        json.dumps(artifacts.probability_reference),
        encoding="utf-8",
    )
    (artifact_dir / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = load_binary_selective_artifacts(settings, artifact_dir=artifact_dir)
    assert bundle.t_up == artifacts.t_up
    assert bundle.t_down == artifacts.t_down
    assert bundle.feature_columns == artifacts.feature_columns
    assert bundle.reference_probabilities


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
