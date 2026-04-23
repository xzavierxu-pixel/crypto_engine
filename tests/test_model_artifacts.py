from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.model.artifacts import discover_latest_artifact_dir, load_two_stage_artifacts
from src.model.train import train_two_stage_model


def test_load_two_stage_artifacts_from_report_and_directory(tmp_path: Path) -> None:
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

    artifact_dir = tmp_path / "bundle"
    artifact_dir.mkdir()
    artifacts.stage1_model.save(artifact_dir / "lightgbm_stage1.stage1.pkl")
    artifacts.stage2_model.save(artifact_dir / "lightgbm_stage2.stage2.pkl")
    artifacts.stage1_calibrator.save(artifact_dir / f"{artifacts.stage1_calibrator.name}.stage1.pkl")
    artifacts.stage2_calibrator.save(artifact_dir / f"{artifacts.stage2_calibrator.name}.stage2.pkl")
    report = {
        "model_plugins": {"stage1": "lightgbm_stage1", "stage2": "lightgbm_stage2"},
        "calibration_plugins": {
            "stage1": artifacts.stage1_calibrator.name,
            "stage2": artifacts.stage2_calibrator.name,
        },
        "stage2_feature_columns": artifacts.stage2_feature_columns,
        "stage1_threshold": artifacts.stage1_threshold,
        "buy_threshold": artifacts.buy_threshold,
        "base_rate": artifacts.base_rate,
        "stage1_probability_reference_path": "stage1_probability_reference.json",
    }
    (artifact_dir / "stage1_probability_reference.json").write_text(
        json.dumps(artifacts.stage1_probability_reference),
        encoding="utf-8",
    )
    report_path = artifact_dir / "training_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    bundle = load_two_stage_artifacts(settings, artifact_dir=artifact_dir)

    assert bundle.stage1_threshold == artifacts.stage1_threshold
    assert bundle.buy_threshold == artifacts.buy_threshold
    assert bundle.feature_columns == artifacts.feature_columns
    assert bundle.stage2_feature_columns == artifacts.stage2_feature_columns
    assert bundle.stage1_reference_probabilities


def test_discover_latest_artifact_dir_returns_most_recent_report_directory(tmp_path: Path) -> None:
    older_dir = tmp_path / "older"
    newer_dir = tmp_path / "newer"
    older_dir.mkdir()
    newer_dir.mkdir()
    older_report = older_dir / "training_report.json"
    newer_report = newer_dir / "training_report.json"
    older_report.write_text("{}", encoding="utf-8")
    newer_report.write_text("{}", encoding="utf-8")

    older_timestamp = older_report.stat().st_mtime - 10
    newer_timestamp = newer_report.stat().st_mtime + 10
    older_report.touch()
    newer_report.touch()

    import os
    os.utime(older_report, (older_timestamp, older_timestamp))
    os.utime(newer_report, (newer_timestamp, newer_timestamp))

    discovered = discover_latest_artifact_dir(tmp_path)

    assert discovered == newer_dir


def test_load_two_stage_artifacts_auto_discovers_latest_bundle_from_artifacts_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
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

    artifact_dir = tmp_path / "artifacts" / "auto-bundle"
    artifact_dir.mkdir(parents=True)
    artifacts.stage1_model.save(artifact_dir / "lightgbm_stage1.stage1.pkl")
    artifacts.stage2_model.save(artifact_dir / "lightgbm_stage2.stage2.pkl")
    artifacts.stage1_calibrator.save(artifact_dir / f"{artifacts.stage1_calibrator.name}.stage1.pkl")
    artifacts.stage2_calibrator.save(artifact_dir / f"{artifacts.stage2_calibrator.name}.stage2.pkl")
    report = {
        "model_plugins": {"stage1": "lightgbm_stage1", "stage2": "lightgbm_stage2"},
        "calibration_plugins": {
            "stage1": artifacts.stage1_calibrator.name,
            "stage2": artifacts.stage2_calibrator.name,
        },
        "stage2_feature_columns": artifacts.stage2_feature_columns,
        "stage1_threshold": artifacts.stage1_threshold,
        "buy_threshold": artifacts.buy_threshold,
        "base_rate": artifacts.base_rate,
        "stage1_probability_reference_path": "stage1_probability_reference.json",
    }
    (artifact_dir / "stage1_probability_reference.json").write_text(
        json.dumps(artifacts.stage1_probability_reference),
        encoding="utf-8",
    )
    (artifact_dir / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    bundle = load_two_stage_artifacts(settings)

    assert bundle.feature_columns == artifacts.feature_columns
    assert bundle.stage1_reference_probabilities
