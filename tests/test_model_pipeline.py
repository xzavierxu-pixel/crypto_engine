from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

from scripts import train_model as train_model_script
from src.core.config import load_settings
from src.core.constants import DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN
from src.data.dataset_builder import build_training_frame
from src.model.infer import predict_frame
from src.model.train import _fit_model, train_binary_selective_model


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


def _unit_settings():
    settings = load_settings()
    return replace(settings, derivatives=replace(settings.derivatives, enabled=False))


def test_train_model_pipeline_and_roundtrip() -> None:
    settings = _unit_settings()
    training = build_training_frame(_build_frame(3500), settings, horizon_name="5m")
    artifacts = train_binary_selective_model(
        training,
        settings,
        train_days=1,
        validation_days=1,
        purge_rows=1,
    )
    assert "balanced_precision" in artifacts.train_metrics
    assert "coverage" in artifacts.validation_metrics
    assert artifacts.train_window["row_count"] > 0
    assert artifacts.validation_window["row_count"] > 0
    assert artifacts.threshold_search["best"]["t_up"] == artifacts.t_up
    assert artifacts.threshold_search["best"]["t_down"] == artifacts.t_down
    assert "side_guarded_best" in artifacts.threshold_search
    assert "train_vs_validation" in artifacts.probability_summary["p_up_ks"]
    assert not artifacts.threshold_frontier.empty
    assert not artifacts.boundary_slices.empty
    assert not artifacts.feature_importance.empty
    assert not artifacts.probability_deciles.empty
    assert "quote_volume" not in artifacts.feature_columns
    assert "taker_buy_volume" not in artifacts.feature_columns

    output_dir = Path("artifacts/test_model_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "binary.pkl"
    artifacts.model.save(model_path)
    loaded = artifacts.model.load(model_path)
    predictions = predict_frame(
        training.frame.tail(5),
        loaded,
        calibrator=artifacts.calibrator,
        feature_columns=artifacts.feature_columns,
    )
    assert len(predictions) == 5
    assert predictions.between(0.0, 1.0).all()


def test_training_frame_uses_prd_sample_weight() -> None:
    settings = _unit_settings()
    training = build_training_frame(_build_frame(500), settings, horizon_name="5m")
    assert training.sample_weight_column == DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN
    assert training.sample_weight is not None
    assert training.sample_weight.equals(training.frame[DEFAULT_STAGE1_SAMPLE_WEIGHT_COLUMN])


def test_fit_model_uses_binary_lightgbm_params(monkeypatch) -> None:
    settings = _unit_settings()
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
    from src.data.dataset_builder import TrainingFrame

    training_frame = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "target": [0, 1, 0, 0], "weight": [1.0] * 4})
    training = TrainingFrame(frame=training_frame, feature_columns=["f1"], target_column="target", sample_weight_column="weight")
    _fit_model(training, settings, stage="binary")
    assert captured[0]["plugin_params"]["scale_pos_weight"] == 3.0
    assert captured[0]["plugin_params"]["objective"] == "binary"


def test_train_model_script_writes_binary_artifacts_and_reports(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "output"
    _build_frame(3500).to_csv(input_path, index=False)
    monkeypatch.setattr(train_model_script, "load_settings", lambda _path="config/settings.yaml": _unit_settings())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--train-window-days",
            "1",
            "--validation-window-days",
            "1",
        ],
    )
    train_model_script.main()
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["t_up"] >= 0.50
    assert manifest["t_down"] <= 0.50
    assert manifest["objective"] == "weighted_binary_selective_direction"
    assert "metrics_path" in manifest
    assert "threshold_frontier_path" in manifest
    assert "boundary_slices_path" in manifest
    assert "regime_slices_path" in manifest
    assert "feature_importance_path" in manifest
    assert "probability_deciles_path" in manifest
    assert "false_up_slices_path" in manifest
    assert "false_down_slices_path" in manifest
    assert manifest["raw_metadata_feature_count"] == 0
    assert "data_availability" in manifest
    assert "threshold_constraint_report" in manifest
    assert "threshold_constraint_satisfied" in manifest["threshold_constraint_report"]
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "threshold_frontier.csv").exists()
    assert (output_dir / "boundary_slices.csv").exists()
    assert (output_dir / "regime_slices.csv").exists()
    assert (output_dir / "feature_importance.csv").exists()
    assert (output_dir / "probability_deciles.csv").exists()
    assert (output_dir / "false_up_slices.csv").exists()
    assert (output_dir / "false_down_slices.csv").exists()
    assert (output_dir / "data_quality" / "dqc_summary.txt").exists()
