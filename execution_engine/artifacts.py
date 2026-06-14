from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from execution_engine.config import BaselineConfig, PriceEstimatorConfig


@dataclass(frozen=True)
class BaselineArtifact:
    artifact_dir: Path
    manifest: dict[str, Any]
    model_path: Path
    calibrator_path: Path
    model_plugin: str
    calibration_plugin: str
    feature_columns: list[str]
    t_up: float
    t_down: float
    threshold_policy: dict[str, Any]
    target_semantics: dict[str, Any]


@dataclass(frozen=True)
class PriceEstimatorArtifact:
    artifact_dir: Path
    manifest: dict[str, Any]
    model_path: Path
    model: Any
    feature_columns: list[str]
    prediction_column: str
    selected_side_column: str
    yes_value: str
    no_value: str
    round_decimals: int
    best_ask_offset: float
    fallback_price_mode: str
    prediction_transform: str | None = None


def _first_existing(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidate files exists under {root}: {candidates}")


def load_baseline_artifact(config: BaselineConfig) -> BaselineArtifact:
    artifact_dir = Path(config.artifact_dir)
    manifest_path = artifact_dir / config.manifest_file
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    model_plugin = str(manifest["model_plugin"])
    calibration_plugin = str(manifest["calibration_plugin"])
    model_file = config.model_file or f"{model_plugin}.binary.pkl"
    calibrator_file = config.calibrator_file or f"{calibration_plugin}.binary.pkl"
    model_path = _first_existing(artifact_dir, [model_file])
    calibrator_path = _first_existing(artifact_dir, [calibrator_file])
    feature_columns = [str(column) for column in manifest["feature_columns"]]

    t_up = manifest.get("t_up")
    t_down = manifest.get("t_down")
    if t_up is None or t_down is None:
        threshold_report = manifest.get("threshold_constraint_report", {})
        t_up = threshold_report.get("side_guardrail_t_up")
        t_down = threshold_report.get("side_guardrail_t_down")
    if t_up is None or t_down is None:
        raise ValueError("Baseline artifact does not define t_up and t_down thresholds.")

    target_semantics = manifest.get("target_semantics") or manifest.get("model_target_semantics") or {}
    if not isinstance(target_semantics, dict):
        target_semantics = {}
    threshold_policy = manifest.get("threshold_policy") or {}
    if not isinstance(threshold_policy, dict):
        threshold_policy = {}

    return BaselineArtifact(
        artifact_dir=artifact_dir,
        manifest=manifest,
        model_path=model_path,
        calibrator_path=calibrator_path,
        model_plugin=model_plugin,
        calibration_plugin=calibration_plugin,
        feature_columns=feature_columns,
        t_up=float(t_up),
        t_down=float(t_down),
        threshold_policy=dict(threshold_policy),
        target_semantics=dict(target_semantics),
    )


def _load_pickle_model(path: Path) -> Any:
    if path.suffix.lower() == ".cbm":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(path)
        return model
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        try:
            import joblib  # type: ignore
        except ImportError:
            raise
        return joblib.load(path)


def load_price_estimator_artifact(config: PriceEstimatorConfig) -> PriceEstimatorArtifact | None:
    if not config.enabled:
        return None
    if not config.artifact_dir:
        raise ValueError("price_estimator.artifact_dir is required when price_estimator.enabled is true.")
    artifact_dir = Path(config.artifact_dir)
    manifest_path = artifact_dir / config.manifest_file
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    model_file = config.model_file or manifest.get("model_file") or "q80.binary.pkl"
    model_path = _first_existing(artifact_dir, [str(model_file)])
    feature_columns_payload = manifest.get("feature_columns", [])
    feature_columns_file = manifest.get("feature_columns_file")
    if not feature_columns_payload and feature_columns_file:
        with (artifact_dir / str(feature_columns_file)).open("r", encoding="utf-8") as handle:
            feature_columns_payload = json.load(handle)
    feature_columns = [str(column) for column in feature_columns_payload]
    if not feature_columns:
        raise ValueError("Price estimator artifact manifest must define feature_columns.")

    return PriceEstimatorArtifact(
        artifact_dir=artifact_dir,
        manifest=manifest,
        model_path=model_path,
        model=_load_pickle_model(model_path),
        feature_columns=feature_columns,
        prediction_column=str(manifest.get("prediction_column", config.prediction_column)),
        selected_side_column=str(manifest.get("selected_side_column", config.selected_side_column)),
        yes_value=str(manifest.get("yes_value", config.yes_value)),
        no_value=str(manifest.get("no_value", config.no_value)),
        round_decimals=int(manifest.get("round_decimals", config.round_decimals)),
        best_ask_offset=float(manifest.get("best_ask_offset", config.best_ask_offset)),
        fallback_price_mode=str(manifest.get("fallback_price_mode", config.fallback_price_mode)),
        prediction_transform=manifest.get("prediction_transform"),
    )

