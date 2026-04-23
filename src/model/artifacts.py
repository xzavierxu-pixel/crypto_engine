from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.calibration.base import CalibrationPlugin
from src.calibration.registry import load_calibration_plugin
from src.core.config import Settings
from src.core.constants import DEFAULT_STAGE1_PROBABILITY_COLUMN
from src.model.base import ModelPlugin
from src.model.registry import load_model_plugin


@dataclass(frozen=True)
class TwoStageArtifactBundle:
    stage1_model: ModelPlugin
    stage2_model: ModelPlugin
    stage1_calibrator: CalibrationPlugin
    stage2_calibrator: CalibrationPlugin
    feature_columns: list[str]
    stage2_feature_columns: list[str]
    stage1_threshold: float
    buy_threshold: float
    base_rate: float
    stage1_reference_probabilities: list[float]
    model_version: str | None
    training_report: dict[str, Any]


def discover_latest_artifact_dir(search_root: str | Path = "artifacts") -> Path | None:
    root = Path(search_root)
    if not root.exists():
        return None
    report_paths = [path for path in root.rglob("training_report.json") if path.is_file()]
    if not report_paths:
        return None
    latest_report = max(report_paths, key=lambda path: path.stat().st_mtime)
    return latest_report.parent


def _read_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_stage1_reference_probabilities(report: dict[str, Any], artifact_root: Path | None) -> list[float]:
    reference_path = report.get("stage1_probability_reference_path")
    if reference_path and artifact_root is not None:
        resolved_path = artifact_root / reference_path
        if resolved_path.exists():
            reference_payload = _read_report(resolved_path)
            train_key = "stage1_prob_train" if "stage1_prob_train" in reference_payload else "stage1_prob_oof_train"
            return [float(value) for value in reference_payload.get(train_key, {}).get("sample", [])]

    return [
        float(value)
        for value in report.get("stage1_probability_reference", {})
        .get("stage1_prob_train", report.get("stage1_probability_reference", {}).get("stage1_prob_oof_train", {}))
        .get("sample", [])
    ]


def load_two_stage_artifacts(
    settings: Settings,
    *,
    report_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    stage1_model_path: str | Path | None = None,
    stage2_model_path: str | Path | None = None,
    stage1_calibrator_path: str | Path | None = None,
    stage2_calibrator_path: str | Path | None = None,
) -> TwoStageArtifactBundle:
    resolved_report_path: Path | None = None
    resolved_artifact_dir: Path | None = Path(artifact_dir) if artifact_dir is not None else None
    if resolved_artifact_dir is None and report_path is None:
        resolved_artifact_dir = discover_latest_artifact_dir()
    if report_path is not None:
        resolved_report_path = Path(report_path)
    elif resolved_artifact_dir is not None:
        candidate = resolved_artifact_dir / "training_report.json"
        if candidate.exists():
            resolved_report_path = candidate

    report: dict[str, Any] = _read_report(resolved_report_path) if resolved_report_path is not None else {}
    artifact_root = resolved_artifact_dir if resolved_artifact_dir is not None else (
        resolved_report_path.parent if resolved_report_path is not None else None
    )

    model_plugins = report.get("model_plugins", {})
    calibration_plugins = report.get("calibration_plugins", {})
    stage1_model_name = model_plugins.get("stage1", settings.model.resolve_plugin(stage="stage1"))
    stage2_model_name = model_plugins.get("stage2", settings.model.resolve_plugin(stage="stage2"))
    stage1_calibrator_name = calibration_plugins.get("stage1", settings.calibration.resolve_plugin(stage="stage1"))
    stage2_calibrator_name = calibration_plugins.get("stage2", settings.calibration.resolve_plugin(stage="stage2"))

    if artifact_root is not None:
        stage1_model_path = stage1_model_path or artifact_root / f"{stage1_model_name}.stage1.pkl"
        stage2_model_path = stage2_model_path or artifact_root / f"{stage2_model_name}.stage2.pkl"
        stage1_calibrator_path = stage1_calibrator_path or artifact_root / f"{stage1_calibrator_name}.stage1.pkl"
        stage2_calibrator_path = stage2_calibrator_path or artifact_root / f"{stage2_calibrator_name}.stage2.pkl"

    if None in {stage1_model_path, stage2_model_path, stage1_calibrator_path, stage2_calibrator_path}:
        raise ValueError("Two-stage artifact paths are incomplete. Provide report_path or all explicit paths.")

    stage1_reference_probabilities = _load_stage1_reference_probabilities(report, artifact_root)
    stage2_feature_columns = list(report.get("stage2_feature_columns", []))
    feature_columns = list(report.get("feature_columns", []))
    if not feature_columns and stage2_feature_columns:
        feature_columns = [
            column for column in stage2_feature_columns
            if column != DEFAULT_STAGE1_PROBABILITY_COLUMN
        ]

    return TwoStageArtifactBundle(
        stage1_model=load_model_plugin(stage1_model_name, str(stage1_model_path)),
        stage2_model=load_model_plugin(stage2_model_name, str(stage2_model_path)),
        stage1_calibrator=load_calibration_plugin(stage1_calibrator_name, str(stage1_calibrator_path)),
        stage2_calibrator=load_calibration_plugin(stage2_calibrator_name, str(stage2_calibrator_path)),
        feature_columns=feature_columns,
        stage2_feature_columns=stage2_feature_columns,
        stage1_threshold=float(report.get("stage1_threshold", 0.5)),
        buy_threshold=float(report.get("buy_threshold", report.get("base_rate", 0.5))),
        base_rate=float(report.get("base_rate", 0.5)),
        stage1_reference_probabilities=stage1_reference_probabilities,
        model_version=report.get("config_hash"),
        training_report=report,
    )
