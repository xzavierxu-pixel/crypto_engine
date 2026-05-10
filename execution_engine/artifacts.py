from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from execution_engine.config import BaselineConfig


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
    )

