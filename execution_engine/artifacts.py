from __future__ import annotations

import json
import pickle
import sys
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


class SafeLowestPriceGapTorchModel:
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        import numpy as np
        import torch

        repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "price_estimator").is_dir())
        safe_gap_dir = repo_root / "price_estimator" / "safe_lowest_price_gap"
        upper_bound_dir = repo_root / "price_estimator" / "upper_bound_mlp"
        scripts_dir = repo_root / "price_estimator" / "scripts"
        for path in (str(safe_gap_dir), str(upper_bound_dir), str(scripts_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from train_safe_lowest_price_gap import bucket_conf_ok, infer_prices  # type: ignore
        from train_upper_bound_mlp import Preprocessor, UpperBoundMLP  # type: ignore

        self._np = np
        self._torch = torch
        self._bucket_conf_ok = bucket_conf_ok
        self._infer_prices = infer_prices
        preprocessor_payload = dict(checkpoint["preprocessor"])
        self.preprocessor = Preprocessor(**preprocessor_payload)
        model_config = checkpoint["model"]
        self.model = UpperBoundMLP(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims=[int(value) for value in model_config["hidden_dims"]],
            dropout=[float(value) for value in model_config["dropout"]],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        calibration = checkpoint["calibration"]
        self.delta = float(calibration["delta"])
        self.bucket_miss_threshold = float(calibration["bucket_miss_threshold"])
        self.bucket_model = calibration["bucket_model"]
        target = checkpoint["target"]
        self.tick_size = float(target["tick_size"])
        self.tick_rounding_tolerance = float(target["tick_rounding_tolerance"])

    def predict(self, frame: Any) -> Any:
        with self._torch.no_grad():
            x = self.preprocessor.transform(frame)
            logits = self.model(self._torch.from_numpy(x.astype(self._np.float32))).detach().cpu().numpy().reshape(-1)
        f_model = 1.0 / (1.0 + self._np.exp(-logits))
        if "p_side" not in frame.columns:
            raise ValueError("Safe lowest price gap estimator requires p_side in the runtime feature frame.")
        p_side = frame["p_side"].astype(float).to_numpy()
        conf_ok = self._bucket_conf_ok(frame, self.bucket_model, self.bucket_miss_threshold)
        prediction = self._infer_prices(
            f_model,
            p_side,
            conf_ok,
            self.delta,
            self.tick_size,
            self.tick_rounding_tolerance,
        )
        import pandas as pd

        return pd.DataFrame(
            {
                "p_pred": prediction.p_pred,
                "safe_gap_action": prediction.action,
                "safe_gap_conf_ok": prediction.conf_ok,
                "safe_gap_f_model": f_model,
            },
            index=frame.index,
        )


def _load_safe_gap_torch_model(path: Path) -> SafeLowestPriceGapTorchModel:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return SafeLowestPriceGapTorchModel(checkpoint)


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

    model_format = str(manifest.get("model_format", ""))
    if model_format == "safe_lowest_price_gap_torch":
        model = _load_safe_gap_torch_model(model_path)
    else:
        model = _load_pickle_model(model_path)

    return PriceEstimatorArtifact(
        artifact_dir=artifact_dir,
        manifest=manifest,
        model_path=model_path,
        model=model,
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

