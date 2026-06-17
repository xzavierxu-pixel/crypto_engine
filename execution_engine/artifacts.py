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


class SafeGapPreprocessor:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.numeric_columns = [str(column) for column in payload["numeric_columns"]]
        self.categorical_columns = [str(column) for column in payload["categorical_columns"]]
        self.medians = {str(key): float(value) for key, value in payload["medians"].items()}
        self.means = {str(key): float(value) for key, value in payload["means"].items()}
        self.scales = {str(key): float(value) for key, value in payload["scales"].items()}
        self.categories = {
            str(key): [str(item) for item in value]
            for key, value in payload["categories"].items()
        }

    def transform(self, frame: Any) -> Any:
        import numpy as np

        blocks: list[Any] = []
        for column in self.numeric_columns:
            values = frame[column].astype(float).fillna(self.medians[column]).to_numpy()
            blocks.append(((values - self.means[column]) / self.scales[column]).reshape(-1, 1))
        for column in self.categorical_columns:
            values = frame[column].astype("string").fillna("missing").astype(str)
            categories = self.categories[column]
            mapping = {category: idx for idx, category in enumerate(categories)}
            arr = np.zeros((len(frame), len(categories)), dtype=np.float32)
            missing_idx = mapping.get("missing", 0)
            idx = values.map(mapping).fillna(missing_idx).astype(int).to_numpy()
            arr[np.arange(len(frame)), idx] = 1.0
            blocks.append(arr)
        return np.hstack(blocks).astype(np.float32)


class SafeLowestPriceGapNumpyModel:
    def __init__(self, payload: dict[str, Any], arrays: dict[str, Any]) -> None:
        self.payload = payload
        self.arrays = arrays
        self.preprocessor = SafeGapPreprocessor(dict(payload["preprocessor"]))
        calibration = payload["calibration"]
        self.delta = float(calibration["delta"])
        self.bucket_miss_threshold = float(calibration["bucket_miss_threshold"])
        self.bucket_model = calibration["bucket_model"]
        target = payload["target"]
        self.tick_size = float(target["tick_size"])
        self.tick_rounding_tolerance = float(target["tick_rounding_tolerance"])

    def predict(self, frame: Any) -> Any:
        import numpy as np
        import pandas as pd

        x = self.preprocessor.transform(frame)
        logits = self._forward(x).reshape(-1)
        f_model = self._sigmoid(logits)
        if "p_side" not in frame.columns:
            raise ValueError("Safe lowest price gap estimator requires p_side in the runtime feature frame.")
        p_side = frame["p_side"].astype(float).to_numpy()
        conf_ok = self._bucket_conf_ok(frame)
        p_pred, action = self._infer_prices(
            f_model,
            p_side,
            conf_ok,
        )

        return pd.DataFrame(
            {
                "p_pred": p_pred,
                "safe_gap_action": action,
                "safe_gap_conf_ok": conf_ok,
                "safe_gap_f_model": f_model,
            },
            index=frame.index,
        )

    def _forward(self, x: Any) -> Any:
        import numpy as np

        out = x
        out = self._linear(out, "net__0")
        out = self._silu(out)
        out = self._layer_norm(out, "net__2")
        out = self._linear(out, "net__4")
        out = self._silu(out)
        out = self._layer_norm(out, "net__6")
        out = self._linear(out, "net__8")
        out = self._silu(out)
        return self._linear(out, "net__10")

    def _sigmoid(self, x: Any) -> Any:
        import numpy as np

        return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))

    def _silu(self, x: Any) -> Any:
        return x * self._sigmoid(x)

    def _linear(self, x: Any, prefix: str) -> Any:
        return x @ self.arrays[f"{prefix}__weight"].T + self.arrays[f"{prefix}__bias"]

    def _layer_norm(self, x: Any, prefix: str) -> Any:
        import numpy as np

        mean = x.mean(axis=1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + 1e-5)
        return normalized * self.arrays[f"{prefix}__weight"] + self.arrays[f"{prefix}__bias"]

    def _bucket_conf_ok(self, frame: Any) -> Any:
        import numpy as np
        import pandas as pd

        keys = [str(key) for key in self.bucket_model.get("keys", [])]
        if not keys:
            labels = pd.Series(["global"] * len(frame), index=frame.index)
        else:
            bucket_frame = pd.DataFrame(index=frame.index)
            for key in keys:
                if key in frame.columns:
                    bucket_frame[key] = frame[key].astype("string").fillna("missing").astype(str)
                else:
                    bucket_frame[key] = "missing"
            labels = bucket_frame.astype(str).agg("|".join, axis=1)
        table = self.bucket_model.get("table", {})
        global_miss_rate = float(self.bucket_model.get("global_miss_rate", 1.0))
        miss_rate = labels.map(lambda value: float(table.get(str(value), {}).get("miss_rate", global_miss_rate)))
        return miss_rate.to_numpy(dtype=float) <= self.bucket_miss_threshold

    def _infer_prices(self, f_model: Any, p_side: Any, conf_ok: Any) -> tuple[Any, Any]:
        import numpy as np

        raw = f_model + self.delta
        ticked = np.ceil(raw / self.tick_size - self.tick_rounding_tolerance) * self.tick_size
        ticked = np.maximum(ticked, 0.0)
        p_pred = np.minimum(ticked, p_side)
        action = np.full(len(f_model), "active", dtype=object)
        clamp = conf_ok & (raw >= p_side)
        action[clamp] = "clamp_over_pside"
        action[~conf_ok] = "abstain_low_conf"
        p_pred[~conf_ok] = p_side[~conf_ok]
        p_pred[clamp] = p_side[clamp]
        return p_pred, action.astype(str)


def _load_safe_gap_numpy_model(path: Path) -> SafeLowestPriceGapNumpyModel:
    import json
    import numpy as np

    loaded = np.load(path, allow_pickle=False)
    payload = json.loads(str(loaded["metadata"]))
    arrays = {key: loaded[key] for key in loaded.files if key != "metadata"}
    return SafeLowestPriceGapNumpyModel(payload, arrays)


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
    if model_format == "safe_lowest_price_gap_numpy":
        model = _load_safe_gap_numpy_model(model_path)
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

