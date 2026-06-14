from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_deploy_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return read_json(resolve_path(config["paths"]["deploy_artifact_dir"]) / "artifact_manifest.json")


def load_feature_columns(config: dict[str, Any]) -> list[str]:
    feature_manifest = config.get("paths", {}).get("feature_manifest")
    manifest = read_json(feature_manifest) if feature_manifest else load_deploy_manifest(config)
    return list(manifest["feature_columns"])


def p_side_bucket(p_side: pd.Series) -> pd.Series:
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 1.01]
    labels = ["0.50_0.55", "0.55_0.60", "0.60_0.65", "0.65_0.70", "0.70_1.00"]
    return pd.cut(p_side, bins=bins, labels=labels, include_lowest=True, right=False).astype("string").fillna("missing")


def time_to_lowest_bucket(seconds: pd.Series) -> pd.Series:
    bins = [-math.inf, 60, 120, 180, 240, math.inf]
    labels = ["000_060", "060_120", "120_180", "180_240", "missing_or_late"]
    return pd.cut(seconds, bins=bins, labels=labels, right=True).astype("string").fillna("missing")


def market_time_bucket(ts: pd.Series) -> pd.Series:
    hour = pd.to_datetime(ts, utc=True).dt.hour
    return pd.cut(
        hour,
        bins=[-1, 7, 15, 23],
        labels=["asia", "europe", "us"],
        include_lowest=True,
    ).astype("string")


def logit_price(price: pd.Series | np.ndarray, eps: float) -> np.ndarray:
    clipped = np.clip(np.asarray(price, dtype=float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def ensure_no_forbidden_features(feature_columns: list[str], forbidden: list[str]) -> None:
    forbidden_set = set(forbidden)
    overlap = sorted(forbidden_set.intersection(feature_columns))
    if overlap:
        raise ValueError(f"Forbidden feature columns present: {overlap}")


def signal_thresholds(config: dict[str, Any], manifest: dict[str, Any] | None = None) -> tuple[float, float]:
    signal = config.get("sample_filter", {}).get("signal", {})
    t_up = signal.get("t_up")
    t_down = signal.get("t_down")
    if t_up is not None and t_down is not None:
        return float(t_up), float(t_down)
    manifest = manifest if manifest is not None else load_deploy_manifest(config)
    return float(manifest["t_up"]), float(manifest["t_down"])


def apply_sample_filter(
    df: pd.DataFrame,
    config: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_filter = config.get("sample_filter", {})
    if not bool(sample_filter.get("enabled", False)):
        return df, {"enabled": False, "input_rows": int(len(df)), "output_rows": int(len(df))}

    mode = str(sample_filter.get("mode", "accepted_signal"))
    if mode != "accepted_signal":
        raise ValueError(f"Unsupported sample_filter.mode: {mode}")
    t_up, t_down = signal_thresholds(config, manifest)
    p_up = pd.to_numeric(df["p_up"], errors="coerce")
    mask = (p_up >= t_up) | (p_up <= t_down)
    out = df.loc[mask].copy()
    return out, {
        "enabled": True,
        "mode": mode,
        "t_up": t_up,
        "t_down": t_down,
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "retention": float(len(out) / len(df)) if len(df) else float("nan"),
        "up_signal_rows": int((p_up >= t_up).sum()),
        "down_signal_rows": int((p_up <= t_down).sum()),
    }
