from __future__ import annotations

import pandas as pd

from src.calibration.base import CalibrationPlugin
from src.data.dataset_builder import infer_feature_columns
from src.model.base import ModelPlugin


def predict_frame(
    frame: pd.DataFrame,
    model: ModelPlugin,
    calibrator: CalibrationPlugin | None = None,
    feature_columns: list[str] | None = None,
    target_semantics: dict | None = None,
) -> pd.Series:
    resolved_columns = feature_columns or infer_feature_columns(frame)
    raw = model.predict_proba(frame[resolved_columns])
    probabilities = raw if calibrator is None else calibrator.transform(raw)
    semantics = target_semantics or {}
    if semantics.get("training_target") != "first_minute_follow":
        return probabilities
    if "fm_ret" in frame.columns:
        first_minute_return = pd.to_numeric(frame["fm_ret"], errors="coerce")
    elif "ret_1" in frame.columns:
        first_minute_return = pd.to_numeric(frame["ret_1"], errors="coerce")
    else:
        raise ValueError("first_minute_follow inference requires fm_ret or ret_1 in the feature frame.")
    first_minute_up = first_minute_return >= 0.0
    return probabilities.where(first_minute_up, 1.0 - probabilities).astype("float64").clip(0.0, 1.0)


def predict_frame_multiclass(
    frame: pd.DataFrame,
    model: ModelPlugin,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    resolved_columns = feature_columns or infer_feature_columns(frame)
    return model.predict_proba_multiclass(frame[resolved_columns])
