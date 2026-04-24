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
) -> pd.Series:
    resolved_columns = feature_columns or infer_feature_columns(frame)
    raw = model.predict_proba(frame[resolved_columns])
    if calibrator is None:
        return raw
    return calibrator.transform(raw)


def predict_frame_multiclass(
    frame: pd.DataFrame,
    model: ModelPlugin,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    resolved_columns = feature_columns or infer_feature_columns(frame)
    return model.predict_proba_multiclass(frame[resolved_columns])
