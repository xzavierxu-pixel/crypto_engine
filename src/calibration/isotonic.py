from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.calibration.base import CalibrationPlugin


class IsotonicCalibration(CalibrationPlugin):
    name = "isotonic"

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_proba: pd.Series, y_true: pd.Series) -> "IsotonicCalibration":
        self.model.fit(raw_proba.to_numpy(), y_true.to_numpy())
        return self

    def transform(self, raw_proba: pd.Series) -> pd.Series:
        calibrated = self.model.predict(raw_proba.to_numpy())
        return pd.Series(calibrated, index=raw_proba.index, name=raw_proba.name).clip(0.0, 1.0)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self.model, handle)

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibration":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        plugin = cls()
        plugin.model = model
        return plugin
