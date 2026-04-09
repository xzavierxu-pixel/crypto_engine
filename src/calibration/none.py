from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.calibration.base import CalibrationPlugin


class NoCalibration(CalibrationPlugin):
    name = "none"

    def fit(self, raw_proba: pd.Series, y_true: pd.Series) -> "NoCalibration":
        return self

    def transform(self, raw_proba: pd.Series) -> pd.Series:
        return raw_proba.clip(0.0, 1.0)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump({"name": self.name}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "NoCalibration":
        with Path(path).open("rb") as handle:
            pickle.load(handle)
        return cls()
