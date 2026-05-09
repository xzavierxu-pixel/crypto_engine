from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.base import CalibrationPlugin


class TemperatureScalingCalibration(CalibrationPlugin):
    name = "temperature"

    def __init__(self, temperature: float = 1.0, eps: float = 1e-6) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        self.temperature = float(temperature)
        self.eps = float(eps)

    def fit(self, raw_proba: pd.Series, y_true: pd.Series) -> "TemperatureScalingCalibration":
        return self

    def transform(self, raw_proba: pd.Series) -> pd.Series:
        clipped = raw_proba.astype("float64").clip(self.eps, 1.0 - self.eps)
        logits = np.log(clipped / (1.0 - clipped))
        calibrated = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        return pd.Series(calibrated, index=raw_proba.index, name=raw_proba.name).clip(0.0, 1.0)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump({"temperature": self.temperature, "eps": self.eps}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScalingCalibration":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls(**payload)
