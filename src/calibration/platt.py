from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.calibration.base import CalibrationPlugin


class PlattScalingCalibration(CalibrationPlugin):
    name = "platt"

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | dict[int, float] | None = None,
    ) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, class_weight=self.class_weight)

    def fit(self, raw_proba: pd.Series, y_true: pd.Series) -> "PlattScalingCalibration":
        X = raw_proba.to_numpy().reshape(-1, 1)
        self.model.fit(X, y_true.to_numpy())
        return self

    def transform(self, raw_proba: pd.Series) -> pd.Series:
        X = raw_proba.to_numpy().reshape(-1, 1)
        calibrated = self.model.predict_proba(X)[:, 1]
        return pd.Series(calibrated, index=raw_proba.index, name=raw_proba.name).clip(0.0, 1.0)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self.model, handle)

    @classmethod
    def load(cls, path: str | Path) -> "PlattScalingCalibration":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        plugin = cls()
        plugin.model = model
        return plugin
