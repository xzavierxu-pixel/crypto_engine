from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.calibration.base import CalibrationPlugin


class PlattLogitCalibration(CalibrationPlugin):
    name = "platt_logit"

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | dict[int, float] | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.eps = float(eps)
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, class_weight=self.class_weight)

    def _to_logit_frame(self, raw_proba: pd.Series) -> np.ndarray:
        clipped = raw_proba.astype("float64").clip(self.eps, 1.0 - self.eps)
        logits = np.log(clipped / (1.0 - clipped))
        return logits.to_numpy().reshape(-1, 1)

    def fit(self, raw_proba: pd.Series, y_true: pd.Series) -> "PlattLogitCalibration":
        self.model.fit(self._to_logit_frame(raw_proba), y_true.to_numpy())
        return self

    def transform(self, raw_proba: pd.Series) -> pd.Series:
        calibrated = self.model.predict_proba(self._to_logit_frame(raw_proba))[:, 1]
        return pd.Series(calibrated, index=raw_proba.index, name=raw_proba.name).clip(0.0, 1.0)

    def save(self, path: str | Path) -> None:
        payload = {"model": self.model, "eps": self.eps}
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "PlattLogitCalibration":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        plugin = cls(eps=payload.get("eps", 1e-6))
        plugin.model = payload["model"]
        return plugin
