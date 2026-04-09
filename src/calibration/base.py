from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class CalibrationPlugin(ABC):
    name: str

    @abstractmethod
    def fit(self, raw_proba: pd.Series, y_true: pd.Series) -> "CalibrationPlugin":
        raise NotImplementedError

    @abstractmethod
    def transform(self, raw_proba: pd.Series) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "CalibrationPlugin":
        raise NotImplementedError
