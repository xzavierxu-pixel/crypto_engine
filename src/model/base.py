from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class ModelPlugin(ABC):
    name: str

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        sample_weight_valid: pd.Series | None = None,
    ) -> "ModelPlugin":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError(f"{self.__class__.__name__} does not support regression predictions.")

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "ModelPlugin":
        raise NotImplementedError
