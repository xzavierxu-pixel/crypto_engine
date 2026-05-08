from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from src.model.base import ModelPlugin


class XGBoostClassifierPlugin(ModelPlugin):
    name = "xgboost"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.model = XGBClassifier(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        sample_weight_valid: pd.Series | None = None,
    ) -> "XGBoostClassifierPlugin":
        fit_kwargs: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            fit_kwargs["verbose"] = False
            if sample_weight_valid is not None:
                fit_kwargs["sample_weight_eval_set"] = [sample_weight_valid]
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.model.predict_proba(X)[:, 1]
        return pd.Series(probabilities, index=X.index, name="p_up")

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump({"params": self.params, "model": self.model}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostClassifierPlugin":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        plugin = cls(**payload["params"])
        plugin.model = payload["model"]
        return plugin
