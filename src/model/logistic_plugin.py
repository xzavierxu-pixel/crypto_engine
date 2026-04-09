from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.base import ModelPlugin


class LogisticRegressionPlugin(ModelPlugin):
    name = "logistic"

    def __init__(self, **params: Any) -> None:
        default_params = {
            "C": 1.0,
            "max_iter": 2000,
            "solver": "lbfgs",
            "random_state": 42,
        }
        default_params.update(params)
        self.params = default_params
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**default_params)),
            ]
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        sample_weight_valid: pd.Series | None = None,
    ) -> "LogisticRegressionPlugin":
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["classifier__sample_weight"] = sample_weight
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.model.predict_proba(X)[:, 1]
        return pd.Series(probabilities, index=X.index, name="p_up")

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump({"params": self.params, "model": self.model}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionPlugin":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        plugin = cls(**payload["params"])
        plugin.model = payload["model"]
        return plugin
