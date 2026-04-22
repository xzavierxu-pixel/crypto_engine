from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from src.model.base import ModelPlugin


class LightGBMClassifierPlugin(ModelPlugin):
    name = "lightgbm"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.fit_params: dict[str, Any] = {}
        model_params = dict(params)

        early_stopping_rounds = model_params.pop("early_stopping_rounds", None)
        eval_metric = model_params.pop("eval_metric", None)
        if early_stopping_rounds is not None:
            self.fit_params["early_stopping_rounds"] = int(early_stopping_rounds)
        if eval_metric is not None:
            self.fit_params["eval_metric"] = eval_metric

        self.model = LGBMClassifier(**model_params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        sample_weight_valid: pd.Series | None = None,
    ) -> "LightGBMClassifierPlugin":
        fit_kwargs: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            if sample_weight_valid is not None:
                fit_kwargs["eval_sample_weight"] = [sample_weight_valid]
            if "eval_metric" in self.fit_params:
                fit_kwargs["eval_metric"] = self.fit_params["eval_metric"]
            if "early_stopping_rounds" in self.fit_params:
                fit_kwargs["callbacks"] = [
                    early_stopping(self.fit_params["early_stopping_rounds"], verbose=False)
                ]
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
    def load(cls, path: str | Path) -> "LightGBMClassifierPlugin":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        plugin = cls(**payload["params"])
        plugin.model = payload["model"]
        return plugin
