from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

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

        self.objective = str(model_params.get("objective", "binary"))
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
            fit_kwargs["eval_set"] = [(X_train, y_train), (X_valid, y_valid)]
            fit_kwargs["eval_names"] = ["train", "validation"]
            if sample_weight_valid is not None:
                eval_sample_weight = [sample_weight if sample_weight is not None else None, sample_weight_valid]
                fit_kwargs["eval_sample_weight"] = eval_sample_weight
            elif sample_weight is not None:
                fit_kwargs["eval_sample_weight"] = [sample_weight, None]
            if "eval_metric" in self.fit_params:
                fit_kwargs["eval_metric"] = self.fit_params["eval_metric"]
            callbacks = [log_evaluation(period=10)]
            if "early_stopping_rounds" in self.fit_params:
                callbacks.append(
                    early_stopping(self.fit_params["early_stopping_rounds"], verbose=False)
                )
            fit_kwargs["callbacks"] = callbacks
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.model.predict_proba(X)[:, 1]
        return pd.Series(probabilities, index=X.index, name="p_up")

    def predict_proba_multiclass(self, X: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.model.predict_proba(X)
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError("Expected multiclass probabilities with three columns.")
        return pd.DataFrame(
            probabilities,
            index=X.index,
            columns=["p_down", "p_flat", "p_up"],
        )

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
