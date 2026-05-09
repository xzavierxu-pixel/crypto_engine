from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from src.model.base import ModelPlugin


class CatBoostLightGBMLogitBlendPlugin(ModelPlugin):
    name = "catboost_lgbm_logit_blend"

    def __init__(self, **params: Any) -> None:
        raw_params = dict(params)
        self.catboost_weight = float(raw_params.pop("catboost_weight", 0.85))
        self.catboost_params = dict(raw_params.pop("catboost", {}))
        self.lightgbm_params = dict(raw_params.pop("lightgbm", {}))
        self.catboost_params.setdefault("verbose", False)
        self.lightgbm_params.setdefault("verbosity", -1)
        self.params = {
            "catboost_weight": self.catboost_weight,
            "catboost": self.catboost_params,
            "lightgbm": self.lightgbm_params,
        }
        self.catboost_model = CatBoostClassifier(**self.catboost_params)
        self.lightgbm_model = LGBMClassifier(**self.lightgbm_params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        sample_weight_valid: pd.Series | None = None,
    ) -> "CatBoostLightGBMLogitBlendPlugin":
        catboost_fit_kwargs: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            catboost_fit_kwargs["eval_set"] = (X_valid, y_valid)
        if sample_weight is not None:
            catboost_fit_kwargs["sample_weight"] = sample_weight
        self.catboost_model.fit(X_train, y_train, **catboost_fit_kwargs)

        lightgbm_fit_kwargs: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            lightgbm_fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            lightgbm_fit_kwargs["eval_names"] = ["validation"]
            if sample_weight_valid is not None:
                lightgbm_fit_kwargs["eval_sample_weight"] = [sample_weight_valid]
        if sample_weight is not None:
            lightgbm_fit_kwargs["sample_weight"] = sample_weight
        self.lightgbm_model.fit(X_train, y_train, **lightgbm_fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        catboost_proba = self.catboost_model.predict_proba(X)[:, 1]
        lightgbm_proba = self.lightgbm_model.predict_proba(X)[:, 1]
        catboost_logit = self._logit(catboost_proba)
        lightgbm_logit = self._logit(lightgbm_proba)
        blended_logit = (
            self.catboost_weight * catboost_logit
            + (1.0 - self.catboost_weight) * lightgbm_logit
        )
        probabilities = 1.0 / (1.0 + np.exp(-blended_logit))
        return pd.Series(probabilities, index=X.index, name="p_up")

    def get_feature_importance(self) -> np.ndarray:
        catboost_importance = self.catboost_model.get_feature_importance()
        lightgbm_importance = self.lightgbm_model.booster_.feature_importance(importance_type="gain")
        if len(catboost_importance) != len(lightgbm_importance):
            return np.array([], dtype="float64")
        return (
            self.catboost_weight * catboost_importance
            + (1.0 - self.catboost_weight) * lightgbm_importance
        )

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(
                {
                    "params": self.params,
                    "catboost_model": self.catboost_model,
                    "lightgbm_model": self.lightgbm_model,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostLightGBMLogitBlendPlugin":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        plugin = cls(**payload["params"])
        plugin.catboost_model = payload["catboost_model"]
        plugin.lightgbm_model = payload["lightgbm_model"]
        return plugin

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        return np.log(clipped / (1.0 - clipped))
