from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.model.base import ModelPlugin


class CatBoostSeedEnsemblePlugin(ModelPlugin):
    name = "catboost_ensemble"

    def __init__(self, **params: Any) -> None:
        raw_params = dict(params)
        seeds = raw_params.pop("seeds", None)
        n_seeds = int(raw_params.pop("n_seeds", 3))
        base_seed = int(raw_params.get("random_seed", 42))
        if seeds is None:
            seeds = [base_seed + offset for offset in range(n_seeds)]
        self.seeds = [int(seed) for seed in seeds]
        raw_params.setdefault("verbose", False)
        self.params = {**raw_params, "seeds": self.seeds}
        self.model_params = raw_params
        self.models: list[CatBoostClassifier] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        sample_weight_valid: pd.Series | None = None,
    ) -> "CatBoostSeedEnsemblePlugin":
        self.models = []
        for seed in self.seeds:
            params = dict(self.model_params)
            params["random_seed"] = seed
            model = CatBoostClassifier(**params)
            fit_kwargs: dict[str, Any] = {}
            if X_valid is not None and y_valid is not None:
                fit_kwargs["eval_set"] = (X_valid, y_valid)
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            model.fit(X_train, y_train, **fit_kwargs)
            self.models.append(model)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if not self.models:
            raise ValueError("CatBoostSeedEnsemblePlugin has not been fitted.")
        probabilities = np.mean([model.predict_proba(X)[:, 1] for model in self.models], axis=0)
        return pd.Series(probabilities, index=X.index, name="p_up")

    def get_feature_importance(self) -> np.ndarray:
        if not self.models:
            return np.array([], dtype="float64")
        return np.mean([model.get_feature_importance() for model in self.models], axis=0)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump({"params": self.params, "models": self.models}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostSeedEnsemblePlugin":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        plugin = cls(**payload["params"])
        plugin.models = payload["models"]
        return plugin
