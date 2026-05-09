from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.model.base import ModelPlugin


class CatBoostRegimePlugin(ModelPlugin):
    name = "catboost_regime"

    def __init__(self, **params: Any) -> None:
        raw_params = dict(params)
        self.regime_feature = str(raw_params.pop("regime_feature", "rv_5"))
        self.min_regime_rows = int(raw_params.pop("min_regime_rows", 1000))
        raw_params.setdefault("verbose", False)
        self.params = {
            **raw_params,
            "regime_feature": self.regime_feature,
            "min_regime_rows": self.min_regime_rows,
        }
        self.model_params = raw_params
        self.global_model: CatBoostClassifier | None = None
        self.regime_models: dict[str, CatBoostClassifier] = {}
        self.regime_counts: dict[str, int] = {}
        self.quantiles: tuple[float, float] | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        sample_weight_valid: pd.Series | None = None,
    ) -> "CatBoostRegimePlugin":
        self.global_model = CatBoostClassifier(**self.model_params)
        fit_kwargs: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = (X_valid, y_valid)
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self.global_model.fit(X_train, y_train, **fit_kwargs)

        self.regime_models = {}
        self.regime_counts = {}
        train_regimes = self._fit_regime_labels(X_train)
        valid_regimes = self._regime_labels(X_valid) if X_valid is not None else None
        for regime in ("low", "mid", "high"):
            train_mask = train_regimes == regime
            count = int(train_mask.sum())
            self.regime_counts[regime] = count
            if count < self.min_regime_rows:
                continue
            model = CatBoostClassifier(**self.model_params)
            regime_fit_kwargs: dict[str, Any] = {}
            if X_valid is not None and y_valid is not None and valid_regimes is not None:
                valid_mask = valid_regimes == regime
                if bool(valid_mask.any()):
                    regime_fit_kwargs["eval_set"] = (X_valid.loc[valid_mask], y_valid.loc[valid_mask])
            if sample_weight is not None:
                regime_fit_kwargs["sample_weight"] = sample_weight.loc[train_mask]
            model.fit(X_train.loc[train_mask], y_train.loc[train_mask], **regime_fit_kwargs)
            self.regime_models[regime] = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if self.global_model is None:
            raise ValueError("CatBoostRegimePlugin has not been fitted.")
        probabilities = pd.Series(self.global_model.predict_proba(X)[:, 1], index=X.index, name="p_up")
        regimes = self._regime_labels(X)
        for regime, model in self.regime_models.items():
            mask = regimes == regime
            if bool(mask.any()):
                probabilities.loc[mask] = model.predict_proba(X.loc[mask])[:, 1]
        return probabilities

    def get_feature_importance(self) -> np.ndarray:
        models = list(self.regime_models.values())
        if self.global_model is not None:
            models.append(self.global_model)
        if not models:
            return np.array([], dtype="float64")
        return np.mean([model.get_feature_importance() for model in models], axis=0)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(
                {
                    "params": self.params,
                    "global_model": self.global_model,
                    "regime_models": self.regime_models,
                    "regime_counts": self.regime_counts,
                    "quantiles": self.quantiles,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostRegimePlugin":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        plugin = cls(**payload["params"])
        plugin.global_model = payload["global_model"]
        plugin.regime_models = payload["regime_models"]
        plugin.regime_counts = payload["regime_counts"]
        plugin.quantiles = payload["quantiles"]
        return plugin

    def _fit_regime_labels(self, X: pd.DataFrame) -> pd.Series:
        if self.regime_feature not in X.columns:
            self.quantiles = None
            return pd.Series("global", index=X.index)
        values = X[self.regime_feature].astype("float64").replace([np.inf, -np.inf], np.nan)
        q_low, q_high = values.quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy(dtype="float64")
        self.quantiles = (float(q_low), float(q_high))
        return self._regime_labels(X)

    def _regime_labels(self, X: pd.DataFrame | None) -> pd.Series:
        if X is None:
            return pd.Series(dtype="object")
        if self.regime_feature not in X.columns or self.quantiles is None:
            return pd.Series("global", index=X.index)
        q_low, q_high = self.quantiles
        values = X[self.regime_feature].astype("float64").replace([np.inf, -np.inf], np.nan)
        labels = np.where(values <= q_low, "low", np.where(values <= q_high, "mid", "high"))
        labels = pd.Series(labels, index=X.index)
        labels.loc[values.isna()] = "global"
        return labels
