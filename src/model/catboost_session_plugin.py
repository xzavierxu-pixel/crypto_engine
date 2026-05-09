from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.model.base import ModelPlugin


class CatBoostSessionPlugin(ModelPlugin):
    name = "catboost_session"

    def __init__(self, **params: Any) -> None:
        raw_params = dict(params)
        self.min_session_rows = int(raw_params.pop("min_session_rows", 500))
        raw_params.setdefault("verbose", False)
        self.params = {**raw_params, "min_session_rows": self.min_session_rows}
        self.model_params = raw_params
        self.global_model: CatBoostClassifier | None = None
        self.session_models: dict[str, CatBoostClassifier] = {}
        self.session_counts: dict[str, int] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        sample_weight_valid: pd.Series | None = None,
    ) -> "CatBoostSessionPlugin":
        self.global_model = CatBoostClassifier(**self.model_params)
        fit_kwargs: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = (X_valid, y_valid)
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self.global_model.fit(X_train, y_train, **fit_kwargs)

        self.session_models = {}
        self.session_counts = {}
        train_sessions = self._session_labels(X_train)
        valid_sessions = self._session_labels(X_valid) if X_valid is not None else None
        for session in ("asia", "europe", "us"):
            train_mask = train_sessions == session
            count = int(train_mask.sum())
            self.session_counts[session] = count
            if count < self.min_session_rows:
                continue
            model = CatBoostClassifier(**self.model_params)
            session_fit_kwargs: dict[str, Any] = {}
            if X_valid is not None and y_valid is not None and valid_sessions is not None:
                valid_mask = valid_sessions == session
                if bool(valid_mask.any()):
                    session_fit_kwargs["eval_set"] = (X_valid.loc[valid_mask], y_valid.loc[valid_mask])
            if sample_weight is not None:
                session_fit_kwargs["sample_weight"] = sample_weight.loc[train_mask]
            model.fit(X_train.loc[train_mask], y_train.loc[train_mask], **session_fit_kwargs)
            self.session_models[session] = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if self.global_model is None:
            raise ValueError("CatBoostSessionPlugin has not been fitted.")
        probabilities = pd.Series(self.global_model.predict_proba(X)[:, 1], index=X.index, name="p_up")
        sessions = self._session_labels(X)
        for session, model in self.session_models.items():
            mask = sessions == session
            if bool(mask.any()):
                probabilities.loc[mask] = model.predict_proba(X.loc[mask])[:, 1]
        return probabilities

    def get_feature_importance(self) -> np.ndarray:
        models = list(self.session_models.values())
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
                    "session_models": self.session_models,
                    "session_counts": self.session_counts,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostSessionPlugin":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        plugin = cls(**payload["params"])
        plugin.global_model = payload["global_model"]
        plugin.session_models = payload["session_models"]
        plugin.session_counts = payload["session_counts"]
        return plugin

    @staticmethod
    def _session_labels(X: pd.DataFrame | None) -> pd.Series:
        if X is None:
            return pd.Series(dtype="object")
        if "hour_sin" not in X.columns or "hour_cos" not in X.columns:
            return pd.Series("global", index=X.index)
        angle = np.arctan2(X["hour_sin"].to_numpy(), X["hour_cos"].to_numpy())
        hours = np.mod(angle, 2 * np.pi) * 24.0 / (2 * np.pi)
        labels = np.where(hours < 8.0, "asia", np.where(hours < 16.0, "europe", "us"))
        return pd.Series(labels, index=X.index)
