from __future__ import annotations

import pandas as pd

from src.model.lightgbm_plugin import LightGBMClassifierPlugin


def test_lightgbm_plugin_routes_eval_metric_and_early_stopping_to_fit(monkeypatch) -> None:
    captured: dict[str, object] = {}
    callback_names: list[str] = []

    def fake_early_stopping(rounds, verbose=False):
        callback_names.append("early_stopping")
        return ("early_stopping", rounds, verbose)

    def fake_log_evaluation(period=1):
        callback_names.append("log_evaluation")
        return ("log_evaluation", period)

    class FakeLGBMClassifier:
        def __init__(self, **params) -> None:
            captured["init_params"] = params

        def fit(self, X_train, y_train, **kwargs) -> None:
            captured["fit_kwargs"] = kwargs

        def predict_proba(self, X):
            return [[0.4, 0.6] for _ in range(len(X))]

    monkeypatch.setattr("src.model.lightgbm_plugin.LGBMClassifier", FakeLGBMClassifier)
    monkeypatch.setattr("src.model.lightgbm_plugin.early_stopping", fake_early_stopping)
    monkeypatch.setattr("src.model.lightgbm_plugin.log_evaluation", fake_log_evaluation)

    plugin = LightGBMClassifierPlugin(
        n_estimators=200,
        learning_rate=0.03,
        early_stopping_rounds=50,
        eval_metric="binary_logloss",
    )
    X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0]})
    y_train = pd.Series([0, 1, 0])
    X_valid = pd.DataFrame({"f1": [3.0, 4.0]})
    y_valid = pd.Series([1, 0])
    sample_weight = pd.Series([1.0, 1.5, 1.0])
    sample_weight_valid = pd.Series([1.0, 2.0])

    plugin.fit(
        X_train,
        y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        sample_weight=sample_weight,
        sample_weight_valid=sample_weight_valid,
    )

    assert captured["init_params"] == {
        "n_estimators": 200,
        "learning_rate": 0.03,
    }
    fit_kwargs = captured["fit_kwargs"]
    assert fit_kwargs["eval_set"] == [(X_train, y_train), (X_valid, y_valid)]
    assert fit_kwargs["eval_names"] == ["train", "validation"]
    assert fit_kwargs["eval_sample_weight"] == [sample_weight, sample_weight_valid]
    assert fit_kwargs["eval_metric"] == "binary_logloss"
    assert fit_kwargs["sample_weight"] is sample_weight
    assert fit_kwargs["callbacks"]
    assert callback_names == ["log_evaluation", "early_stopping"]
