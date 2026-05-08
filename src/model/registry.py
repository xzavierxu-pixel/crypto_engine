from __future__ import annotations

from src.core.config import Settings
from src.model.base import ModelPlugin
from src.model.catboost_ensemble_plugin import CatBoostSeedEnsemblePlugin
from src.model.catboost_plugin import CatBoostClassifierPlugin
from src.model.lightgbm_plugin import LightGBMClassifierPlugin
from src.model.logistic_plugin import LogisticRegressionPlugin


MODEL_PLUGINS: dict[str, type[ModelPlugin]] = {
    "lightgbm": LightGBMClassifierPlugin,
    "lightgbm_stage1": LightGBMClassifierPlugin,
    "lightgbm_stage2": LightGBMClassifierPlugin,
    "catboost": CatBoostClassifierPlugin,
    "catboost_ensemble": CatBoostSeedEnsemblePlugin,
    "logistic": LogisticRegressionPlugin,
}


def create_model_plugin(
    settings: Settings,
    plugin_name: str | None = None,
    plugin_params: dict | None = None,
    stage: str | None = None,
) -> ModelPlugin:
    target = plugin_name or settings.model.resolve_plugin(stage=stage)
    try:
        plugin_cls = MODEL_PLUGINS[target]
    except KeyError as exc:
        raise KeyError(f"Unknown model plugin '{target}'.") from exc
    resolved_params = dict(settings.model.plugins.get(target, {}))
    if plugin_params:
        resolved_params.update(plugin_params)
    return plugin_cls(**resolved_params)


def load_model_plugin(plugin_name: str, path: str) -> ModelPlugin:
    try:
        plugin_cls = MODEL_PLUGINS[plugin_name]
    except KeyError as exc:
        raise KeyError(f"Unknown model plugin '{plugin_name}'.") from exc
    return plugin_cls.load(path)
