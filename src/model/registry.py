from __future__ import annotations

from src.core.config import Settings
from src.model.base import ModelPlugin
from src.model.catboost_plugin import CatBoostClassifierPlugin
from src.model.lightgbm_plugin import LightGBMClassifierPlugin


MODEL_PLUGINS: dict[str, type[ModelPlugin]] = {
    "lightgbm": LightGBMClassifierPlugin,
    "catboost": CatBoostClassifierPlugin,
}


def create_model_plugin(settings: Settings, plugin_name: str | None = None) -> ModelPlugin:
    target = plugin_name or settings.model.active_plugin
    try:
        plugin_cls = MODEL_PLUGINS[target]
    except KeyError as exc:
        raise KeyError(f"Unknown model plugin '{target}'.") from exc
    plugin_params = settings.model.plugins.get(target, {})
    return plugin_cls(**plugin_params)


def load_model_plugin(plugin_name: str, path: str) -> ModelPlugin:
    try:
        plugin_cls = MODEL_PLUGINS[plugin_name]
    except KeyError as exc:
        raise KeyError(f"Unknown model plugin '{plugin_name}'.") from exc
    return plugin_cls.load(path)
