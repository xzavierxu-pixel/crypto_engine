from __future__ import annotations

from src.core.config import Settings
from src.horizons.base import HorizonSpec


def get_horizon_spec(settings: Settings, name: str | None = None) -> HorizonSpec:
    config = settings.horizons.get_active_spec(name)
    return HorizonSpec(
        name=name or settings.horizons.active[0],
        minutes=config.minutes,
        grid_minutes=config.grid_minutes,
        label_builder=config.label_builder,
        feature_profile=config.feature_profile,
        signal_policy=config.signal_policy,
        sizing_plugin=config.sizing_plugin,
        label_params=dict(config.label_params),
    )
