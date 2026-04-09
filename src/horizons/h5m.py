from __future__ import annotations

from src.horizons.base import HorizonSpec


H5M_SPEC = HorizonSpec(
    name="5m",
    minutes=5,
    grid_minutes=5,
    label_builder="grid_direction",
    feature_profile="core_5m",
    signal_policy="default_edge_policy",
    sizing_plugin="fixed_fraction",
)
