from __future__ import annotations

from src.labels.base import LabelBuilder
from src.labels.grid_direction import GridDirectionLabelBuilder
from src.labels.polymarket_resolved import PolymarketResolvedLabelBuilder


LABEL_BUILDERS: dict[str, LabelBuilder] = {
    "grid_direction": GridDirectionLabelBuilder(),
    "polymarket_resolved": PolymarketResolvedLabelBuilder(),
}


def get_label_builder(name: str) -> LabelBuilder:
    try:
        return LABEL_BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown label builder '{name}'.") from exc
