from __future__ import annotations

from src.labels.base import LabelBuilder
from src.labels.grid_direction import GridDirectionLabelBuilder


LABEL_BUILDERS: dict[str, LabelBuilder] = {
    "grid_direction": GridDirectionLabelBuilder(),
}


def get_label_builder(name: str) -> LabelBuilder:
    try:
        return LABEL_BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown label builder '{name}'.") from exc
