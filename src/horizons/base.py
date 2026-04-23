from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    minutes: int
    grid_minutes: int
    label_builder: str
    feature_profile: str
    signal_policy: str | None = None
    sizing_plugin: str | None = None
    label_params: dict[str, Any] | None = None

    @property
    def future_close_offset(self) -> int:
        return self.minutes - 1
