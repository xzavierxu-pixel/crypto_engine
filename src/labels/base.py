from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.core.config import Settings
from src.horizons.base import HorizonSpec


class LabelBuilder(ABC):
    name: str

    @abstractmethod
    def build(
        self,
        df: pd.DataFrame,
        settings: Settings,
        horizon: HorizonSpec,
        select_grid_only: bool | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError
