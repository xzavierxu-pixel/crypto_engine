from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.core.config import FeatureProfileConfig, Settings


class FeaturePack(ABC):
    name: str

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        raise NotImplementedError
