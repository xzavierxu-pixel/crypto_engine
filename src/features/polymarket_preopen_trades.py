from __future__ import annotations

import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.data.polymarket_trades import PREOPEN_TRADE_FEATURE_PREFIX
from src.features.base import FeaturePack


class PolymarketPreopenTradesFeaturePack(FeaturePack):
    name = "polymarket_preopen_trades"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        columns = [column for column in df.columns if column.startswith(PREOPEN_TRADE_FEATURE_PREFIX)]
        if not columns:
            return pd.DataFrame(index=df.index)
        features = df[columns].copy()
        return features.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
