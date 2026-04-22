from __future__ import annotations

import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class Intra5mStructureFeaturePack(FeaturePack):
    name = "intra_5m_structure"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        past_close = df["close"].shift(1)
        past_open = df["open"].shift(1)
        candle_up = (past_close > past_open).astype(float)
        candle_down = (past_close < past_open).astype(float)
        one_bar_ret = past_close.pct_change(1, fill_method=None)

        features["last_1m_up"] = candle_up
        features["up_ratio_5"] = candle_up.rolling(5).mean()
        features["down_ratio_5"] = candle_down.rolling(5).mean()
        features["max_up_ret_5"] = one_bar_ret.rolling(5).max()
        features["max_down_ret_5"] = one_bar_ret.rolling(5).min()
        return features
