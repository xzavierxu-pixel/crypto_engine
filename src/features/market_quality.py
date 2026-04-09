from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class MarketQualityFeaturePack(FeaturePack):
    name = "market_quality"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        if "volume" not in df.columns:
            return features

        windows = profile.market_quality_windows or [5, 20]
        volume = df["volume"].shift(1)
        past_close = df["close"].shift(1)
        returns = past_close.pct_change(fill_method=None)
        flat_candle = past_close.diff().abs().le(1e-12)
        nonzero_volume = volume.gt(0)
        dollar_volume = past_close * volume

        for window in windows:
            min_periods = window
            features[f"nz_volume_share_{window}"] = (
                nonzero_volume.astype(float).rolling(window=window, min_periods=min_periods).mean()
            )
            features[f"flat_share_{window}"] = (
                flat_candle.astype(float).rolling(window=window, min_periods=min_periods).mean()
            )
            features[f"abs_ret_mean_{window}"] = (
                returns.abs().rolling(window=window, min_periods=min_periods).mean()
            )
            features[f"dollar_vol_mean_{window}"] = (
                dollar_volume.rolling(window=window, min_periods=min_periods).mean()
            )

        return features
