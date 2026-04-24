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
        stale_trade_flag = (nonzero_volume & returns.abs().le(1e-5)).astype(float)

        for window in windows:
            min_periods = window
            short_p20 = volume.rolling(window=window, min_periods=min_periods).quantile(0.2)
            long_reference_window = max(window * 3, window + 10)
            long_p20 = volume.rolling(
                window=long_reference_window,
                min_periods=long_reference_window,
            ).quantile(0.2)
            below_short_p20 = (volume <= short_p20).astype(float)
            below_long_p20 = (volume <= long_p20).astype(float)

            features[f"low_volume_flag_share_{window}"] = (
                below_long_p20.rolling(window=window, min_periods=min_periods).mean()
            )
            features[f"volume_below_rolling_p20_share_{window}"] = (
                below_short_p20.rolling(window=window, min_periods=min_periods).mean()
            )
            features[f"stale_trade_share_{window}"] = (
                stale_trade_flag.rolling(window=window, min_periods=min_periods).mean()
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
