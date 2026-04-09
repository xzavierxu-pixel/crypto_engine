from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class FlowProxyFeaturePack(FeaturePack):
    name = "flow_proxy"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        if "volume" not in df.columns:
            return features

        past_open = df["open"].shift(1)
        past_high = df["high"].shift(1)
        past_low = df["low"].shift(1)
        past_close = df["close"].shift(1)
        volume = df["volume"].shift(1)
        candle_range = (past_high - past_low).replace(0, np.nan)
        candle_body = past_close - past_open
        upper_wick = past_high - np.maximum(past_open, past_close)
        lower_wick = np.minimum(past_open, past_close) - past_low
        clv = ((past_close - past_low) - (past_high - past_close)) / candle_range
        dollar_volume = past_close * volume
        ret_1 = past_close.pct_change(fill_method=None)
        range_pct = candle_range / past_close.replace(0, np.nan)
        rolling_volume = volume.rolling(
            window=profile.flow_volume_window,
            min_periods=profile.flow_volume_window,
        ).mean()

        features["clv_1"] = clv
        features["clv_x_volume_1"] = clv * volume
        features["clv_x_dollar_volume_1"] = clv * dollar_volume

        wick_pressure = (lower_wick - upper_wick) / candle_range
        features["wick_pressure_1"] = wick_pressure
        features["wick_pressure_x_volume_1"] = wick_pressure * volume

        signed_dollar_volume_1 = np.sign(ret_1.fillna(0.0)) * dollar_volume
        features["signed_dollar_volume_1"] = signed_dollar_volume_1
        features[f"signed_dollar_volume_{profile.flow_volume_window}"] = signed_dollar_volume_1.rolling(
            window=profile.flow_volume_window,
            min_periods=profile.flow_volume_window,
        ).sum()
        features[f"range_expansion_x_volume_{profile.flow_volume_window}"] = range_pct * rolling_volume
        features[f"body_x_volume_{profile.flow_volume_window}"] = (
            candle_body.abs() / past_open.replace(0, np.nan)
        ) * rolling_volume

        return features
