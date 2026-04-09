from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class CandleStructureFeaturePack(FeaturePack):
    name = "candle_structure"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        prev_open = df["open"].shift(1)
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        prev_close = df["close"].shift(1)

        prev_range = (prev_high - prev_low).replace(0, np.nan)
        candle_body = prev_close - prev_open
        upper_wick = prev_high - np.maximum(prev_open, prev_close)
        lower_wick = np.minimum(prev_open, prev_close) - prev_low

        features["body_pct_1"] = candle_body / prev_open.replace(0, np.nan)
        features["true_range_pct_1"] = (prev_high - prev_low) / prev_close.replace(0, np.nan)
        features["upper_wick_ratio_1"] = upper_wick / prev_range
        features["lower_wick_ratio_1"] = lower_wick / prev_range
        features["close_location_1"] = (prev_close - prev_low) / prev_range

        for window in profile.range_windows:
            rolling_mean = prev_close.rolling(window=window, min_periods=window).mean()
            rolling_std = prev_close.rolling(window=window, min_periods=window).std()
            abs_diff_sum = prev_close.diff().abs().rolling(window=window, min_periods=window).sum()
            net_move = prev_close.diff(window).abs()
            features[f"close_z_{window}"] = (prev_close - rolling_mean) / rolling_std.replace(0, np.nan)
            features[f"efficiency_{window}"] = net_move / abs_diff_sum.replace(0, np.nan)

        return features
