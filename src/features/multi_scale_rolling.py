from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class MultiScaleRollingFeaturePack(FeaturePack):
    name = "multi_scale_rolling"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        windows = profile.multi_scale_windows or [5, 15, 30, 60]

        past_close = df["close"].shift(1)
        past_high = df["high"].shift(1)
        past_low = df["low"].shift(1)
        base_return = past_close.pct_change(fill_method=None)
        abs_return = base_return.abs()
        return_variation = base_return.diff().abs()
        positive_return = (base_return > 0).astype("float64")

        for window in windows:
            rolling_high = past_high.rolling(window=window, min_periods=window).max()
            rolling_low = past_low.rolling(window=window, min_periods=window).min()
            rolling_range = rolling_high - rolling_low
            close_denom = past_close.replace(0, np.nan)
            range_denom = rolling_range.replace(0, np.nan)

            features[f"ms_ret_mean_{window}"] = base_return.rolling(window=window, min_periods=window).mean()
            features[f"ms_abs_ret_mean_{window}"] = abs_return.rolling(window=window, min_periods=window).mean()
            features[f"ms_ret_variation_{window}"] = return_variation.rolling(window=window, min_periods=window).mean()
            features[f"ms_positive_return_share_{window}"] = positive_return.rolling(
                window=window,
                min_periods=window,
            ).mean()
            features[f"ms_range_pct_{window}"] = rolling_range / close_denom
            features[f"ms_close_pos_{window}"] = (past_close - rolling_low) / range_denom

        return features
