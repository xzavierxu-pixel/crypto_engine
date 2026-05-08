from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class HTFContextFeaturePack(FeaturePack):
    name = "htf_context"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        past_close = df["close"].shift(1)
        past_high = df["high"].shift(1)
        past_low = df["low"].shift(1)

        for timeframe_minutes in profile.htf_context_timeframes:
            timeframe_label = f"{timeframe_minutes}m"

            returns = past_close.pct_change(timeframe_minutes, fill_method=None)
            rv_window = timeframe_minutes * profile.htf_context_vol_window
            rv = past_close.pct_change(fill_method=None).rolling(
                window=rv_window,
                min_periods=timeframe_minutes,
            ).std()
            range_window = timeframe_minutes * profile.htf_context_range_window
            rolling_high = past_high.rolling(
                window=range_window,
                min_periods=timeframe_minutes,
            ).max()
            rolling_low = past_low.rolling(
                window=range_window,
                min_periods=timeframe_minutes,
            ).min()
            rolling_range = (rolling_high - rolling_low).replace(0, np.nan)
            zscore_window = timeframe_minutes * profile.htf_context_zscore_window
            rolling_mean = past_close.rolling(
                window=zscore_window,
                min_periods=timeframe_minutes,
            ).mean()
            rolling_std = past_close.rolling(
                window=zscore_window,
                min_periods=timeframe_minutes,
            ).std()
            efficiency_window = timeframe_minutes * profile.htf_context_efficiency_window
            net_move = past_close.diff(timeframe_minutes).abs()
            abs_diff_sum = past_close.diff().abs().rolling(
                window=efficiency_window,
                min_periods=timeframe_minutes,
            ).sum()
            trend_return = past_close.pct_change(
                timeframe_minutes,
                fill_method=None,
            )

            features[f"htf_ret_{timeframe_label}_1"] = returns
            features[f"htf_rv_{timeframe_label}"] = rv
            features[f"htf_range_pos_{timeframe_label}"] = (past_close - rolling_low) / rolling_range
            features[f"htf_close_z_{timeframe_label}"] = (past_close - rolling_mean) / rolling_std.replace(0, np.nan)
            features[f"htf_efficiency_{timeframe_label}"] = net_move / abs_diff_sum.replace(0, np.nan)

            if timeframe_minutes >= 15:
                features[f"htf_regime_trend_strength_{timeframe_label}"] = trend_return.abs() / rv.replace(0, np.nan)

        return features
