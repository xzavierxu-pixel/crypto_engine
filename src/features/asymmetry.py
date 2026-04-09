from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class AsymmetryFeaturePack(FeaturePack):
    name = "asymmetry"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        past_open = df["open"].shift(1)
        past_high = df["high"].shift(1)
        past_low = df["low"].shift(1)
        past_close = df["close"].shift(1)
        returns = past_close.pct_change(fill_method=None)
        candle_range = (past_high - past_low).replace(0, np.nan)
        upper_wick = past_high - np.maximum(past_open, past_close)
        lower_wick = np.minimum(past_open, past_close) - past_low
        body = past_close - past_open

        for window in profile.asymmetry_rv_windows:
            upside_sq = returns.clip(lower=0.0).pow(2)
            downside_sq = (-returns.clip(upper=0.0)).pow(2)
            features[f"upside_rv_{window}"] = np.sqrt(
                upside_sq.rolling(window=window, min_periods=window).mean()
            )
            features[f"downside_rv_{window}"] = np.sqrt(
                downside_sq.rolling(window=window, min_periods=window).mean()
            )

        for window in profile.asymmetry_skew_windows:
            features[f"realized_skew_{window}"] = returns.rolling(
                window=window,
                min_periods=window,
            ).skew()

        wick_imbalance_1 = (lower_wick - upper_wick) / candle_range
        body_imbalance_1 = body / candle_range
        for window in profile.asymmetry_imbalance_windows:
            features[f"wick_imbalance_{window}"] = wick_imbalance_1.rolling(
                window=window,
                min_periods=window,
            ).mean()
            features[f"body_imbalance_{window}"] = body_imbalance_1.rolling(
                window=window,
                min_periods=window,
            ).mean()

        return features
