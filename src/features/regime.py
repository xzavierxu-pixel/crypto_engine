from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class RegimeFeaturePack(FeaturePack):
    name = "regime"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        if not profile.use_regime_features:
            return features

        past_close = df["close"].shift(1)
        ret_1 = past_close.pct_change(fill_method=None)

        short_window = min(profile.vol_windows) if profile.vol_windows else 3
        long_window = max(profile.vol_windows) if profile.vol_windows else max(short_window * 2, 10)
        short_vol = ret_1.rolling(window=short_window, min_periods=short_window).std()
        long_vol = ret_1.rolling(window=long_window, min_periods=long_window).std()
        short_return = past_close.pct_change(short_window, fill_method=None)

        features["regime_vol_ratio"] = short_vol / long_vol.replace(0, np.nan)
        features["regime_trend_strength"] = short_return.abs() / short_vol.replace(0, np.nan)
        return features
