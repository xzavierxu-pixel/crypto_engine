from __future__ import annotations

import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class VolatilityFeaturePack(FeaturePack):
    name = "volatility"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        base_return = df["close"].shift(1).pct_change(fill_method=None)
        for window in profile.vol_windows:
            features[f"rv_{window}"] = base_return.rolling(window=window, min_periods=window).std()
        return features
