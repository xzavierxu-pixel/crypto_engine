from __future__ import annotations

import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class MomentumFeaturePack(FeaturePack):
    name = "momentum"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        past_close = df["close"].shift(1)
        for window in profile.momentum_windows:
            features[f"ret_{window}"] = past_close.pct_change(window, fill_method=None)
        return features
