from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class VolumeFeaturePack(FeaturePack):
    name = "volume"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        if "volume" not in df.columns:
            return features

        volume = df["volume"].shift(1)
        past_close = df["close"].shift(1)
        ret_1 = past_close.pct_change(fill_method=None)

        features["signed_volume_1"] = np.sign(ret_1.fillna(0.0)) * volume

        for window in profile.volume_windows:
            rolling_mean = volume.rolling(window=window, min_periods=window).mean()
            rolling_std = volume.rolling(window=window, min_periods=window).std()
            volume_sum = volume.rolling(window=window, min_periods=window).sum()

            features[f"relative_volume_{window}"] = volume / rolling_mean.replace(0, np.nan)
            features[f"volume_z_{window}"] = (volume - rolling_mean) / rolling_std.replace(0, np.nan)
            features[f"volume_share_{window}"] = volume / volume_sum.replace(0, np.nan)

        return features
