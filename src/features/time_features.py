from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.core.constants import DEFAULT_TIMESTAMP_COLUMN
from src.features.base import FeaturePack


class TimeFeaturePack(FeaturePack):
    name = "time"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        if not profile.use_time_features:
            return features

        ts = pd.to_datetime(df[DEFAULT_TIMESTAMP_COLUMN], utc=True)
        hour = ts.dt.hour + ts.dt.minute / 60.0
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        features["minute_bucket"] = ts.dt.minute.astype("float64")
        return features
