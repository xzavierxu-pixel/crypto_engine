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
        minute = ts.dt.minute.astype("float64")
        features["minute_sin"] = np.sin(2 * np.pi * minute / 60.0)
        features["minute_cos"] = np.cos(2 * np.pi * minute / 60.0)
        weekday = ts.dt.dayofweek.astype("float64")
        features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
        features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
        return features
