from __future__ import annotations

import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class LaggedFeaturePack(FeaturePack):
    name = "lagged"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        feature_map: dict[str, pd.Series] = {}
        missing = [name for name in profile.lagged_feature_names if name not in df.columns]
        if missing:
            raise ValueError(
                "Lagged feature pack requires base features to be present first. "
                f"Missing columns: {missing}"
            )

        for column_name in profile.lagged_feature_names:
            source = df[column_name]
            for lag in profile.lagged_feature_lags:
                feature_map[f"{column_name}_lag{lag}"] = source.shift(lag)

        return pd.DataFrame(feature_map, index=df.index)
