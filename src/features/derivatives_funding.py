from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


class DerivativesFundingFeaturePack(FeaturePack):
    name = "derivatives_funding"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        if not settings.derivatives.enabled or not settings.derivatives.funding.enabled:
            return pd.DataFrame(index=df.index)
        if "raw_funding_rate" not in df.columns:
            raise ValueError(
                "Funding feature pack requires a 'raw_funding_rate' column from the derivatives raw frame."
            )

        funding_rate = df["raw_funding_rate"].shift(1)
        window = settings.derivatives.funding.zscore_window
        features = pd.DataFrame(index=df.index)
        features["funding_rate"] = funding_rate
        features["funding_rate_lag1"] = funding_rate.shift(1)
        features["funding_rate_change_1"] = funding_rate.diff()
        features[f"funding_rate_zscore_{window}"] = _rolling_zscore(funding_rate, window)
        features["funding_is_pos"] = (funding_rate > 0).astype("float64")
        features["funding_abs"] = funding_rate.abs()
        return features
