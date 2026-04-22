from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


class DerivativesBasisFeaturePack(FeaturePack):
    name = "derivatives_basis"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        if not settings.derivatives.enabled or not settings.derivatives.basis.enabled:
            return pd.DataFrame(index=df.index)

        if "close" not in df.columns:
            raise ValueError("Basis feature pack requires spot close prices.")

        basis_settings = settings.derivatives.basis
        window = basis_settings.zscore_window
        spot_close = df["close"].shift(1).replace(0, np.nan)
        features = pd.DataFrame(index=df.index)

        if basis_settings.use_mark_price:
            if "raw_mark_price" not in df.columns:
                raise ValueError(
                    "Basis feature pack requires a 'raw_mark_price' column when mark price features are enabled."
                )
            mark_price = df["raw_mark_price"].shift(1)
            basis_mark_spot = mark_price / spot_close - 1.0
            features["basis_mark_spot"] = basis_mark_spot
            features["basis_mark_spot_lag1"] = basis_mark_spot.shift(1)
            features["basis_mark_spot_change_1"] = basis_mark_spot.diff()
            features[f"basis_mark_spot_zscore_{window}"] = _rolling_zscore(basis_mark_spot, window)
            features["basis_sign"] = np.sign(basis_mark_spot).astype("float64")

        if basis_settings.use_index_price:
            if "raw_index_price" not in df.columns:
                raise ValueError(
                    "Basis feature pack requires a 'raw_index_price' column when index price features are enabled."
                )
            index_price = df["raw_index_price"].shift(1)
            basis_index_spot = index_price / spot_close - 1.0
            features["basis_index_spot"] = basis_index_spot

        if basis_settings.use_premium_index:
            if "raw_premium_index" not in df.columns:
                raise ValueError(
                    "Basis feature pack requires a 'raw_premium_index' column when premium index features are enabled."
                )
            premium_index = df["raw_premium_index"].shift(1)
            features["premium_index"] = premium_index
            features[f"premium_index_zscore_{window}"] = _rolling_zscore(premium_index, window)

        return features
