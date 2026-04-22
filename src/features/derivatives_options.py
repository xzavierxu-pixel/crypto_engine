from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


class DerivativesOptionsFeaturePack(FeaturePack):
    name = "derivatives_options"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        if not settings.derivatives.enabled or not settings.derivatives.options.enabled:
            return pd.DataFrame(index=df.index)
        if "raw_atm_iv_near" not in df.columns:
            raise ValueError(
                "Options feature pack requires a 'raw_atm_iv_near' column from the derivatives raw frame."
            )

        options_config = settings.derivatives.options
        atm_iv = df["raw_atm_iv_near"].shift(1)
        iv_term_slope = df["raw_iv_term_slope"].shift(1) if "raw_iv_term_slope" in df.columns else pd.Series(
            np.nan,
            index=df.index,
        )
        iv_term_slope = pd.to_numeric(iv_term_slope, errors="coerce").fillna(0.0)
        iv_zscore = _rolling_zscore(atm_iv, options_config.zscore_window)

        features = pd.DataFrame(index=df.index)
        features["atm_iv_near"] = atm_iv
        features["iv_term_slope"] = iv_term_slope
        features["iv_change_1h"] = atm_iv.pct_change(options_config.change_window, fill_method=None)
        features["iv_zscore"] = iv_zscore
        threshold = float(options_config.regime_zscore_threshold)
        features["iv_regime"] = np.select(
            [
                iv_zscore >= threshold,
                iv_zscore <= -threshold,
            ],
            [
                1.0,
                -1.0,
            ],
            default=0.0,
        )
        return features
