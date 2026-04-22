from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float((x_centered**2).sum())

    def compute(values: np.ndarray) -> float:
        valid = np.isfinite(values)
        if valid.sum() < window:
            return float("nan")
        y = values.astype(float)
        y_centered = y - y.mean()
        return float(np.dot(x_centered, y_centered) / denominator)

    return series.rolling(window=window, min_periods=window).apply(compute, raw=True)


class DerivativesOIFeaturePack(FeaturePack):
    name = "derivatives_oi"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        if not settings.derivatives.enabled or not settings.derivatives.oi.enabled:
            return pd.DataFrame(index=df.index)
        if "raw_open_interest" not in df.columns:
            return pd.DataFrame(index=df.index)

        oi_config = settings.derivatives.oi
        oi_level = df["raw_open_interest"].shift(1)
        features = pd.DataFrame(index=df.index)
        features["oi_level"] = oi_level
        if "raw_oi_notional" in df.columns:
            features["oi_notional_level"] = df["raw_oi_notional"].shift(1)

        change_windows = sorted(set(oi_config.change_windows))
        short_window = change_windows[0]
        long_window = change_windows[-1]
        for window in change_windows:
            label = f"{window}m" if window < 60 else f"{window // 60}h"
            features[f"oi_change_{label}"] = oi_level.pct_change(window, fill_method=None)

        features["oi_zscore"] = _rolling_zscore(oi_level, oi_config.zscore_window)
        features["oi_slope"] = _rolling_slope(oi_level, oi_config.slope_window)

        short_change = features[f"oi_change_{short_window}m"] if short_window < 60 else features[f"oi_change_{short_window // 60}h"]
        features["oi_x_basis"] = short_change * df.get("basis_mark_spot", pd.Series(np.nan, index=df.index))
        features["oi_x_funding"] = short_change * df.get("funding_rate", pd.Series(np.nan, index=df.index))
        return features
