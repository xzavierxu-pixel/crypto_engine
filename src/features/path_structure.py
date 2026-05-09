from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)

    def compute(values: np.ndarray) -> float:
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)

    return series.rolling(window=window, min_periods=window).apply(compute, raw=True)


def _consecutive_streaks(returns: pd.Series, direction: str) -> pd.Series:
    streaks: list[int] = []
    current = 0
    for value in returns.fillna(0.0):
        match direction:
            case "up":
                current = current + 1 if value > 0 else 0
            case "down":
                current = current + 1 if value < 0 else 0
            case _:
                raise ValueError(f"Unsupported direction '{direction}'.")
        streaks.append(current)
    return pd.Series(streaks, index=returns.index, dtype="float64")


class PathStructureFeaturePack(FeaturePack):
    name = "path_structure"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        past_close = df["close"].shift(1)
        past_high = df["high"].shift(1)
        past_low = df["low"].shift(1)
        base_return = past_close.pct_change(fill_method=None)

        features["consecutive_up"] = _consecutive_streaks(base_return, "up")
        features["consecutive_down"] = _consecutive_streaks(base_return, "down")

        for window in profile.slope_windows:
            features[f"slope_{window}"] = _rolling_slope(past_close, window)

        for window in profile.range_windows:
            rolling_high = past_high.rolling(window=window, min_periods=window).max()
            rolling_low = past_low.rolling(window=window, min_periods=window).min()
            denom = (rolling_high - rolling_low).replace(0, np.nan)
            features[f"range_{window}"] = rolling_high - rolling_low
            features[f"range_pos_{window}"] = (past_close - rolling_low) / denom

            if profile.use_vwap_distance and "volume" in df.columns:
                volume = df["volume"].shift(1)
                turnover = (past_close * volume).rolling(window=window, min_periods=window).sum()
                volume_sum = volume.rolling(window=window, min_periods=window).sum().replace(0, np.nan)
                rolling_vwap = turnover / volume_sum
                features[f"vwap_dist_{window}"] = (past_close - rolling_vwap) / rolling_vwap.replace(0, np.nan)

        return features
