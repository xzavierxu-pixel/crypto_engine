from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


def _rolling_percent_rank(series: pd.Series, window: int) -> pd.Series:
    def compute(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float((values <= values[-1]).mean())

    return series.rolling(window=window, min_periods=window).apply(compute, raw=True)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return components.max(axis=1)


class CompressionBreakoutFeaturePack(FeaturePack):
    name = "compression_breakout"

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
        past_range = past_high - past_low
        rolling_mean = past_close.rolling(
            window=profile.compression_window,
            min_periods=profile.compression_window,
        ).mean()
        rolling_std = past_close.rolling(
            window=profile.compression_window,
            min_periods=profile.compression_window,
        ).std()
        upper_band = rolling_mean + 2.0 * rolling_std
        lower_band = rolling_mean - 2.0 * rolling_std
        bb_width = (upper_band - lower_band) / rolling_mean.replace(0, np.nan)
        donchian_high = past_high.rolling(
            window=profile.compression_window,
            min_periods=profile.compression_window,
        ).max()
        donchian_low = past_low.rolling(
            window=profile.compression_window,
            min_periods=profile.compression_window,
        ).min()
        donchian_width = (donchian_high - donchian_low) / past_close.replace(0, np.nan)

        tr = _true_range(past_high, past_low, past_close)
        atr_short = tr.rolling(
            window=profile.compression_atr_short_window,
            min_periods=profile.compression_atr_short_window,
        ).mean()
        atr_long = tr.rolling(
            window=profile.compression_atr_long_window,
            min_periods=profile.compression_atr_long_window,
        ).mean()
        atr_ratio = atr_short / atr_long.replace(0, np.nan)

        features[f"bb_width_{profile.compression_window}"] = bb_width
        features[f"bb_width_pct_rank_{profile.compression_rank_window}"] = _rolling_percent_rank(
            bb_width,
            profile.compression_rank_window,
        )
        features[f"donchian_width_{profile.compression_window}"] = donchian_width
        features[
            f"atr_ratio_{profile.compression_atr_short_window}_{profile.compression_atr_long_window}"
        ] = atr_ratio

        for nr_window in profile.compression_nr_windows:
            rolling_min = past_range.rolling(window=nr_window, min_periods=nr_window).min()
            features[f"nr{nr_window}_flag"] = (past_range <= rolling_min).astype("float64")

        features[f"breakout_up_dist_{profile.compression_window}"] = (
            donchian_high - past_close
        ) / past_close.replace(0, np.nan)
        features[f"breakout_down_dist_{profile.compression_window}"] = (
            past_close - donchian_low
        ) / past_close.replace(0, np.nan)

        bb_rank = features[f"bb_width_pct_rank_{profile.compression_rank_window}"]
        atr_rank = _rolling_percent_rank(atr_ratio, profile.compression_rank_window)
        donchian_rank = _rolling_percent_rank(donchian_width, profile.compression_rank_window)
        nr_components = [
            features[f"nr{nr_window}_flag"] for nr_window in profile.compression_nr_windows
        ]
        nr_component = sum(nr_components) / len(nr_components) if nr_components else 0.0
        features["compression_score"] = pd.concat(
            [
                1.0 - bb_rank,
                1.0 - atr_rank,
                1.0 - donchian_rank,
                nr_component if isinstance(nr_component, pd.Series) else pd.Series(nr_component, index=df.index),
            ],
            axis=1,
        ).mean(axis=1)

        return features
