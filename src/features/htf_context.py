from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.core.constants import DEFAULT_TIMESTAMP_COLUMN
from src.features.base import FeaturePack


def _build_completed_htf_candles(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    bucket_start = df[DEFAULT_TIMESTAMP_COLUMN].dt.floor(f"{timeframe_minutes}min")
    grouped = df.assign(_bucket_start=bucket_start).groupby("_bucket_start", sort=True)
    counts = grouped["open"].size()
    completed = counts[counts >= timeframe_minutes].index
    if completed.empty:
        return pd.DataFrame(columns=[DEFAULT_TIMESTAMP_COLUMN, "open", "high", "low", "close", "volume"])

    htf = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum() if "volume" in df.columns else 0.0,
        }
    ).loc[completed]
    htf = htf.reset_index(names="_bucket_start")
    htf[DEFAULT_TIMESTAMP_COLUMN] = htf["_bucket_start"] + pd.to_timedelta(timeframe_minutes, unit="min")
    return htf.drop(columns=["_bucket_start"])


def _merge_features_to_base(df: pd.DataFrame, htf_features: pd.DataFrame) -> pd.DataFrame:
    if htf_features.empty:
        return pd.DataFrame(index=df.index)

    anchor = df[[DEFAULT_TIMESTAMP_COLUMN]].copy()
    anchor["_row_id"] = df.index
    merged = pd.merge_asof(
        anchor.sort_values(DEFAULT_TIMESTAMP_COLUMN),
        htf_features.sort_values(DEFAULT_TIMESTAMP_COLUMN),
        on=DEFAULT_TIMESTAMP_COLUMN,
        direction="backward",
    )
    return merged.set_index("_row_id").reindex(df.index).drop(columns=[DEFAULT_TIMESTAMP_COLUMN])


class HTFContextFeaturePack(FeaturePack):
    name = "htf_context"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for timeframe_minutes in profile.htf_context_timeframes:
            htf = _build_completed_htf_candles(df, timeframe_minutes=timeframe_minutes)
            if htf.empty:
                continue

            timeframe_label = f"{timeframe_minutes}m"
            close = htf["close"]
            returns = close.pct_change(fill_method=None)
            rv = returns.rolling(
                window=profile.htf_context_vol_window,
                min_periods=profile.htf_context_vol_window,
            ).std()
            rolling_high = htf["high"].rolling(
                window=profile.htf_context_range_window,
                min_periods=profile.htf_context_range_window,
            ).max()
            rolling_low = htf["low"].rolling(
                window=profile.htf_context_range_window,
                min_periods=profile.htf_context_range_window,
            ).min()
            rolling_range = (rolling_high - rolling_low).replace(0, np.nan)
            rolling_mean = close.rolling(
                window=profile.htf_context_zscore_window,
                min_periods=profile.htf_context_zscore_window,
            ).mean()
            rolling_std = close.rolling(
                window=profile.htf_context_zscore_window,
                min_periods=profile.htf_context_zscore_window,
            ).std()
            net_move = close.diff(profile.htf_context_efficiency_window).abs()
            abs_diff_sum = close.diff().abs().rolling(
                window=profile.htf_context_efficiency_window,
                min_periods=profile.htf_context_efficiency_window,
            ).sum()
            trend_return = close.pct_change(
                profile.htf_context_trend_strength_window,
                fill_method=None,
            )

            htf_features = pd.DataFrame(
                {
                    DEFAULT_TIMESTAMP_COLUMN: htf[DEFAULT_TIMESTAMP_COLUMN],
                    f"htf_ret_{timeframe_label}_1": returns,
                    f"htf_rv_{timeframe_label}": rv,
                    f"htf_range_pos_{timeframe_label}": (close - rolling_low) / rolling_range,
                    f"htf_close_z_{timeframe_label}": (close - rolling_mean) / rolling_std.replace(0, np.nan),
                    f"htf_efficiency_{timeframe_label}": net_move / abs_diff_sum.replace(0, np.nan),
                }
            )

            if timeframe_minutes >= 15:
                htf_features[f"htf_regime_trend_strength_{timeframe_label}"] = (
                    trend_return.abs() / rv.replace(0, np.nan)
                )

            features = pd.concat([features, _merge_features_to_base(df, htf_features)], axis=1)

        return features
