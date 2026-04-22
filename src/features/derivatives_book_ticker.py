from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


class DerivativesBookTickerFeaturePack(FeaturePack):
    name = "derivatives_book_ticker"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        if not settings.derivatives.enabled or not settings.derivatives.book_ticker.enabled:
            return pd.DataFrame(index=df.index)
        required_columns = {"raw_bid_price", "raw_bid_qty", "raw_ask_price", "raw_ask_qty"}
        missing = sorted(required_columns.difference(df.columns))
        if missing:
            return pd.DataFrame(index=df.index)

        bid_price = pd.to_numeric(df["raw_bid_price"], errors="coerce").shift(1)
        ask_price = pd.to_numeric(df["raw_ask_price"], errors="coerce").shift(1)
        bid_qty = pd.to_numeric(df["raw_bid_qty"], errors="coerce").shift(1)
        ask_qty = pd.to_numeric(df["raw_ask_qty"], errors="coerce").shift(1)
        mid_price = (bid_price + ask_price) / 2.0
        total_qty = bid_qty + ask_qty
        safe_total_qty = total_qty.replace(0, np.nan)
        microprice = ((ask_price * bid_qty) + (bid_price * ask_qty)) / safe_total_qty

        window = settings.derivatives.book_ticker.zscore_window
        features = pd.DataFrame(index=df.index)
        features["book_spread_bps"] = ((ask_price - bid_price) / mid_price.replace(0, np.nan)) * 10000.0
        features["book_imbalance"] = (bid_qty - ask_qty) / safe_total_qty
        features["book_microprice_offset_bps"] = ((microprice - mid_price) / mid_price.replace(0, np.nan)) * 10000.0
        features["book_top_depth_total"] = total_qty
        features[f"book_spread_bps_zscore_{window}"] = _rolling_zscore(features["book_spread_bps"], window)
        return features
