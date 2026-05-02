from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class CompletedBarMicrostructureFeaturePack(FeaturePack):
    name = "completed_bar_microstructure"

    def transform(self, df: pd.DataFrame, settings: Settings, profile: FeatureProfileConfig) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        open_ = pd.to_numeric(df["open"], errors="coerce").shift(1)
        high = pd.to_numeric(df["high"], errors="coerce").shift(1)
        low = pd.to_numeric(df["low"], errors="coerce").shift(1)
        close = pd.to_numeric(df["close"], errors="coerce").shift(1)
        volume = pd.to_numeric(df.get("volume", pd.Series(np.nan, index=df.index)), errors="coerce").shift(1)

        safe_close = close.replace(0, np.nan)
        safe_open = open_.replace(0, np.nan)
        safe_volume = volume.replace(0, np.nan)
        features["prev_bar_return"] = close / safe_open - 1.0
        features["prev_bar_range_bps"] = ((high - low) / safe_close) * 10000.0
        features["prev_bar_body_bps"] = ((close - open_) / safe_open) * 10000.0
        features["prev_bar_upper_wick_bps"] = ((high - pd.concat([open_, close], axis=1).max(axis=1)) / safe_close) * 10000.0
        features["prev_bar_lower_wick_bps"] = ((pd.concat([open_, close], axis=1).min(axis=1) - low) / safe_close) * 10000.0
        features["prev_bar_log_volume"] = np.log1p(volume.clip(lower=0.0))
        quote_column = "quote_volume" if "quote_volume" in df.columns else "quote_asset_volume" if "quote_asset_volume" in df.columns else None
        count_column = (
            "count"
            if "count" in df.columns
            else "number_of_trades"
            if "number_of_trades" in df.columns
            else "trade_count"
            if "trade_count" in df.columns
            else None
        )
        taker_base_column = (
            "taker_buy_volume"
            if "taker_buy_volume" in df.columns
            else "taker_buy_base_asset_volume"
            if "taker_buy_base_asset_volume" in df.columns
            else "taker_buy_base_volume"
            if "taker_buy_base_volume" in df.columns
            else None
        )
        taker_quote_column = (
            "taker_buy_quote_volume"
            if "taker_buy_quote_volume" in df.columns
            else "taker_buy_quote_asset_volume"
            if "taker_buy_quote_asset_volume" in df.columns
            else None
        )

        quote_volume = None
        if quote_column is not None:
            quote_volume = pd.to_numeric(df[quote_column], errors="coerce").shift(1)
            safe_quote_volume = quote_volume.replace(0, np.nan)
            features["prev_bar_quote_volume"] = quote_volume
            features["prev_bar_log_quote_volume"] = np.log1p(quote_volume.clip(lower=0.0))
            features["prev_bar_vwap_proxy"] = quote_volume / safe_volume

        if count_column is not None:
            trade_count = pd.to_numeric(df[count_column], errors="coerce").shift(1)
        else:
            trade_count = pd.Series(0.0, index=df.index, dtype="float64")
        safe_trade_count = trade_count.replace(0, np.nan)
        features["prev_bar_trade_count"] = trade_count
        features["prev_bar_log_trade_count"] = np.log1p(trade_count.clip(lower=0.0))
        features["prev_bar_avg_trade_size"] = (volume / safe_trade_count).fillna(0.0)
        for window in (3, 5, 10, 20):
            rolling_count_mean = trade_count.rolling(window, min_periods=window).mean()
            rolling_count_std = trade_count.rolling(window, min_periods=window).std().replace(0, np.nan)
            rolling_count_sum = trade_count.rolling(window, min_periods=window).sum()
            features[f"legal_prev_trade_count_sum_{window}"] = rolling_count_sum.fillna(0.0)
            features[f"legal_prev_relative_trade_count_{window}"] = (
                trade_count / rolling_count_mean.replace(0, np.nan)
            ).fillna(0.0)
            features[f"legal_prev_trade_count_z_{window}"] = (
                (trade_count - rolling_count_mean) / rolling_count_std
            ).fillna(0.0)
        if quote_volume is not None:
            features["prev_bar_avg_quote_per_trade"] = (quote_volume / safe_trade_count).fillna(0.0)

        has_taker_base = taker_base_column is not None
        if has_taker_base:
            taker_buy_volume = pd.to_numeric(df[taker_base_column], errors="coerce").shift(1)
            taker_buy_ratio = (taker_buy_volume / safe_volume).clip(0.0, 1.0).fillna(0.0)
            taker_sell_volume = (volume - taker_buy_volume).clip(lower=0.0).fillna(0.0)
            taker_net_volume = 2.0 * taker_buy_volume - volume
        else:
            taker_buy_volume = pd.Series(0.0, index=df.index, dtype="float64")
            taker_buy_ratio = pd.Series(0.5, index=df.index, dtype="float64")
            taker_sell_volume = pd.Series(0.0, index=df.index, dtype="float64")
            taker_net_volume = pd.Series(0.0, index=df.index, dtype="float64")
        features["prev_bar_taker_buy_volume"] = taker_buy_volume
        features["prev_bar_taker_buy_ratio"] = taker_buy_ratio
        features["prev_bar_taker_sell_ratio"] = (1.0 - taker_buy_ratio).clip(0.0, 1.0)
        features["prev_bar_taker_imbalance"] = (2.0 * taker_buy_ratio - 1.0).clip(-1.0, 1.0)
        features["prev_bar_taker_net_volume"] = taker_net_volume
        features["prev_bar_taker_buy_ratio_delta_1"] = taker_buy_ratio.diff(1).fillna(0.0)
        features["prev_bar_taker_buy_ratio_mean_5"] = taker_buy_ratio.rolling(5, min_periods=5).mean().fillna(0.0)
        taker_ratio_std = taker_buy_ratio.rolling(20, min_periods=20).std().replace(0, np.nan)
        taker_ratio_mean = taker_buy_ratio.rolling(20, min_periods=20).mean()
        features["prev_bar_taker_buy_ratio_zscore_20"] = ((taker_buy_ratio - taker_ratio_mean) / taker_ratio_std).fillna(0.0)
        features["legal_prev_log_taker_buy_base_volume"] = np.log1p(taker_buy_volume.clip(lower=0.0))
        features["legal_prev_taker_sell_base_volume"] = taker_sell_volume
        for window in (3, 5, 10, 20):
            rolling_buy_mean = taker_buy_volume.rolling(window, min_periods=window).mean()
            rolling_buy_std = taker_buy_volume.rolling(window, min_periods=window).std().replace(0, np.nan)
            rolling_buy_sum = taker_buy_volume.rolling(window, min_periods=window).sum()
            features[f"legal_prev_taker_buy_base_volume_sum_{window}"] = rolling_buy_sum.fillna(0.0)
            features[f"legal_prev_relative_taker_buy_base_volume_{window}"] = (
                taker_buy_volume / rolling_buy_mean.replace(0, np.nan)
            ).fillna(0.0)
            features[f"legal_prev_taker_buy_base_volume_z_{window}"] = (
                (taker_buy_volume - rolling_buy_mean) / rolling_buy_std
            ).fillna(0.0)

        if taker_quote_column is not None and quote_volume is not None:
            taker_buy_quote_volume = pd.to_numeric(df[taker_quote_column], errors="coerce").shift(1)
            taker_buy_quote_ratio = (taker_buy_quote_volume / safe_quote_volume).clip(0.0, 1.0)
            features["prev_bar_taker_buy_quote_volume"] = taker_buy_quote_volume
            features["prev_bar_taker_buy_quote_ratio"] = taker_buy_quote_ratio
            features["prev_bar_taker_quote_imbalance"] = (2.0 * taker_buy_quote_ratio - 1.0).clip(-1.0, 1.0)
            features["prev_bar_taker_net_quote_volume"] = 2.0 * taker_buy_quote_volume - quote_volume
        return features


class FlowPressureFeaturePack(FeaturePack):
    name = "flow_pressure"

    def transform(self, df: pd.DataFrame, settings: Settings, profile: FeatureProfileConfig) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        close = pd.to_numeric(df["close"], errors="coerce").shift(1)
        open_ = pd.to_numeric(df["open"], errors="coerce").shift(1)
        volume = pd.to_numeric(df["volume"], errors="coerce").shift(1)
        taker_buy_volume = pd.to_numeric(
            df.get("taker_buy_volume", df.get("taker_buy_base_asset_volume", pd.Series(np.nan, index=df.index))),
            errors="coerce",
        ).shift(1)
        has_taker_flow = taker_buy_volume.notna().any()
        if has_taker_flow:
            signed_volume = 2.0 * taker_buy_volume - volume
        else:
            signed = np.sign(close - open_).fillna(0.0)
            signed_volume = signed * volume
        dollar_flow = signed_volume * close
        total_volume = volume.replace(0, np.nan)

        features["taker_buy_ratio"] = ((signed_volume.clip(lower=0.0)) / total_volume).clip(0.0, 1.0)
        features["taker_sell_ratio"] = ((-signed_volume.clip(upper=0.0)) / total_volume).clip(0.0, 1.0)
        features["taker_imbalance"] = features["taker_buy_ratio"] - features["taker_sell_ratio"]
        features["taker_imbalance_mean_5"] = features["taker_imbalance"].rolling(5, min_periods=5).mean()
        imbalance_std = features["taker_imbalance"].rolling(20, min_periods=20).std().replace(0, np.nan)
        imbalance_mean = features["taker_imbalance"].rolling(20, min_periods=20).mean()
        features["taker_imbalance_zscore_20"] = ((features["taker_imbalance"] - imbalance_mean) / imbalance_std).fillna(0.0)
        features["taker_imbalance_slope"] = features["taker_imbalance"].diff(1)
        features["signed_dollar_flow"] = dollar_flow
        return features


class BookPressureFeaturePack(FeaturePack):
    name = "book_pressure"

    def transform(self, df: pd.DataFrame, settings: Settings, profile: FeatureProfileConfig) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        required = {"raw_bid_price", "raw_bid_qty", "raw_ask_price", "raw_ask_qty"}
        if not settings.derivatives.enabled or not settings.derivatives.book_ticker.enabled or not required.issubset(df.columns):
            return features
        bid_price = pd.to_numeric(df["raw_bid_price"], errors="coerce").shift(1)
        ask_price = pd.to_numeric(df["raw_ask_price"], errors="coerce").shift(1)
        bid_qty = pd.to_numeric(df["raw_bid_qty"], errors="coerce").shift(1)
        ask_qty = pd.to_numeric(df["raw_ask_qty"], errors="coerce").shift(1)
        mid_price = (bid_price + ask_price) / 2.0
        total_qty = (bid_qty + ask_qty).replace(0, np.nan)
        microprice = ((ask_price * bid_qty) + (bid_price * ask_qty)) / total_qty

        features["spread_bps"] = ((ask_price - bid_price) / mid_price.replace(0, np.nan)) * 10000.0
        features["mid_price"] = mid_price
        features["microprice"] = microprice
        features["bid_ask_qty_imbalance"] = (bid_qty - ask_qty) / total_qty
        features["spread_change"] = features["spread_bps"].diff(1)
        features["imbalance_change"] = features["bid_ask_qty_imbalance"].diff(1)
        features["short_horizon_mid_drift"] = mid_price.pct_change(5, fill_method=None)
        return features


class SecondLevelMicrostructureFeaturePack(FeaturePack):
    name = "second_level_microstructure"

    def transform(self, df: pd.DataFrame, settings: Settings, profile: FeatureProfileConfig) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        close = pd.to_numeric(df["close"], errors="coerce").shift(1)
        volume = pd.to_numeric(df["volume"], errors="coerce").shift(1)
        returns = close.pct_change(fill_method=None)
        for window in (5, 10, 30, 60):
            minute_window = max(1, int(np.ceil(window / 60)))
            features[f"micro_ret_{window}s"] = close.pct_change(minute_window, fill_method=None)
            if minute_window == 1:
                features[f"micro_rv_{window}s"] = returns.abs()
            else:
                features[f"micro_rv_{window}s"] = returns.rolling(minute_window, min_periods=minute_window).std()
        features["second_level_price_slope"] = close.diff(1)
        features["second_level_signed_dollar_flow"] = np.sign(returns.fillna(0.0)) * close * volume
        features["second_level_volume_burst"] = volume / volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
        features["price_direction_flips_30s"] = (np.sign(returns) != np.sign(returns.shift(1))).astype(float)
        features["last_second_reversal_flag"] = (np.sign(returns) != np.sign(returns.shift(1))).astype(float)
        features["late_window_acceleration_flag"] = (returns.diff(1) > 0).astype(float)
        return features


class EventWindowBurstFeaturePack(FeaturePack):
    name = "event_window_burst"

    def transform(self, df: pd.DataFrame, settings: Settings, profile: FeatureProfileConfig) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        open_ = pd.to_numeric(df["open"], errors="coerce").shift(1)
        high = pd.to_numeric(df["high"], errors="coerce").shift(1)
        low = pd.to_numeric(df["low"], errors="coerce").shift(1)
        close = pd.to_numeric(df["close"], errors="coerce").shift(1)
        volume = pd.to_numeric(df["volume"], errors="coerce").shift(1)
        returns = close.pct_change(fill_method=None)
        bullish = (close > open_).astype(float)
        bearish = (close < open_).astype(float)
        candle_range = (high - low).replace(0, np.nan)
        body = (close - open_).abs()

        features["consecutive_bullish_bar_count"] = bullish.rolling(5, min_periods=5).sum()
        features["consecutive_bearish_bar_count"] = bearish.rolling(5, min_periods=5).sum()
        features["burst_volume_flag"] = (volume > volume.rolling(20, min_periods=20).mean() * 1.5).astype(float)
        features["wick_rejection_count"] = ((body / candle_range) < 0.35).astype(float).rolling(5, min_periods=5).sum()
        features["compression_to_expansion_flag"] = (candle_range > candle_range.rolling(20, min_periods=20).mean()).astype(float)
        features["max_positive_single_bar_return"] = returns.rolling(5, min_periods=5).max()
        features["max_negative_single_bar_return"] = returns.rolling(5, min_periods=5).min()
        features["directional_persistence_score"] = np.sign(returns).rolling(5, min_periods=5).sum()
        return features


class SideSpecificTransformsFeaturePack(FeaturePack):
    name = "side_specific_transforms"

    def transform(self, df: pd.DataFrame, settings: Settings, profile: FeatureProfileConfig) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        taker_imbalance = df.get("taker_imbalance", pd.Series(0.0, index=df.index)).astype("float64")
        mid_drift = df.get("short_horizon_mid_drift", pd.Series(0.0, index=df.index)).astype("float64")
        basis = df.get("basis_bps", df.get("premium_index", pd.Series(0.0, index=df.index))).astype("float64")

        features["positive_taker_imbalance"] = taker_imbalance.clip(lower=0.0)
        features["negative_taker_imbalance"] = (-taker_imbalance.clip(upper=0.0))
        features["bullish_burst_score"] = df.get("consecutive_bullish_bar_count", pd.Series(0.0, index=df.index))
        features["bearish_burst_score"] = df.get("consecutive_bearish_bar_count", pd.Series(0.0, index=df.index))
        features["upward_mid_drift"] = mid_drift.clip(lower=0.0)
        features["downward_mid_drift"] = -mid_drift.clip(upper=0.0)
        features["positive_basis_pressure"] = basis.clip(lower=0.0)
        features["negative_basis_pressure"] = -basis.clip(upper=0.0)
        return features
