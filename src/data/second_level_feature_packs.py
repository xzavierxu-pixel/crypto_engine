from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


DEFAULT_SECOND_LEVEL_PACKS = [
    "second_level_momentum",
    "second_level_volatility",
    "second_level_volume",
    "second_level_candle_structure",
    "second_level_path_structure",
    "second_level_trade_microstructure",
    "second_level_book_microstructure",
    "second_level_depth",
    "second_level_cross_market",
    "second_level_interaction_bank",
    "second_level_lagged",
]


@dataclass(frozen=True)
class SecondLevelFeatureProfile:
    packs: list[str] = field(default_factory=lambda: list(DEFAULT_SECOND_LEVEL_PACKS))
    windows: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 15, 30, 60, 120, 300])
    compact_windows: list[int] = field(default_factory=lambda: [5, 10, 30, 60, 300])
    slope_windows: list[int] = field(default_factory=lambda: [5, 10, 30, 60])
    range_windows: list[int] = field(default_factory=lambda: [5, 10, 30, 60, 300])
    lagged_feature_names: list[str] = field(default_factory=lambda: [
        "sl_return_5s",
        "sl_return_10s",
        "sl_return_30s",
        "sl_return_60s",
        "sl_rv_30s",
        "sl_taker_imbalance_30s",
        "sl_signed_dollar_flow_30s",
        "sl_trade_count_30s",
        "sl_total_volume_30s",
        "sl_spread_bps",
        "sl_bid_ask_qty_imbalance",
        "sl_microprice_premium",
        "sl_mirror_ret_30s",
        "sl_mirror_relative_volume_30s",
    ])
    lagged_feature_lags: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 30, 60])


class SecondLevelFeaturePack(ABC):
    name: str

    @abstractmethod
    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        raise NotImplementedError


def _safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    return left / right.replace(0, np.nan)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    return (series - series.shift(window - 1)) / float(max(1, window - 1))


def _streaks(returns: pd.Series, direction: str) -> pd.Series:
    streaks: list[int] = []
    current = 0
    for value in returns.fillna(0.0):
        if direction == "up":
            current = current + 1 if value > 0 else 0
        elif direction == "down":
            current = current + 1 if value < 0 else 0
        else:
            raise ValueError(f"Unsupported streak direction: {direction}")
        streaks.append(current)
    return pd.Series(streaks, index=returns.index, dtype="float64")


def _existing(frame: pd.DataFrame, names: list[str]) -> list[str]:
    return [name for name in names if name in frame.columns]


class SecondLevelMomentumPack(SecondLevelFeaturePack):
    name = "second_level_momentum"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        close = pd.to_numeric(store["sec_close"], errors="coerce")
        features: dict[str, pd.Series] = {}
        for window in profile.windows:
            lagged = close.shift(window)
            features[f"sl_mirror_ret_{window}s"] = close / lagged - 1.0
            features[f"sl_mirror_log_ret_{window}s"] = np.log(close / lagged).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame(features, index=store.index)


class SecondLevelVolatilityPack(SecondLevelFeaturePack):
    name = "second_level_volatility"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        close = pd.to_numeric(store["sec_close"], errors="coerce")
        ret = close.pct_change(fill_method=None).fillna(0.0)
        features: dict[str, pd.Series] = {}
        for window in profile.windows:
            features[f"sl_mirror_rv_{window}s"] = ret.rolling(window, min_periods=1).std(ddof=0)
            features[f"sl_mirror_rvar_{window}s"] = ret.rolling(window, min_periods=1).var(ddof=0)
            features[f"sl_mirror_mean_abs_ret_{window}s"] = ret.abs().rolling(window, min_periods=1).mean()
            features[f"sl_mirror_max_abs_ret_{window}s"] = ret.abs().rolling(window, min_periods=1).max()
        return pd.DataFrame(features, index=store.index)


class SecondLevelVolumePack(SecondLevelFeaturePack):
    name = "second_level_volume"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        volume = pd.to_numeric(store["sec_volume"], errors="coerce").fillna(0.0)
        quote_volume = pd.to_numeric(store["sec_quote_volume"], errors="coerce").fillna(0.0)
        trade_count = pd.to_numeric(store["sec_trade_count"], errors="coerce").fillna(0.0)
        ret = pd.to_numeric(store["sec_close"], errors="coerce").pct_change(fill_method=None).fillna(0.0)
        features: dict[str, pd.Series] = {
            "sl_mirror_signed_volume_1s": np.sign(ret) * volume,
            "sl_mirror_signed_quote_volume_1s": np.sign(ret) * quote_volume,
        }
        for source_name, series in {
            "volume": volume,
            "quote_volume": quote_volume,
            "trade_count": trade_count,
        }.items():
            for window in profile.compact_windows:
                rolling_mean = series.rolling(window=window, min_periods=1).mean()
                rolling_std = series.rolling(window=window, min_periods=2).std()
                rolling_sum = series.rolling(window=window, min_periods=1).sum()
                features[f"sl_mirror_relative_{source_name}_{window}s"] = _safe_divide(series, rolling_mean)
                features[f"sl_mirror_{source_name}_z_{window}s"] = _safe_divide(series - rolling_mean, rolling_std)
                features[f"sl_mirror_{source_name}_share_{window}s"] = _safe_divide(series, rolling_sum)
        return pd.DataFrame(features, index=store.index)


class SecondLevelCandleStructurePack(SecondLevelFeaturePack):
    name = "second_level_candle_structure"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        open_ = pd.to_numeric(store["sec_open"], errors="coerce")
        high = pd.to_numeric(store["sec_high"], errors="coerce")
        low = pd.to_numeric(store["sec_low"], errors="coerce")
        close = pd.to_numeric(store["sec_close"], errors="coerce")
        range_ = (high - low).replace(0, np.nan)
        body = close - open_
        upper = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower = pd.concat([open_, close], axis=1).min(axis=1) - low
        features: dict[str, pd.Series] = {
            "sl_mirror_body_pct_1s": _safe_divide(body, open_),
            "sl_mirror_true_range_pct_1s": _safe_divide(high - low, close),
            "sl_mirror_upper_wick_ratio_1s": _safe_divide(upper, range_),
            "sl_mirror_lower_wick_ratio_1s": _safe_divide(lower, range_),
            "sl_mirror_close_location_1s": _safe_divide(close - low, range_),
        }
        for window in profile.range_windows:
            rolling_mean = close.rolling(window=window, min_periods=1).mean()
            rolling_std = close.rolling(window=window, min_periods=2).std()
            abs_path = close.diff().abs().rolling(window=window, min_periods=1).sum()
            net_move = close.diff(window).abs()
            features[f"sl_mirror_close_z_{window}s"] = _safe_divide(close - rolling_mean, rolling_std)
            features[f"sl_mirror_efficiency_{window}s"] = _safe_divide(net_move, abs_path)
        return pd.DataFrame(features, index=store.index)


class SecondLevelPathStructurePack(SecondLevelFeaturePack):
    name = "second_level_path_structure"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        close = pd.to_numeric(store["sec_close"], errors="coerce")
        high = pd.to_numeric(store["sec_high"], errors="coerce")
        low = pd.to_numeric(store["sec_low"], errors="coerce")
        volume = pd.to_numeric(store["sec_volume"], errors="coerce").fillna(0.0)
        ret = close.pct_change(fill_method=None)
        features: dict[str, pd.Series] = {
            "sl_mirror_consecutive_up": _streaks(ret, "up"),
            "sl_mirror_consecutive_down": _streaks(ret, "down"),
        }
        for window in profile.slope_windows:
            features[f"sl_mirror_slope_{window}s"] = _rolling_slope(close, window)
            features[f"sl_mirror_return_slope_{window}s"] = _rolling_slope(ret.fillna(0.0), window)
        for window in profile.range_windows:
            rolling_high = high.rolling(window=window, min_periods=1).max()
            rolling_low = low.rolling(window=window, min_periods=1).min()
            denom = (rolling_high - rolling_low).replace(0, np.nan)
            turnover = (close * volume).rolling(window=window, min_periods=1).sum()
            volume_sum = volume.rolling(window=window, min_periods=1).sum()
            vwap = _safe_divide(turnover, volume_sum)
            features[f"sl_mirror_range_{window}s"] = rolling_high - rolling_low
            features[f"sl_mirror_range_pos_{window}s"] = _safe_divide(close - rolling_low, denom)
            features[f"sl_mirror_vwap_dist_{window}s"] = _safe_divide(close - vwap, vwap)
        return pd.DataFrame(features, index=store.index)


class SecondLevelTradeMicrostructurePack(SecondLevelFeaturePack):
    name = "second_level_trade_microstructure"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        for window in profile.compact_windows:
            imbalance = store.get(f"sl_taker_imbalance_{window}s")
            signed_flow = store.get(f"sl_signed_dollar_flow_{window}s")
            count = store.get(f"sl_trade_count_{window}s")
            if imbalance is not None:
                features[f"sl_taker_imbalance_positive_{window}s"] = imbalance.clip(lower=0.0)
                features[f"sl_taker_imbalance_negative_{window}s"] = -imbalance.clip(upper=0.0)
                features[f"sl_taker_imbalance_delta_{window}s"] = imbalance.diff(1)
            if signed_flow is not None:
                rolling_mean = signed_flow.rolling(window=300, min_periods=30).mean()
                rolling_std = signed_flow.rolling(window=300, min_periods=30).std(ddof=0)
                features[f"sl_signed_dollar_flow_z_{window}s"] = _safe_divide(signed_flow - rolling_mean, rolling_std)
                features[f"sl_signed_dollar_flow_delta_{window}s"] = signed_flow.diff(1)
            if count is not None:
                rolling_mean = count.rolling(window=300, min_periods=30).mean()
                rolling_std = count.rolling(window=300, min_periods=30).std(ddof=0)
                features[f"sl_trade_count_z_{window}s"] = _safe_divide(count - rolling_mean, rolling_std)
        return pd.DataFrame(features, index=store.index)


class SecondLevelBookMicrostructurePack(SecondLevelFeaturePack):
    name = "second_level_book_microstructure"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        for name in ["sl_spread_bps", "sl_bid_ask_qty_imbalance", "sl_microprice_premium", "sl_ofi_30s"]:
            if name not in store.columns:
                continue
            series = pd.to_numeric(store[name], errors="coerce")
            features[f"{name}_delta_1s"] = series.diff(1)
            features[f"{name}_rolling_z_300s"] = _safe_divide(
                series - series.rolling(300, min_periods=30).mean(),
                series.rolling(300, min_periods=30).std(ddof=0),
            )
        if {"sl_bid_ask_qty_imbalance", "sl_spread_bps"}.issubset(store.columns):
            features["sl_book_imbalance_x_spread_bps"] = store["sl_bid_ask_qty_imbalance"] * store["sl_spread_bps"]
        return pd.DataFrame(features, index=store.index)


class SecondLevelDepthPack(SecondLevelFeaturePack):
    name = "second_level_depth"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        depth_columns = _existing(store, [
            "sl_bid_depth_5",
            "sl_ask_depth_5",
            "sl_depth_imbalance_5",
            "sl_weighted_depth_imbalance_5",
            "sl_book_slope_bid",
            "sl_book_slope_ask",
        ])
        features: dict[str, pd.Series] = {}
        for name in depth_columns:
            series = pd.to_numeric(store[name], errors="coerce")
            features[f"{name}_delta_1s"] = series.diff(1)
            features[f"{name}_rolling_z_300s"] = _safe_divide(
                series - series.rolling(300, min_periods=30).mean(),
                series.rolling(300, min_periods=30).std(ddof=0),
            )
        return pd.DataFrame(features, index=store.index)


class SecondLevelCrossMarketPack(SecondLevelFeaturePack):
    name = "second_level_cross_market"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        prefixes = ("sl_perp_", "sl_btc_minus_eth_", "sl_crypto_beta_")
        columns = [name for name in store.columns if name.startswith(prefixes)]
        features: dict[str, pd.Series] = {}
        for name in columns:
            series = pd.to_numeric(store[name], errors="coerce")
            features[f"{name}_delta_1s"] = series.diff(1)
            features[f"{name}_rolling_z_300s"] = _safe_divide(
                series - series.rolling(300, min_periods=30).mean(),
                series.rolling(300, min_periods=30).std(ddof=0),
            )
        return pd.DataFrame(features, index=store.index)


class SecondLevelInteractionBankPack(SecondLevelFeaturePack):
    name = "second_level_interaction_bank"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        returns = _existing(store, ["sl_mirror_ret_5s", "sl_mirror_ret_10s", "sl_mirror_ret_30s", "sl_mirror_ret_60s", "sl_return_30s"])
        realized_vol = _existing(store, ["sl_mirror_rv_10s", "sl_mirror_rv_30s", "sl_mirror_rv_60s", "sl_rv_30s"])
        flow = _existing(store, ["sl_taker_imbalance_30s", "sl_signed_dollar_flow_30s", "sl_mirror_relative_volume_30s"])
        book = _existing(store, ["sl_spread_bps", "sl_bid_ask_qty_imbalance", "sl_microprice_premium"])
        features: dict[str, pd.Series] = {}
        for left in returns:
            for right in realized_vol + flow + book:
                features[f"sl_interaction__{left}__x__{right}"] = store[left] * store[right]
                features[f"sl_interaction__{left}__div__{right}"] = _safe_divide(store[left], store[right])
        for left in flow:
            for right in book:
                features[f"sl_interaction__{left}__x__{right}"] = store[left] * store[right]
        return pd.DataFrame(features, index=store.index)


class SecondLevelLaggedPack(SecondLevelFeaturePack):
    name = "second_level_lagged"

    def transform(self, store: pd.DataFrame, profile: SecondLevelFeatureProfile) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        for name in profile.lagged_feature_names:
            if name not in store.columns:
                continue
            for lag in profile.lagged_feature_lags:
                features[f"{name}_lag{lag}s"] = store[name].shift(lag)
        return pd.DataFrame(features, index=store.index)


SECOND_LEVEL_FEATURE_PACKS: dict[str, SecondLevelFeaturePack] = {
    pack.name: pack
    for pack in [
        SecondLevelMomentumPack(),
        SecondLevelVolatilityPack(),
        SecondLevelVolumePack(),
        SecondLevelCandleStructurePack(),
        SecondLevelPathStructurePack(),
        SecondLevelTradeMicrostructurePack(),
        SecondLevelBookMicrostructurePack(),
        SecondLevelDepthPack(),
        SecondLevelCrossMarketPack(),
        SecondLevelInteractionBankPack(),
        SecondLevelLaggedPack(),
    ]
}


def get_second_level_feature_pack(name: str) -> SecondLevelFeaturePack:
    try:
        return SECOND_LEVEL_FEATURE_PACKS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown second-level feature pack '{name}'.") from exc


def build_second_level_pack_features(
    store: pd.DataFrame,
    profile: SecondLevelFeatureProfile,
) -> pd.DataFrame:
    output = store.copy()
    for pack_name in profile.packs:
        pack = get_second_level_feature_pack(pack_name)
        additions = pack.transform(output, profile).replace([np.inf, -np.inf], np.nan)
        if additions.empty:
            continue
        duplicate_columns = [column for column in additions.columns if column in output.columns]
        if duplicate_columns:
            additions = additions.drop(columns=duplicate_columns)
        output = pd.concat([output, additions], axis=1).copy()
    return output
