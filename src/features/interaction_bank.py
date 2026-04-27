from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


def _existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _safe_ratio(left: pd.Series, right: pd.Series) -> pd.Series:
    return left / right.replace(0, np.nan)


def _target(series: pd.Series, target_index: pd.Index | None) -> pd.Series:
    if target_index is None:
        return series
    return series.loc[target_index]


def _left_right(
    left: pd.Series,
    right: pd.Series,
    target_index: pd.Index | None,
) -> tuple[pd.Series, pd.Series]:
    if target_index is None:
        return left, right
    return left.loc[target_index], right.loc[target_index]


class InteractionBankFeaturePack(FeaturePack):
    name = "interaction_bank"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        feature_map: dict[str, pd.Series] = {}
        raw_target_index = df.attrs.get("target_feature_index")
        target_index = pd.Index(raw_target_index) if raw_target_index is not None else None

        returns = _existing(df, ["ret_1", "ret_3", "ret_5", "ret_10", "ret_15"])
        realized_vol = _existing(df, ["rv_3", "rv_5", "rv_10", "rv_30"])
        relative_volume = _existing(df, ["relative_volume_3", "relative_volume_5", "relative_volume_10", "relative_volume_20"])
        volume_z = _existing(df, ["volume_z_3", "volume_z_5", "volume_z_10", "volume_z_20"])
        volume_share = _existing(df, ["volume_share_3", "volume_share_5", "volume_share_10", "volume_share_20"])
        range_pos = _existing(df, ["range_pos_3", "range_pos_5", "range_pos_10"])
        vwap_dist = _existing(df, ["vwap_dist_3", "vwap_dist_5", "vwap_dist_10"])
        close_z = _existing(df, ["close_z_3", "close_z_5", "close_z_10"])
        efficiency = _existing(df, ["efficiency_3", "efficiency_5", "efficiency_10"])
        market_state = _existing(
            df,
            [
                "low_volume_flag_share_5",
                "low_volume_flag_share_20",
                "volume_below_rolling_p20_share_5",
                "volume_below_rolling_p20_share_20",
                "stale_trade_share_5",
                "stale_trade_share_20",
                "flat_share_5",
                "flat_share_20",
                "abs_ret_mean_5",
                "abs_ret_mean_20",
                "dollar_vol_mean_5",
                "dollar_vol_mean_20",
            ],
        )
        context = _existing(
            df,
            [
                "compression_score",
                "clv_1",
                "wick_pressure_1",
                "up_ratio_5",
                "down_ratio_5",
                "max_up_ret_5",
                "max_down_ret_5",
                "htf_ret_15m_1",
                "htf_rv_15m",
                "htf_range_pos_15m",
                "htf_close_z_15m",
                "htf_efficiency_15m",
                "htf_regime_trend_strength_15m",
                "htf_ret_60m_1",
                "htf_rv_60m",
                "htf_range_pos_60m",
                "htf_close_z_60m",
                "htf_efficiency_60m",
                "htf_regime_trend_strength_60m",
            ],
        )

        pairwise_groups = [
            ("ret_term", returns),
            ("rv_term", realized_vol),
            ("relative_volume_term", relative_volume),
            ("volume_z_term", volume_z),
            ("volume_share_term", volume_share),
            ("range_pos_term", range_pos),
            ("vwap_dist_term", vwap_dist),
            ("close_z_term", close_z),
            ("efficiency_term", efficiency),
        ]
        for prefix, columns in pairwise_groups:
            for left_index in range(len(columns)):
                for right_index in range(left_index + 1, len(columns)):
                    left_name = columns[left_index]
                    right_name = columns[right_index]
                    left = df[left_name]
                    right = df[right_name]
                    left_target, right_target = _left_right(left, right, target_index)
                    feature_map[f"{prefix}_spread__{left_name}__{right_name}"] = left_target - right_target
                    feature_map[f"{prefix}_ratio__{left_name}__{right_name}"] = _safe_ratio(
                        left_target,
                        right_target,
                    )

        for left_name in returns:
            left = df[left_name]
            for right_name in realized_vol:
                right = df[right_name]
                left_target, right_target = _left_right(left, right, target_index)
                feature_map[f"ret_vol_product__{left_name}__{right_name}"] = left_target * right_target
                feature_map[f"ret_vol_ratio__{left_name}__{right_name}"] = _safe_ratio(left_target, right_target)
            for right_name in relative_volume:
                left_target, right_target = _left_right(left, df[right_name], target_index)
                feature_map[f"ret_relative_volume_product__{left_name}__{right_name}"] = left_target * right_target
            for right_name in volume_z:
                left_target, right_target = _left_right(left, df[right_name], target_index)
                feature_map[f"ret_volume_z_product__{left_name}__{right_name}"] = left_target * right_target
            for right_name in range_pos:
                left_target, right_target = _left_right(left, df[right_name], target_index)
                feature_map[f"ret_range_pos_product__{left_name}__{right_name}"] = left_target * right_target
            for right_name in close_z:
                left_target, right_target = _left_right(left, df[right_name], target_index)
                feature_map[f"ret_close_z_product__{left_name}__{right_name}"] = left_target * right_target
            for right_name in efficiency:
                left_target, right_target = _left_right(left, df[right_name], target_index)
                feature_map[f"ret_efficiency_product__{left_name}__{right_name}"] = left_target * right_target

        for left_name in volume_z:
            for right_name in relative_volume:
                left_target, right_target = _left_right(df[left_name], df[right_name], target_index)
                feature_map[f"volume_state_product__{left_name}__{right_name}"] = left_target * right_target

        for left_name in range_pos:
            for right_name in close_z:
                left_target, right_target = _left_right(df[left_name], df[right_name], target_index)
                feature_map[f"price_location_product__{left_name}__{right_name}"] = left_target * right_target

        scalar_context = _existing(
            df,
            [
                "compression_score",
                "clv_1",
                "wick_pressure_1",
                "up_ratio_5",
                "down_ratio_5",
                "max_up_ret_5",
                "max_down_ret_5",
            ],
        )
        for left_name in scalar_context:
            left = df[left_name]
            for right_name in returns:
                left_target, right_target = _left_right(left, df[right_name], target_index)
                feature_map[f"context_ret_product__{left_name}__{right_name}"] = left_target * right_target
            for right_name in relative_volume:
                left_target, right_target = _left_right(left, df[right_name], target_index)
                feature_map[f"context_relative_volume_product__{left_name}__{right_name}"] = (
                    left_target * right_target
                )

        stability_columns = _existing(
            df,
            returns
            + realized_vol
            + relative_volume
            + volume_z
            + range_pos
            + close_z
            + efficiency
            + market_state[:6]
            + scalar_context[:4],
        )
        for column_name in stability_columns:
            series = df[column_name]
            for window in (6, 12, 24):
                rolling_mean = series.rolling(window=window, min_periods=window).mean()
                rolling_std = series.rolling(window=window, min_periods=window).std()
                feature_map[f"{column_name}_rolling_z_{window}"] = (
                    (series - rolling_mean) / rolling_std.replace(0, np.nan)
                ).fillna(0.0).pipe(_target, target_index)
                feature_map[f"{column_name}_mean_gap_{window}"] = _target(series - rolling_mean, target_index)

        transition_columns = _existing(
            df,
            returns
            + realized_vol
            + relative_volume
            + volume_z
            + range_pos
            + close_z
            + efficiency
            + market_state
            + context,
        )
        for column_name in transition_columns:
            series = df[column_name]
            series_target = _target(series, target_index)
            feature_map[f"{column_name}_delta_1"] = series_target - _target(series.shift(1), target_index)
            feature_map[f"{column_name}_delta_3"] = series_target - _target(series.shift(3), target_index)
            feature_map[f"{column_name}_delta_6"] = series_target - _target(series.shift(6), target_index)

        return pd.DataFrame(feature_map, index=target_index if target_index is not None else df.index)
