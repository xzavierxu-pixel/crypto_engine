from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import Settings, load_settings
from src.core.constants import DEFAULT_SETTINGS_PATH, DEFAULT_TARGET_COLUMN, DEFAULT_TIMESTAMP_COLUMN
from src.horizons.registry import get_horizon_spec
from src.labels.registry import get_label_builder
from src.services.feature_service import FeatureService

try:
    from freqtrade.strategy.interface import IStrategy  # type: ignore
except ImportError:  # pragma: no cover - local development fallback
    class IStrategy:  # type: ignore
        """Fallback base class to keep the module importable without Freqtrade."""


class BTCGridFreqAIStrategy(IStrategy):
    """
    Thin Freqtrade/FreqAI strategy adapter.

    Business logic lives in shared core modules. This class only wires Freqtrade
    lifecycle calls into the shared feature and label builders.
    """

    INTERFACE_VERSION = 3
    timeframe = "1m"
    startup_candle_count = 180
    can_short = False
    process_only_new_candles = True
    use_exit_signal = True
    minimal_roi = {"0": 100.0}
    stoploss = -1.0

    def __init__(
        self,
        config: dict | None = None,
        settings: Settings | None = None,
        horizon_name: str = "5m",
    ) -> None:
        raw_config = config or {}
        self.settings = settings or load_settings(
            raw_config.get("strategy_settings_path", DEFAULT_SETTINGS_PATH)
        )
        self.horizon_name = horizon_name
        self.horizon = get_horizon_spec(self.settings, horizon_name)
        self.feature_service = FeatureService(self.settings)
        self.label_builder = get_label_builder(self.horizon.label_builder)
        self.freqai_signal_config = raw_config.get("freqai_signal", {})
        self.entry_probability_threshold = float(
            self.freqai_signal_config.get("entry_probability_threshold", 0.58)
        )
        self.entry_probability_margin = float(
            self.freqai_signal_config.get("entry_probability_margin", 0.16)
        )
        self.exit_probability_threshold = float(
            self.freqai_signal_config.get("exit_probability_threshold", 0.55)
        )
        self.min_nz_volume_share_20 = float(self.freqai_signal_config.get("min_nz_volume_share_20", 0.2))
        self.max_flat_share_20 = float(self.freqai_signal_config.get("max_flat_share_20", 0.95))
        super().__init__(raw_config or {"candle_type_def": "spot"})

    def feature_engineering_expand_all(
        self,
        dataframe: pd.DataFrame,
        period: int,
        metadata: dict,
        **kwargs,
    ) -> pd.DataFrame:
        return dataframe

    def feature_engineering_expand_basic(
        self,
        dataframe: pd.DataFrame,
        metadata: dict,
        **kwargs,
    ) -> pd.DataFrame:
        return dataframe

    def feature_engineering_standard(
        self,
        dataframe: pd.DataFrame,
        metadata: dict,
        **kwargs,
    ) -> pd.DataFrame:
        return self.feature_service.build_freqai_feature_dataframe(
            dataframe,
            horizon_name=self.horizon_name,
        )

    def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict | None = None, **kwargs) -> pd.DataFrame:
        if "date" not in dataframe.columns and DEFAULT_TIMESTAMP_COLUMN in dataframe.columns:
            dataframe = dataframe.rename(columns={DEFAULT_TIMESTAMP_COLUMN: "date"})
        shared_input = dataframe.rename(columns={"date": DEFAULT_TIMESTAMP_COLUMN})
        labeled = self.label_builder.build(
            shared_input,
            self.settings,
            self.horizon,
            select_grid_only=False,
        ).rename(columns={DEFAULT_TIMESTAMP_COLUMN: "date"})
        merged = dataframe.merge(
            labeled[["date", DEFAULT_TARGET_COLUMN, "is_grid_t0"]],
            on="date",
            how="left",
            validate="one_to_one",
        )

        if hasattr(self, "freqai"):
            self.freqai.class_names = ["down", "up"]

        merged["&s-up_or_down"] = np.where(
            merged[DEFAULT_TARGET_COLUMN] == 1.0,
            "up",
            np.where(merged[DEFAULT_TARGET_COLUMN] == 0.0, "down", None),
        )
        merged["grid_dir_target"] = merged[DEFAULT_TARGET_COLUMN]
        return merged

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict | None = None) -> pd.DataFrame:
        if "date" not in dataframe.columns and DEFAULT_TIMESTAMP_COLUMN in dataframe.columns:
            dataframe = dataframe.rename(columns={DEFAULT_TIMESTAMP_COLUMN: "date"})
        if hasattr(self, "freqai") and getattr(self, "freqai", None) is not None:
            return self.freqai.start(dataframe, metadata or {}, self)
        dataframe = self.feature_engineering_standard(dataframe, metadata or {})
        return self.set_freqai_targets(dataframe, metadata or {})

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = None
        if {"do_predict", "is_grid_t0"}.issubset(dataframe.columns):
            probability_ready = {"up", "down"}.issubset(dataframe.columns)
            activity_ready = {"%-nz_volume_share_20", "%-flat_share_20"}.issubset(dataframe.columns)
            conditions = [
                dataframe["do_predict"] == 1,
                dataframe["is_grid_t0"] == True,
            ]
            if probability_ready:
                conditions.extend(
                    [
                        dataframe["up"] >= self.entry_probability_threshold,
                        (dataframe["up"] - dataframe["down"]) >= self.entry_probability_margin,
                    ]
                )
            else:
                conditions.append(dataframe["&s-up_or_down"] == "up")

            if activity_ready:
                conditions.extend(
                    [
                        dataframe["%-nz_volume_share_20"] >= self.min_nz_volume_share_20,
                        dataframe["%-flat_share_20"] <= self.max_flat_share_20,
                    ]
                )

            dataframe.loc[
                np.logical_and.reduce(conditions),
                ["enter_long", "enter_tag"],
            ] = (1, "freqai_up_5m_prob")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        if {"do_predict", "is_grid_t0"}.issubset(dataframe.columns):
            probability_ready = {"down"}.issubset(dataframe.columns)
            exit_condition = (
                (dataframe["do_predict"] == 1)
                & (dataframe["is_grid_t0"] == True)
                & (
                    dataframe["down"] >= self.exit_probability_threshold
                    if probability_ready
                    else dataframe["&s-up_or_down"] == "down"
                )
            )
            dataframe.loc[
                exit_condition,
                "exit_long",
            ] = 1
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        open_time = getattr(trade, "open_date_utc", None) or getattr(trade, "open_date", None)
        if open_time is None:
            return None

        if current_time >= open_time + timedelta(minutes=self.horizon.minutes):
            return "horizon_timeout"

        return None
