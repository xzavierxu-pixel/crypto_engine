from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.core.constants import DEFAULT_SETTINGS_PATH


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    timezone: str


@dataclass(frozen=True)
class MarketConfig:
    exchange: str
    pair: str
    timeframe: str


@dataclass(frozen=True)
class HorizonSpecConfig:
    minutes: int
    grid_minutes: int
    label_builder: str
    feature_profile: str
    signal_policy: str | None = None
    sizing_plugin: str | None = None
    label_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HorizonsConfig:
    active: list[str]
    specs: dict[str, HorizonSpecConfig]

    def get_active_spec(self, name: str | None = None) -> HorizonSpecConfig:
        target = name or self.active[0]
        try:
            return self.specs[target]
        except KeyError as exc:
            raise KeyError(f"Unknown horizon '{target}'.") from exc


@dataclass(frozen=True)
class DatasetConfig:
    train_start: str
    train_end: str
    validation_window_days: int
    strict_grid_only: bool
    drop_incomplete_candles: bool
    sample_quality_filter: dict[str, Any] = field(default_factory=dict)
    sample_weighting: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureProfileConfig:
    packs: list[str]
    momentum_windows: list[int] = field(default_factory=list)
    vol_windows: list[int] = field(default_factory=list)
    volume_windows: list[int] = field(default_factory=list)
    market_quality_windows: list[int] = field(default_factory=list)
    slope_windows: list[int] = field(default_factory=list)
    range_windows: list[int] = field(default_factory=list)
    htf_context_timeframes: list[int] = field(default_factory=list)
    htf_context_vol_window: int = 5
    htf_context_range_window: int = 5
    htf_context_zscore_window: int = 5
    htf_context_efficiency_window: int = 5
    htf_context_trend_strength_window: int = 5
    lagged_feature_names: list[str] = field(default_factory=list)
    lagged_feature_lags: list[int] = field(default_factory=list)
    compression_window: int = 20
    compression_rank_window: int = 100
    compression_atr_short_window: int = 5
    compression_atr_long_window: int = 20
    compression_nr_windows: list[int] = field(default_factory=list)
    asymmetry_rv_windows: list[int] = field(default_factory=list)
    asymmetry_skew_windows: list[int] = field(default_factory=list)
    asymmetry_imbalance_windows: list[int] = field(default_factory=list)
    flow_volume_window: int = 3
    use_vwap_distance: bool = False
    use_regime_features: bool = False
    use_time_features: bool = False


@dataclass(frozen=True)
class FeaturesConfig:
    profiles: dict[str, FeatureProfileConfig]

    def get_profile(self, name: str) -> FeatureProfileConfig:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise KeyError(f"Unknown feature profile '{name}'.") from exc


@dataclass(frozen=True)
class PluginGroupConfig:
    active_plugin: str
    plugins: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalConfig:
    policies: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str
    active_adapter: str
    active_mapper: str
    safeguards: dict[str, Any] = field(default_factory=dict)
    polymarket: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathsConfig:
    artifacts_dir: str
    model_dir: str
    logs_dir: str


@dataclass(frozen=True)
class Settings:
    project: ProjectConfig
    market: MarketConfig
    horizons: HorizonsConfig
    dataset: DatasetConfig
    features: FeaturesConfig
    model: PluginGroupConfig
    calibration: PluginGroupConfig
    signal: SignalConfig
    sizing: PluginGroupConfig
    execution: ExecutionConfig
    paths: PathsConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Settings":
        horizons = HorizonsConfig(
            active=list(payload["horizons"]["active"]),
            specs={
                name: HorizonSpecConfig(**spec_payload)
                for name, spec_payload in payload["horizons"]["specs"].items()
            },
        )
        features = FeaturesConfig(
            profiles={
                name: FeatureProfileConfig(**profile_payload)
                for name, profile_payload in payload["features"]["profiles"].items()
            }
        )
        return cls(
            project=ProjectConfig(**payload["project"]),
            market=MarketConfig(**payload["market"]),
            horizons=horizons,
            dataset=DatasetConfig(**payload["dataset"]),
            features=features,
            model=PluginGroupConfig(**payload["model"]),
            calibration=PluginGroupConfig(**payload["calibration"]),
            signal=SignalConfig(**payload["signal"]),
            sizing=PluginGroupConfig(**payload["sizing"]),
            execution=ExecutionConfig(**payload["execution"]),
            paths=PathsConfig(**payload["paths"]),
        )


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return Settings.from_dict(payload)
