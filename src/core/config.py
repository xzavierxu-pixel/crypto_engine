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
class ThresholdSearchConfig:
    enabled: bool = True
    t_up_min: float = 0.50
    t_up_max: float = 0.60
    t_down_min: float = 0.40
    t_down_max: float = 0.50
    step: float = 0.005
    enforce_min_side_share: bool = False
    min_side_share: float = 0.20
    min_up_signals: int = 50
    min_down_signals: int = 50
    min_total_signals: int = 150
    stage1_coverage_min: float = 0.60
    stage1_coverage_max: float = 0.80
    min_active_samples: int = 25
    min_end_to_end_coverage: float = 0.30


@dataclass(frozen=True)
class ObjectiveConfig:
    label: str = "settlement_direction"
    optimize_metric: str = "balanced_precision"
    min_coverage: float = 0.60
    tie_breaker_metric: str = "coverage"
    balanced_precision_tie_tolerance: float = 0.002


@dataclass(frozen=True)
class SampleWeightingConfig:
    enabled: bool = True
    mode: str = "linear_ramp"
    min_abs_return: float = 0.0001
    full_weight_abs_return: float = 0.0005
    min_weight: float = 0.20
    max_weight: float = 1.00


@dataclass(frozen=True)
class ValidationConfig:
    mode: str = "chronological_validation"
    train_days: int = 30
    validation_days: int = 30
    report_worst_fold: bool = False


@dataclass(frozen=True)
class ReportingConfig:
    include_precision_coverage_frontier: bool = True
    include_boundary_slices: bool = True
    include_regime_slices: bool = True
    include_calibration_metrics: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    train_start: str
    train_end: str
    validation_window_days: int = 30
    train_window_days: int = 30
    strict_grid_only: bool = True
    drop_incomplete_candles: bool = True
    walk_forward: dict[str, Any] = field(default_factory=dict)
    threshold_search: ThresholdSearchConfig = field(default_factory=ThresholdSearchConfig)
    sample_quality_filter: dict[str, Any] = field(default_factory=dict)


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
class TwoStageLabelsConfig:
    active_return_threshold: float = 0.0002


@dataclass(frozen=True)
class LabelsConfig:
    two_stage: TwoStageLabelsConfig = field(default_factory=TwoStageLabelsConfig)


@dataclass(frozen=True)
class DerivativesFundingConfig:
    enabled: bool = False
    path: str | None = None
    archive_path: str | None = None
    source: str | None = None
    ffill_until_next: bool = True
    zscore_window: int = 720


@dataclass(frozen=True)
class DerivativesBasisConfig:
    enabled: bool = False
    path: str | None = None
    archive_path: str | None = None
    source: str | None = None
    use_mark_price: bool = True
    use_index_price: bool = True
    use_premium_index: bool = True
    zscore_window: int = 720


@dataclass(frozen=True)
class DerivativesOIConfig:
    enabled: bool = False
    path: str | None = None
    archive_path: str | None = None
    frequency: str | None = None
    zscore_window: int = 288
    change_windows: list[int] = field(default_factory=lambda: [5, 60])
    slope_window: int = 5


@dataclass(frozen=True)
class DerivativesOptionsConfig:
    enabled: bool = False
    path: str | None = None
    archive_path: str | None = None
    source: str | None = None
    zscore_window: int = 288
    change_window: int = 60
    regime_zscore_threshold: float = 1.0


@dataclass(frozen=True)
class DerivativesBookTickerConfig:
    enabled: bool = False
    path: str | None = None
    archive_path: str | None = None
    source: str | None = None
    zscore_window: int = 288


@dataclass(frozen=True)
class DerivativesConfig:
    enabled: bool = False
    exchange: str = ""
    symbol_spot: str = ""
    symbol_perp: str = ""
    path_mode: str = "latest"
    funding: DerivativesFundingConfig = field(default_factory=DerivativesFundingConfig)
    basis: DerivativesBasisConfig = field(default_factory=DerivativesBasisConfig)
    oi: DerivativesOIConfig = field(default_factory=DerivativesOIConfig)
    options: DerivativesOptionsConfig = field(default_factory=DerivativesOptionsConfig)
    book_ticker: DerivativesBookTickerConfig = field(default_factory=DerivativesBookTickerConfig)


@dataclass(frozen=True)
class SecondLevelFeatureStoreConfig:
    enabled: bool = True
    data_root: str = "./artifacts/data_v2"
    feature_store_version: str = "second_level_v2"
    feature_profile: str = "expanded_v2"
    feature_store_path: str | None = "./artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT/second_features.parquet"
    artifact_root: str = "./artifacts/data_v2/second_level"
    market: str = "BTCUSDT"
    exchange: str = "binance"
    large_trade_quantile: float = 0.95
    large_trade_window_seconds: int = 300
    optimize_metric: str = "balanced_precision"
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_profile_payload(self) -> dict[str, Any]:
        if not self.profiles:
            return {}
        try:
            return self.profiles[self.feature_profile]
        except KeyError as exc:
            raise KeyError(f"Unknown second-level feature profile '{self.feature_profile}'.") from exc


@dataclass(frozen=True)
class DataBackfillMarketConfig:
    enabled: bool = False
    symbols: list[str] = field(default_factory=list)
    data_types: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class DataBackfillOptionConfig:
    enabled: bool = False
    symbols: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class DataBackfillConfig:
    provider: str = ""
    start_date: str = ""
    use_monthly_for_full_months: bool = True
    use_daily_for_open_month_tail: bool = True
    use_daily_for_full_month_fallback: bool = False
    verify_checksum: bool = True
    spot: DataBackfillMarketConfig = field(default_factory=DataBackfillMarketConfig)
    futures_um: DataBackfillMarketConfig = field(default_factory=DataBackfillMarketConfig)
    futures_cm: DataBackfillMarketConfig = field(default_factory=DataBackfillMarketConfig)
    option: DataBackfillOptionConfig = field(default_factory=DataBackfillOptionConfig)


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
    active_plugin: str | None = None
    active_plugins: dict[str, str] = field(default_factory=dict)
    plugins: dict[str, dict[str, Any]] = field(default_factory=dict)
    stage2_class_weight: str | dict[int, float] | None = None

    def resolve_plugin(self, stage: str | None = None) -> str:
        if stage is not None and stage in self.active_plugins:
            return self.active_plugins[stage]
        if self.active_plugin is None:
            raise KeyError(f"No active plugin configured for stage '{stage}'.")
        return self.active_plugin


@dataclass(frozen=True)
class TwoStagePolicyConfig:
    stage1_threshold: float | None = None
    up_threshold: float | None = None
    down_threshold: float | None = None
    margin_threshold: float | None = None


@dataclass(frozen=True)
class SelectiveBinaryPolicyConfig:
    t_up: float | None = None
    t_down: float | None = None


@dataclass(frozen=True)
class SignalConfig:
    policies: dict[str, dict[str, Any]]

    def get_two_stage_policy(self, name: str) -> TwoStagePolicyConfig:
        try:
            payload = self.policies[name]
        except KeyError as exc:
            raise KeyError(f"Unknown signal policy '{name}'.") from exc
        return TwoStagePolicyConfig(**payload)

    def get_selective_binary_policy(self, name: str) -> SelectiveBinaryPolicyConfig:
        try:
            payload = self.policies[name]
        except KeyError as exc:
            raise KeyError(f"Unknown signal policy '{name}'.") from exc
        return SelectiveBinaryPolicyConfig(**payload)


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str
    active_adapter: str
    active_mapper: str
    safeguards: dict[str, Any] = field(default_factory=dict)
    polymarket: dict[str, Any] = field(default_factory=dict)
    fixed_contract_size: float = 5.0


@dataclass(frozen=True)
class PathsConfig:
    artifacts_dir: str
    model_dir: str
    logs_dir: str


@dataclass(frozen=True)
class Settings:
    project: ProjectConfig
    market: MarketConfig
    objective: ObjectiveConfig
    horizons: HorizonsConfig
    dataset: DatasetConfig
    sample_weighting: SampleWeightingConfig
    threshold_search: ThresholdSearchConfig
    validation: ValidationConfig
    features: FeaturesConfig
    labels: LabelsConfig
    derivatives: DerivativesConfig
    second_level: SecondLevelFeatureStoreConfig
    data_backfill: DataBackfillConfig
    model: PluginGroupConfig
    calibration: PluginGroupConfig
    signal: SignalConfig
    sizing: PluginGroupConfig
    execution: ExecutionConfig
    paths: PathsConfig
    reporting: ReportingConfig

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
        derivatives_payload = payload.get("derivatives", {})
        second_level_payload = payload.get("second_level", {})
        data_backfill_payload = payload.get("data_backfill", {})
        return cls(
            project=ProjectConfig(**payload["project"]),
            market=MarketConfig(**payload["market"]),
            objective=ObjectiveConfig(**payload.get("objective", {})),
            horizons=horizons,
            dataset=DatasetConfig(
                train_start=payload["dataset"]["train_start"],
                train_end=payload["dataset"]["train_end"],
                validation_window_days=payload["dataset"].get("validation_window_days", 30),
                train_window_days=payload["dataset"].get("train_window_days", 30),
                strict_grid_only=payload["dataset"].get("strict_grid_only", True),
                drop_incomplete_candles=payload["dataset"].get("drop_incomplete_candles", True),
                walk_forward=payload["dataset"].get("walk_forward", {}),
                threshold_search=ThresholdSearchConfig(**payload["dataset"].get("threshold_search", {})),
                sample_quality_filter=payload["dataset"].get("sample_quality_filter", {}),
            ),
            sample_weighting=SampleWeightingConfig(**payload.get("sample_weighting", {})),
            threshold_search=ThresholdSearchConfig(**payload.get("threshold_search", {})),
            validation=ValidationConfig(**payload.get("validation", {})),
            features=features,
            labels=LabelsConfig(
                two_stage=TwoStageLabelsConfig(**payload.get("labels", {}).get("two_stage", {}))
            ),
            derivatives=DerivativesConfig(
                enabled=derivatives_payload.get("enabled", False),
                exchange=derivatives_payload.get("exchange", ""),
                symbol_spot=derivatives_payload.get("symbol_spot", ""),
                symbol_perp=derivatives_payload.get("symbol_perp", ""),
                path_mode=derivatives_payload.get("path_mode", "latest"),
                funding=DerivativesFundingConfig(**derivatives_payload.get("funding", {})),
                basis=DerivativesBasisConfig(**derivatives_payload.get("basis", {})),
                oi=DerivativesOIConfig(**derivatives_payload.get("oi", {})),
                options=DerivativesOptionsConfig(**derivatives_payload.get("options", {})),
                book_ticker=DerivativesBookTickerConfig(**derivatives_payload.get("book_ticker", {})),
            ),
            second_level=SecondLevelFeatureStoreConfig(**second_level_payload),
            data_backfill=DataBackfillConfig(
                provider=data_backfill_payload.get("provider", ""),
                start_date=data_backfill_payload.get("start_date", ""),
                use_monthly_for_full_months=data_backfill_payload.get("use_monthly_for_full_months", True),
                use_daily_for_open_month_tail=data_backfill_payload.get("use_daily_for_open_month_tail", True),
                use_daily_for_full_month_fallback=data_backfill_payload.get("use_daily_for_full_month_fallback", False),
                verify_checksum=data_backfill_payload.get("verify_checksum", True),
                spot=DataBackfillMarketConfig(**data_backfill_payload.get("spot", {})),
                futures_um=DataBackfillMarketConfig(**data_backfill_payload.get("futures_um", {})),
                futures_cm=DataBackfillMarketConfig(**data_backfill_payload.get("futures_cm", {})),
                option=DataBackfillOptionConfig(**data_backfill_payload.get("option", {})),
            ),
            model=PluginGroupConfig(**payload["model"]),
            calibration=PluginGroupConfig(**payload["calibration"]),
            signal=SignalConfig(**payload["signal"]),
            sizing=PluginGroupConfig(**payload["sizing"]),
            execution=ExecutionConfig(**payload["execution"]),
            paths=PathsConfig(**payload["paths"]),
            reporting=ReportingConfig(**payload.get("reporting", {})),
        )


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return Settings.from_dict(payload)
