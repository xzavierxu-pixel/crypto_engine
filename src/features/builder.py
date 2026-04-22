from __future__ import annotations

import pandas as pd

from src.core.config import Settings
from src.core.constants import CORE_FEATURE_VERSION, DEFAULT_TIMESTAMP_COLUMN
from src.core.timegrid import add_grid_columns, select_grid_rows
from src.core.validation import normalize_ohlcv_frame
from src.data.derivatives.feature_store import DerivativesFeatureStore
from src.features.registry import get_feature_pack
from src.horizons.registry import get_horizon_spec


DERIVATIVES_HELPER_COLUMN_NAMES = {
    "exchange",
    "symbol",
    "funding_effective_time",
    "funding_interval_hours",
    "market_family",
    "data_type",
    "interval",
    "source_file",
    "source_date",
    "source_granularity",
    "source_version",
    "funding_source_version",
    "basis_source_version",
    "oi_source_version",
    "options_source_version",
    "book_ticker_source_version",
    "checksum_status",
    "ingested_at",
    "download_status",
    "expected_checksum",
    "actual_checksum",
    "calc_time",
    "create_time",
    "base_asset",
    "quote_asset",
}
DERIVATIVES_HELPER_SUFFIXES = ("_oi", "_options", "_book_ticker", "_funding", "_basis")


def _is_derivatives_helper_column(column: str) -> bool:
    if column.startswith("raw_") or column.startswith("derivatives_"):
        return True
    if column in DERIVATIVES_HELPER_COLUMN_NAMES:
        return True
    for suffix in DERIVATIVES_HELPER_SUFFIXES:
        if column.endswith(suffix) and column[: -len(suffix)] in DERIVATIVES_HELPER_COLUMN_NAMES:
            return True
    return False


def build_feature_frame(
    df: pd.DataFrame,
    settings: Settings,
    horizon_name: str | None = None,
    select_grid_only: bool | None = None,
    derivatives_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    normalized = normalize_ohlcv_frame(df, timestamp_column=DEFAULT_TIMESTAMP_COLUMN, require_volume=False)
    horizon = get_horizon_spec(settings, horizon_name)
    profile = settings.features.get_profile(horizon.feature_profile)

    if settings.derivatives.enabled:
        feature_frame = DerivativesFeatureStore(settings).attach_to_spot(
            normalized,
            derivatives_frame=derivatives_frame,
        )
    else:
        feature_frame = normalized.copy()

    for pack_name in profile.packs:
        pack = get_feature_pack(pack_name)
        feature_values = pack.transform(feature_frame, settings, profile)
        feature_frame = pd.concat([feature_frame, feature_values], axis=1)

    helper_columns = [column for column in feature_frame.columns if _is_derivatives_helper_column(column)]
    if helper_columns:
        feature_frame = feature_frame.drop(columns=helper_columns)

    feature_frame = add_grid_columns(feature_frame, grid_minutes=horizon.grid_minutes)
    feature_frame["asset"] = settings.market.pair
    feature_frame["horizon"] = horizon.name
    feature_frame["feature_version"] = CORE_FEATURE_VERSION

    if select_grid_only is None:
        select_grid_only = settings.dataset.strict_grid_only

    if select_grid_only:
        feature_frame = select_grid_rows(feature_frame, grid_minutes=horizon.grid_minutes)

    return feature_frame.reset_index(drop=True)
