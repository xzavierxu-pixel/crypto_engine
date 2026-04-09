from __future__ import annotations

import pandas as pd

from src.core.config import Settings
from src.core.constants import CORE_FEATURE_VERSION, DEFAULT_TIMESTAMP_COLUMN
from src.core.timegrid import add_grid_columns, select_grid_rows
from src.core.validation import normalize_ohlcv_frame
from src.features.registry import get_feature_pack
from src.horizons.registry import get_horizon_spec


def build_feature_frame(
    df: pd.DataFrame,
    settings: Settings,
    horizon_name: str | None = None,
    select_grid_only: bool | None = None,
) -> pd.DataFrame:
    normalized = normalize_ohlcv_frame(df, timestamp_column=DEFAULT_TIMESTAMP_COLUMN, require_volume=False)
    horizon = get_horizon_spec(settings, horizon_name)
    profile = settings.features.get_profile(horizon.feature_profile)

    feature_frame = normalized.copy()
    for pack_name in profile.packs:
        pack = get_feature_pack(pack_name)
        feature_values = pack.transform(feature_frame, settings, profile)
        feature_frame = pd.concat([feature_frame, feature_values], axis=1)

    feature_frame = add_grid_columns(feature_frame, grid_minutes=horizon.grid_minutes)
    feature_frame["asset"] = settings.market.pair
    feature_frame["horizon"] = horizon.name
    feature_frame["feature_version"] = CORE_FEATURE_VERSION

    if select_grid_only is None:
        select_grid_only = settings.dataset.strict_grid_only

    if select_grid_only:
        feature_frame = select_grid_rows(feature_frame, grid_minutes=horizon.grid_minutes)

    return feature_frame.reset_index(drop=True)
