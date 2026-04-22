from __future__ import annotations

import pandas as pd

from src.core.config import Settings
from src.core.constants import CORE_LABEL_VERSION, DEFAULT_TARGET_COLUMN, DEFAULT_TIMESTAMP_COLUMN
from src.core.timegrid import add_grid_columns, select_grid_rows
from src.core.validation import normalize_ohlcv_frame
from src.horizons.base import HorizonSpec
from src.labels.base import LabelBuilder


class GridDirectionLabelBuilder(LabelBuilder):
    name = "grid_direction"

    def build(
        self,
        df: pd.DataFrame,
        settings: Settings,
        horizon: HorizonSpec,
        select_grid_only: bool | None = None,
    ) -> pd.DataFrame:
        normalized = normalize_ohlcv_frame(df, timestamp_column=DEFAULT_TIMESTAMP_COLUMN, require_volume=False)
        labeled = add_grid_columns(normalized, grid_minutes=horizon.grid_minutes)
        future_close = labeled["close"].shift(-horizon.future_close_offset)
        label_params = horizon.label_params or {}
        label_version = str(label_params.get("label_version", CORE_LABEL_VERSION))
        target = (future_close > labeled["open"]).astype("float64")
        target[future_close.isna()] = pd.NA
        target[~labeled["is_grid_t0"]] = pd.NA

        labeled[DEFAULT_TARGET_COLUMN] = target
        labeled["asset"] = settings.market.pair
        labeled["horizon"] = horizon.name
        labeled["label_version"] = label_version

        if select_grid_only is None:
            select_grid_only = settings.dataset.strict_grid_only

        if select_grid_only:
            labeled = select_grid_rows(labeled, grid_minutes=horizon.grid_minutes)

        return labeled.reset_index(drop=True)
