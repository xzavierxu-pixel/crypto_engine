from __future__ import annotations

import pandas as pd

from src.core.constants import DEFAULT_ABS_RETURN_COLUMN, DEFAULT_SIGNED_RETURN_COLUMN, DEFAULT_TIMESTAMP_COLUMN
from src.core.validation import normalize_ohlcv_frame
from src.horizons.base import HorizonSpec


def build_abs_return_frame(
    df: pd.DataFrame,
    horizon: HorizonSpec,
) -> pd.DataFrame:
    normalized = normalize_ohlcv_frame(df, timestamp_column=DEFAULT_TIMESTAMP_COLUMN, require_volume=False)
    future_close = normalized["close"].shift(-horizon.future_close_offset)
    signed_return = (future_close - normalized["open"]) / normalized["open"]
    signed_return[future_close.isna()] = pd.NA
    abs_return = signed_return.abs()
    return pd.DataFrame(
        {
            DEFAULT_TIMESTAMP_COLUMN: normalized[DEFAULT_TIMESTAMP_COLUMN],
            DEFAULT_ABS_RETURN_COLUMN: abs_return,
            DEFAULT_SIGNED_RETURN_COLUMN: signed_return,
        }
    )


def compute_stage1_boundary_weight(abs_return: pd.Series, tau: float) -> pd.Series:
    narrow_band = tau * 0.3
    return pd.Series(
        1.0,
        index=abs_return.index,
        dtype="float64",
    ).mask((abs_return - tau).abs() < narrow_band, 0.2)
