from __future__ import annotations

import pandas as pd

from src.core.constants import DEFAULT_STAGE2_TARGET_COLUMN


def build_three_class_direction_target(signed_return: pd.Series, tau: float) -> pd.Series:
    target = pd.Series(1, index=signed_return.index, dtype="int64")
    target = target.mask(signed_return < -tau, 0)
    target = target.mask(signed_return > tau, 2)
    return target.rename(DEFAULT_STAGE2_TARGET_COLUMN)
