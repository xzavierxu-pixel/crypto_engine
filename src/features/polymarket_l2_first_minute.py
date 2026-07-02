from __future__ import annotations

import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.data.polymarket_l2 import L2_FEATURE_PREFIX, assert_feature_schema_safe
from src.features.base import FeaturePack


class PolymarketL2FirstMinuteFeaturePack(FeaturePack):
    """Expose only precomputed, cutoff-safe L2 columns to the shared builder."""

    name = "polymarket_l2_first_minute_v1"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        columns = [column for column in df.columns if column.startswith(L2_FEATURE_PREFIX)]
        assert_feature_schema_safe(columns)
        if not columns:
            return pd.DataFrame(index=df.index)
        return df[columns].replace([float("inf"), float("-inf")], pd.NA)
