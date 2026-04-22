from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import FeatureProfileConfig, Settings
from src.features.base import FeaturePack


class MomentumAccelerationFeaturePack(FeaturePack):
    name = "momentum_acceleration"

    def transform(
        self,
        df: pd.DataFrame,
        settings: Settings,
        profile: FeatureProfileConfig,
    ) -> pd.DataFrame:
        required = ["ret_1", "ret_3"]
        missing = [name for name in required if name not in df.columns]
        if missing:
            raise ValueError(
                "Momentum acceleration pack requires momentum features to be present first. "
                f"Missing columns: {missing}"
            )

        features = pd.DataFrame(index=df.index)
        features["ret_1_accel"] = df["ret_1"] - df["ret_1"].shift(1)
        features["ret_3_accel"] = df["ret_3"] - df["ret_3"].shift(3)
        features["momentum_reversal"] = np.sign(df["ret_1"]) * np.sign(df["ret_3"])
        return features
