from __future__ import annotations

import math

import pandas as pd

from src.core.config import load_settings
from src.features.builder import build_feature_frame


def test_feature_builder_uses_only_information_before_t0() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T12:00:00Z", periods=240, freq="1min"),
            "open": [100 + index for index in range(240)],
            "high": [101 + index for index in range(240)],
            "low": [99 + index for index in range(240)],
            "close": [100 + index for index in range(240)],
            "volume": [10 + index for index in range(240)],
        }
    )

    feature_frame = build_feature_frame(frame, settings, horizon_name="5m", select_grid_only=False)
    row = feature_frame.loc[feature_frame["timestamp"] == pd.Timestamp("2024-01-01T12:05:00Z")].iloc[0]

    expected_ret_1 = (104 / 103) - 1
    current_bar_ret_1 = (105 / 104) - 1

    assert math.isclose(row["ret_1"], expected_ret_1, rel_tol=1e-9)
    assert not math.isclose(row["ret_1"], current_bar_ret_1, rel_tol=1e-9)
    assert math.isclose(row["minute_sin"], math.sin(2 * math.pi * 5 / 60.0), rel_tol=1e-9)
    assert math.isclose(row["minute_cos"], math.cos(2 * math.pi * 5 / 60.0), rel_tol=1e-9)
    assert math.isclose(row["weekday_sin"], 0.0, abs_tol=1e-12)
    assert math.isclose(row["weekday_cos"], 1.0, rel_tol=1e-9)
    assert "minute_bucket" not in feature_frame.columns
    assert "rv_3" in feature_frame.columns
    assert "range_5" in feature_frame.columns
    assert "regime_vol_ratio" in feature_frame.columns
    assert "relative_volume_5" in feature_frame.columns
    assert "body_pct_1" in feature_frame.columns
    assert "low_volume_flag_share_20" in feature_frame.columns
    assert "volume_below_rolling_p20_share_20" in feature_frame.columns
    assert "stale_trade_share_20" in feature_frame.columns
    assert "flat_share_20" in feature_frame.columns
    assert "abs_ret_mean_20" in feature_frame.columns
    assert "dollar_vol_mean_20" in feature_frame.columns
    assert "htf_ret_15m_1" in feature_frame.columns
    assert "bb_width_20" in feature_frame.columns
    assert "upside_rv_5" in feature_frame.columns
    assert "clv_1" in feature_frame.columns
    assert "ret_1_lag1" in feature_frame.columns
    assert "compression_score_lag12" in feature_frame.columns
    assert "ret_vol_product__ret_5__rv_10" in feature_frame.columns
    assert "relative_volume_20_rolling_z_24" in feature_frame.columns
    assert "ret_1_accel" in feature_frame.columns
    assert "momentum_reversal" in feature_frame.columns
    assert "last_1m_up" in feature_frame.columns
    assert "up_ratio_5" in feature_frame.columns
    assert feature_frame["feature_version"].eq("v7").all()
