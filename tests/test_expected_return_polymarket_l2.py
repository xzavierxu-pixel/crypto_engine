from __future__ import annotations

import pandas as pd
import pytest

from price_estimator.expected_return.expected_return_common import (
    attach_future_low_target,
    derive_selected_side_l2_features,
)


def test_selected_side_features_require_calibrated_probability() -> None:
    frame = pd.DataFrame(
        {
            "selected_side": ["UP", "DOWN"],
            "calibrated_p_up": [0.7, 0.3],
            "pm_l2_1m_up_mid": [0.6, 0.4],
            "pm_l2_1m_down_mid": [0.4, 0.6],
            "pm_l2_1m_up_last_trade": [0.59, 0.39],
            "pm_l2_1m_down_last_trade": [0.41, 0.61],
        }
    )
    result = derive_selected_side_l2_features(frame)
    assert result["p_side"].tolist() == pytest.approx([0.7, 0.7])
    assert result["pm_l2_selected_mid"].tolist() == pytest.approx([0.6, 0.6])
    assert result["pm_l2_selected_p_side_minus_mid"].tolist() == pytest.approx([0.1, 0.1])
    with pytest.raises(ValueError, match="calibrated_p_up"):
        derive_selected_side_l2_features(frame.drop(columns="calibrated_p_up"))


def test_future_low_is_joined_only_as_selected_side_target() -> None:
    frame = pd.DataFrame(
        {
            "market_t0": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "selected_side": ["UP", "DOWN"],
        }
    )
    labels = pd.DataFrame(
        {
            "market_t0": frame["market_t0"],
            "up_future_low_4m": [0.2, 0.3],
            "down_future_low_4m": [0.8, 0.7],
        }
    )
    result = attach_future_low_target(frame, labels)
    assert result["future_low_4m"].tolist() == pytest.approx([0.2, 0.7])
