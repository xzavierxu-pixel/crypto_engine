from __future__ import annotations

import pandas as pd
import pytest

from src.model.reversal_hybrid import (
    apply_four_bucket_abstain_gate,
    compute_decision_metrics,
    p_follow_from_direction_probability,
    search_four_bucket_gate,
)


def test_p_follow_is_recovered_from_final_direction_probability() -> None:
    predictions = pd.DataFrame(
        {
            "p_up": [0.8, 0.8, 0.2, 0.2],
            "first_minute_side": ["YES", "NO", "YES", "NO"],
        }
    )

    p_follow = p_follow_from_direction_probability(predictions)

    assert p_follow.tolist() == pytest.approx([0.8, 0.2, 0.2, 0.8])


def test_four_bucket_gate_abstains_weak_continuation_disagreement_and_keeps_confirmed_reversal() -> None:
    frame = pd.DataFrame(
        {
            "target": [1, 0, 0, 1],
            "p_base": [0.53, 0.48, 0.42, 0.58],
            "base_decision": ["UP", "DOWN", "DOWN", "UP"],
            "first_minute_side": ["YES", "NO", "YES", "NO"],
            "post_first_minute_reversal": [False, False, True, True],
            "p_follow": [0.30, 0.80, 0.30, 0.80],
        }
    )

    gated = apply_four_bucket_abstain_gate(frame, p_follow_cutoff=0.35, base_band=0.05)

    assert gated["hybrid_decision"].tolist() == ["ABSTAIN", "DOWN", "DOWN", "ABSTAIN"]
    assert gated["hybrid_bucket"].tolist() == [
        "bucket_2_base_fm_follow_reversal",
        "bucket_1_base_fm_follow_fm",
        "bucket_3_base_reversal_follow_base",
        "bucket_4_base_reversal_follow_fm",
    ]


def test_decision_metrics_report_required_signal_aliases() -> None:
    metrics = compute_decision_metrics(
        pd.Series([1, 0, 1, 0]),
        pd.Series([0.9, 0.1, 0.4, 0.6]),
        pd.Series(["UP", "DOWN", "ABSTAIN", "UP"]),
        selected_t_up=0.7,
        selected_t_down=0.3,
    )

    assert metrics["accepted_count"] == 3.0
    assert metrics["up_signal_count"] == metrics["up_prediction_count"] == 2.0
    assert metrics["down_signal_count"] == metrics["down_prediction_count"] == 1.0
    assert metrics["signal_coverage"] == metrics["coverage"] == 0.75
    assert metrics["overall_signal_accuracy"] == metrics["accepted_sample_accuracy"]


def test_gate_search_enforces_min_coverage_constraint() -> None:
    frame = pd.DataFrame(
        {
            "target": [1, 1, 0, 0],
            "p_base": [0.9, 0.8, 0.2, 0.1],
            "base_decision": ["UP", "UP", "DOWN", "DOWN"],
            "first_minute_side": ["YES", "YES", "NO", "NO"],
            "post_first_minute_reversal": [False, False, False, False],
            "p_follow": [0.9, 0.9, 0.9, 0.9],
        }
    )

    _, best = search_four_bucket_gate(frame, p_follow_cutoffs=[0.35], base_bands=[0.03], min_coverage=0.70)

    assert best["constraint_satisfied"] is True
    assert best["coverage"] == 1.0
