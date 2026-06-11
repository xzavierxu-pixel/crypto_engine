from __future__ import annotations

import pandas as pd

from scripts.analysis.reversal_precision_gate import apply_reversal_precision_gate


def test_reversal_precision_gate_overrides_and_filters_continuation_conflicts() -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1, 1],
            "p_base": [0.53, 0.52, 0.80],
            "base_decision": ["UP", "UP", "UP"],
            "first_minute_side": ["YES", "YES", "YES"],
            "reversal_risk": [0.70, 0.58, 0.20],
            "follow_reversal_probability": [0.60, 0.50, 0.10],
            "base_confidence": [0.03, 0.02, 0.30],
            "first_minute_abs_return": [0.0, 0.0, 0.0],
        }
    )

    decisions = apply_reversal_precision_gate(
        frame,
        override_threshold=0.65,
        max_base_confidence=0.10,
        min_first_minute_abs_return=0.0,
        min_follow_reversal_probability=0.55,
        low_risk_filter_threshold=0.50,
        low_follow_filter_threshold=0.45,
        max_continuation_risk=0.55,
        max_continuation_follow_reversal=0.55,
        include_abstains=False,
    )

    assert decisions.tolist() == ["DOWN", "ABSTAIN", "UP"]
