from __future__ import annotations

import pandas as pd

from scripts.analysis.reversal_rescue_gate import apply_rescue_gate


def test_rescue_gate_keeps_low_risk_reverses_high_risk_and_abstains_middle() -> None:
    predictions = pd.DataFrame(
        {
            "decision": ["UP", "UP", "UP", "DOWN"],
            "first_minute_side": ["YES", "YES", "YES", "YES"],
        }
    )
    rescue_probability = pd.Series([0.10, 0.50, 0.90, 0.20])

    decisions = apply_rescue_gate(
        predictions,
        rescue_probability,
        keep_threshold=0.20,
        reverse_threshold=0.80,
        keep_base_reversal=True,
        use_abstains=False,
    )

    assert decisions.tolist() == ["UP", "ABSTAIN", "DOWN", "DOWN"]
