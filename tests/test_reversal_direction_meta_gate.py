from __future__ import annotations

import pandas as pd

from scripts.analysis.reversal_direction_meta_gate import _decisions_from_first_minute_relative_probability


def test_first_minute_relative_decisions_follow_reverse_or_abstain() -> None:
    frame = pd.DataFrame({"first_minute_side": ["YES", "NO", "YES", "NO"]})
    p_up = pd.Series([0.80, 0.20, 0.20, 0.80])

    decisions = _decisions_from_first_minute_relative_probability(
        frame,
        p_up,
        t_follow=0.70,
        t_reversal=0.30,
    )

    assert decisions.tolist() == ["UP", "DOWN", "DOWN", "UP"]
