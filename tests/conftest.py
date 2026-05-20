from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def use_legacy_grid_5m_label(settings):
    settings.horizons.specs["5m"] = replace(
        settings.horizons.specs["5m"],
        label_builder="grid_direction",
        label_params={"label_version": "settlement_direction_t0_open_to_t4_close_tie_up_v2"},
    )
    return replace(
        settings,
        decision_alignment=replace(
            settings.decision_alignment,
            enabled=False,
            mode="exact_signal_t0",
            feature_offset_minutes=0,
        ),
    )
