from __future__ import annotations

from scripts.analysis.replay_window_summary import build_replay_summary


def test_replay_window_summary_recomputes_threshold_metrics() -> None:
    replay = {
        "window_start_utc": "2026-05-20T15:23:40+00:00",
        "window_end_utc": "2026-05-21T00:23:40+00:00",
        "current_artifact": {"artifact_dir": "artifact"},
        "combos": [
            {
                "name": "current_model_current_thresholds",
                "rows": [
                    {"p_up": 0.8, "actual_side": "YES", "confidence_bucket": "0.80-0.90"},
                    {"p_up": 0.2, "actual_side": "NO", "confidence_bucket": "0.80-0.90"},
                    {"p_up": 0.5, "actual_side": "YES", "confidence_bucket": "0.50-0.60"},
                ],
            }
        ],
    }

    summary = build_replay_summary(
        replay,
        combo_name="current_model_current_thresholds",
        run_id="test_run",
        t_up=0.7,
        t_down=0.3,
    )

    assert summary["metrics"]["sample_count"] == 3.0
    assert summary["metrics"]["accepted_count"] == 2.0
    assert summary["metrics"]["coverage"] == 2 / 3
    assert summary["metrics"]["accepted_sample_accuracy"] == 1.0
    assert summary["by_confidence_bucket"]["0.80-0.90"]["accepted_count"] == 2.0
