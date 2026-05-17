from __future__ import annotations

import pandas as pd

from scripts.experiments.build_polymarket_resolved_training_frame import (
    _apply_resolved_labels,
    _frame_with_polymarket_slugs,
    _label_frame_from_markets,
    _resolved_btc_up_label,
    _write_resolved_splits,
)


def test_resolved_btc_up_label_uses_winning_outcome_price() -> None:
    up_market = {
        "closed": True,
        "outcomes": '["Up", "Down"]',
        "outcomePrices": '["1", "0"]',
    }
    down_market = {
        "closed": True,
        "outcomes": '["Up", "Down"]',
        "outcomePrices": '["0", "1"]',
    }

    assert _resolved_btc_up_label(up_market, win_threshold=0.99) == (1, "resolved")
    assert _resolved_btc_up_label(down_market, win_threshold=0.99) == (0, "resolved")


def test_apply_resolved_labels_replaces_target_and_records_mismatch() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-04-11T00:15:00Z", "2026-04-11T00:20:00Z"],
                utc=True,
            ),
            "target": [1, 0],
            "feature": [0.1, 0.2],
        }
    )
    frame = _frame_with_polymarket_slugs(frame)
    markets = [
        {
            "id": "1",
            "slug": "btc-updown-5m-1775866500",
            "closed": True,
            "outcomes": '["Up", "Down"]',
            "outcomePrices": '["0", "1"]',
        },
        {
            "id": "2",
            "slug": "btc-updown-5m-1775866800",
            "closed": True,
            "outcomes": '["Up", "Down"]',
            "outcomePrices": '["0", "1"]',
        },
    ]
    labels = _label_frame_from_markets(markets, wanted_slugs=set(frame["polymarket_slug"]), win_threshold=0.99)

    resolved, report = _apply_resolved_labels(frame, labels)

    assert resolved["polymarket_slug"].to_list() == [
        "btc-updown-5m-1775866500",
        "btc-updown-5m-1775866800",
    ]
    assert resolved["target"].to_list() == [0, 0]
    assert resolved["original_target"].to_list() == [1, 0]
    assert resolved["label_mismatch"].to_list() == [True, False]
    assert report["source_rows"] == 2
    assert report["resolved_rows"] == 2
    assert report["mismatch_count"] == 1
    assert report["mismatch_rate"] == 0.5


def test_write_resolved_splits_preserves_development_and_validation(tmp_path) -> None:
    dev_path = tmp_path / "development_frame.parquet"
    valid_path = tmp_path / "validation_frame.parquet"
    merged = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-04-11T00:15:00Z", "2026-04-12T00:15:00Z"],
                utc=True,
            ),
            "target": [1, 0],
            "source_frame_path": [str(dev_path), str(valid_path)],
        }
    )

    outputs = _write_resolved_splits(merged, [dev_path, valid_path], tmp_path / "resolved")

    assert set(outputs) == {"development_frame", "validation_frame"}
    assert pd.read_parquet(outputs["development_frame"])["target"].to_list() == [1]
    assert pd.read_parquet(outputs["validation_frame"])["target"].to_list() == [0]
