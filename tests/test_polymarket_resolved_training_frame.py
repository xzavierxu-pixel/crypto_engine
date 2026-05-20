from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.core.config import DatasetConfig, load_settings
from src.data.dataset_builder import build_training_frame, is_allowed_feature_column
from src.horizons.base import HorizonSpec
from src.labels.polymarket_resolved import (
    PolymarketResolvedLabelBuilder,
    add_polymarket_slugs,
    require_unique_label_slugs,
    resolved_btc_up_label,
)
from scripts.experiments.build_polymarket_resolved_training_frame import (
    _apply_resolved_labels,
    _frame_with_polymarket_slugs,
    _label_frame_from_markets,
    _resolved_btc_up_label,
    _write_resolved_splits,
)
from scripts.model.train_model import _label_metadata_report


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
    assert resolved_btc_up_label(up_market, win_threshold=0.99)[:2] == (1, "resolved")


def test_polymarket_resolved_slug_mapping_uses_market_t0() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-11T00:16:00Z"], utc=True),
            "market_t0": pd.to_datetime(["2026-04-11T00:15:00Z"], utc=True),
        }
    )

    enriched = add_polymarket_slugs(frame)

    assert enriched["polymarket_slug"].to_list() == ["btc-updown-5m-1775866500"]


def test_polymarket_resolved_label_drops_unresolved_and_ambiguous_market(tmp_path) -> None:
    store = pd.DataFrame(
        {
            "polymarket_slug": ["btc-updown-5m-1775866500", "btc-updown-5m-1775866800"],
            "target": [1, None],
            "polymarket_label_status": ["resolved", "ambiguous_outcome_prices"],
            "label_version": ["test_v1", "test_v1"],
        }
    )
    path = tmp_path / "labels.parquet"
    store.to_parquet(path, index=False)
    settings = load_settings()
    horizon = HorizonSpec(
        name="5m",
        minutes=5,
        grid_minutes=5,
        label_builder="polymarket_resolved",
        feature_profile="core_5m",
        label_params={"label_store_path": str(path), "label_version": "test_v1"},
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-11T00:15:00Z", periods=10, freq="1min"),
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1.0] * 10,
        }
    )

    labels = PolymarketResolvedLabelBuilder().build(frame, settings, horizon)

    assert labels["polymarket_slug"].to_list() == ["btc-updown-5m-1775866500"]
    assert labels["target"].to_list() == [1]


def test_polymarket_resolved_label_store_requires_unique_slug() -> None:
    labels = pd.DataFrame(
        {
            "polymarket_slug": ["btc-updown-5m-1", "btc-updown-5m-1"],
            "target": [1, 0],
            "polymarket_label_status": ["resolved", "resolved"],
        }
    )

    try:
        require_unique_label_slugs(labels)
    except ValueError as exc:
        assert "Duplicate Polymarket slugs" in str(exc)
    else:
        raise AssertionError("Expected duplicate slug QA failure.")


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


def test_build_training_frame_uses_polymarket_target_not_btc_target(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-11T00:15:00Z", periods=720, freq="1min"),
            "open": [100.0 + i for i in range(720)],
            "high": [101.0 + i for i in range(720)],
            "low": [99.0 + i for i in range(720)],
            "close": [100.5 + i for i in range(720)],
            "volume": [100.0 + (i % 37) * 3.0 for i in range(720)],
        }
    )
    grid = frame.iloc[::5].copy()
    slugs = add_polymarket_slugs(grid)["polymarket_slug"].to_list()
    store = pd.DataFrame(
        {
            "polymarket_slug": slugs,
            "target": [0] * len(slugs),
            "polymarket_label_status": ["resolved"] * len(slugs),
            "label_version": ["test_v1"] * len(slugs),
            "source": ["polymarket_gamma"] * len(slugs),
        }
    )
    path = tmp_path / "labels.parquet"
    store.to_parquet(path, index=False)
    base_settings = load_settings()
    five = base_settings.horizons.specs["5m"]
    scoped_settings = replace(
        base_settings,
        dataset=DatasetConfig(
            train_start="2026-04-11T00:00:00Z",
            train_end="2026-04-11T12:00:00Z",
            strict_grid_only=True,
            drop_incomplete_candles=True,
        ),
        derivatives=replace(base_settings.derivatives, enabled=False),
        second_level=replace(base_settings.second_level, enabled=False),
    )
    scoped_settings.horizons.specs["5m"] = replace(
        five,
        label_builder="polymarket_resolved",
        label_params={"label_store_path": str(path), "label_version": "test_v1"},
    )

    training = build_training_frame(frame, scoped_settings, horizon_name="5m")

    assert training.frame["target"].eq(0).all()
    assert training.frame["original_btc_direction_target"].eq(1.0).all()
    assert "original_btc_direction_target" not in training.feature_columns
    assert "polymarket_slug" not in training.feature_columns
    assert not is_allowed_feature_column("polymarket_slug")


def test_report_contains_resolved_label_metadata(tmp_path) -> None:
    settings = load_settings()
    settings.horizons.specs["5m"] = replace(
        settings.horizons.specs["5m"],
        label_builder="polymarket_resolved",
        label_params={
            "label_version": "test_v1",
            "label_store_path": str(tmp_path / "labels.parquet"),
        },
    )
    frame = pd.DataFrame(
        {
            "target": [1, 0, 1],
            "label_mismatch_vs_btc_direction": [False, True, False],
        }
    )

    report = _label_metadata_report(
        frame,
        settings,
        horizon_name="5m",
        label_store_path=str(tmp_path / "labels.parquet"),
    )

    assert report["label_source"] == "polymarket_resolved"
    assert report["label_version"] == "test_v1"
    assert report["resolved_label_count"] == 3
    assert report["label_mismatch_count_vs_btc_direction"] == 1
    assert report["polymarket_target_mean"] == 2 / 3
    assert report["coverage_constraint_min"] == 0.70
    assert report["target_semantics"] == "target is Polymarket resolved outcome, not BTC OHLCV direction"
