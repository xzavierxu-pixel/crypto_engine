from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.run_model_experiments import (
    _available_derivatives_sources,
    _collect_existing_results,
    _build_derivatives_progression,
    _build_leaderboard,
    _build_settings_variant,
    _build_variant_summary,
    _ordered_variants,
    _render_summary_markdown,
    _write_summary,
    _resolve_variants,
)
from src.core.config import load_settings
import pandas as pd


def test_resolve_variants_auto_uses_derivatives_when_available() -> None:
    assert _resolve_variants("auto", derivatives_frame_present=False) == ["baseline"]
    assert _resolve_variants("auto", derivatives_frame_present=True) == [
        "baseline",
        "funding",
        "funding_basis",
        "funding_basis_book_ticker",
        "funding_basis_book_ticker_oi",
        "funding_basis_book_ticker_oi_options",
    ]


def test_resolve_variants_auto_degrades_when_book_ticker_is_unavailable() -> None:
    derivatives_frame = pd.DataFrame(
        {
            "raw_funding_rate": [0.0001],
            "raw_mark_price": [100000.0],
            "raw_index_price": [99990.0],
            "raw_premium_index": [0.0002],
            "raw_open_interest": [1234.0],
            "raw_atm_iv_near": [0.55],
        }
    )

    available_sources = _available_derivatives_sources(derivatives_frame)

    assert available_sources == {"funding", "basis", "oi", "options"}
    assert _resolve_variants(
        "auto",
        derivatives_frame_present=True,
        available_derivatives_sources=available_sources,
    ) == [
        "baseline",
        "funding",
        "funding_basis",
        "funding_basis_oi",
        "funding_basis_oi_options",
    ]


def test_available_derivatives_sources_accepts_merged_frame_column_names() -> None:
    derivatives_frame = pd.DataFrame(
        {
            "funding_rate": [0.0001],
            "mark_price": [100000.0],
            "index_price": [99990.0],
            "premium_index": [0.0002],
            "open_interest": [1234.0],
            "atm_iv_near": [0.55],
        }
    )

    assert _available_derivatives_sources(derivatives_frame) == {"funding", "basis", "oi", "options"}


def test_build_settings_variant_toggles_derivatives_without_changing_core_packs() -> None:
    settings = load_settings()

    baseline = _build_settings_variant(settings, "5m", "baseline")
    funding_basis_book_ticker_oi_options = _build_settings_variant(
        settings,
        "5m",
        "funding_basis_book_ticker_oi_options",
    )

    baseline_packs = baseline.features.get_profile("core_5m").packs
    full_packs = funding_basis_book_ticker_oi_options.features.get_profile("core_5m").packs

    assert "momentum" in baseline_packs
    assert "derivatives_funding" not in baseline_packs
    assert "derivatives_basis" not in baseline_packs
    assert "derivatives_oi" not in baseline_packs
    assert baseline.derivatives.enabled is False

    assert "derivatives_funding" in full_packs
    assert "derivatives_basis" in full_packs
    assert "derivatives_book_ticker" in full_packs
    assert "derivatives_oi" in full_packs
    assert "derivatives_options" in full_packs
    assert funding_basis_book_ticker_oi_options.derivatives.enabled is True
    assert funding_basis_book_ticker_oi_options.derivatives.book_ticker.enabled is True
    assert funding_basis_book_ticker_oi_options.derivatives.oi.enabled is True
    assert funding_basis_book_ticker_oi_options.derivatives.options.enabled is True


def test_build_settings_variant_enables_book_ticker_at_the_expected_stage() -> None:
    settings = load_settings()

    funding_basis = _build_settings_variant(settings, "5m", "funding_basis")
    funding_basis_book_ticker = _build_settings_variant(settings, "5m", "funding_basis_book_ticker")

    assert funding_basis.derivatives.book_ticker.enabled is False
    assert "derivatives_book_ticker" not in funding_basis.features.get_profile("core_5m").packs

    assert funding_basis_book_ticker.derivatives.book_ticker.enabled is True
    assert funding_basis_book_ticker.derivatives.oi.enabled is False
    assert funding_basis_book_ticker.derivatives.options.enabled is False
    assert "derivatives_book_ticker" in funding_basis_book_ticker.features.get_profile("core_5m").packs


def test_build_settings_variant_supports_non_book_ticker_progression() -> None:
    settings = load_settings()

    funding_basis_oi = _build_settings_variant(settings, "5m", "funding_basis_oi")
    funding_basis_oi_options = _build_settings_variant(settings, "5m", "funding_basis_oi_options")

    assert funding_basis_oi.derivatives.book_ticker.enabled is False
    assert funding_basis_oi.derivatives.oi.enabled is True
    assert funding_basis_oi.derivatives.options.enabled is False
    assert "derivatives_book_ticker" not in funding_basis_oi.features.get_profile("core_5m").packs
    assert "derivatives_oi" in funding_basis_oi.features.get_profile("core_5m").packs

    assert funding_basis_oi_options.derivatives.book_ticker.enabled is False
    assert funding_basis_oi_options.derivatives.oi.enabled is True
    assert funding_basis_oi_options.derivatives.options.enabled is True
    assert "derivatives_options" in funding_basis_oi_options.features.get_profile("core_5m").packs


def test_experiment_summary_helpers_build_rankings_and_variant_rollups() -> None:
    results = [
        {
            "variant": "baseline",
            "model_plugins": {"stage1": "logistic", "stage2": "logistic"},
            "feature_count": 120,
            "duration_seconds": 10.123,
            "train_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.01, "precision_up": 0.30, "precision_down": 0.31, "coverage_end_to_end": 0.50}},
            "validation_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.008, "precision_up": 0.25, "precision_down": 0.26, "coverage_end_to_end": 0.40, "sample_count": 200.0}},
            "walk_forward_summary": {"trade_pnl.pnl_per_sample_mean": 0.007, "trade_pnl.pnl_per_sample_min": 0.001, "precision_up_mean": 0.25, "precision_down_mean": 0.26},
            "overfit_gap": {"pnl_per_sample": 0.002, "precision_sum": 0.10, "coverage": 0.10},
            "derivatives": {"enabled": False, "packs": []},
        },
        {
            "variant": "funding_basis",
            "model_plugins": {"stage1": "catboost", "stage2": "catboost"},
            "feature_count": 130,
            "duration_seconds": 25.5,
            "train_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.05, "precision_up": 0.40, "precision_down": 0.38, "coverage_end_to_end": 0.60}},
            "validation_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.04, "precision_up": 0.36, "precision_down": 0.35, "coverage_end_to_end": 0.55, "sample_count": 200.0}},
            "walk_forward_summary": {"trade_pnl.pnl_per_sample_mean": 0.035, "trade_pnl.pnl_per_sample_min": 0.020, "precision_up_mean": 0.35, "precision_down_mean": 0.34},
            "overfit_gap": {"pnl_per_sample": 0.01, "precision_sum": 0.07, "coverage": 0.05},
            "derivatives": {"enabled": True, "packs": ["derivatives_funding", "derivatives_basis"]},
        },
        {
            "variant": "funding",
            "model_plugins": {"stage1": "lightgbm_stage1", "stage2": "lightgbm_stage2"},
            "feature_count": 125,
            "duration_seconds": 18.0,
            "train_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.03, "precision_up": 0.34, "precision_down": 0.33, "coverage_end_to_end": 0.54}},
            "validation_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.02, "precision_up": 0.31, "precision_down": 0.30, "coverage_end_to_end": 0.50, "sample_count": 200.0}},
            "walk_forward_summary": {"trade_pnl.pnl_per_sample_mean": 0.018, "trade_pnl.pnl_per_sample_min": 0.010, "precision_up_mean": 0.30, "precision_down_mean": 0.29},
            "overfit_gap": {"pnl_per_sample": 0.01, "precision_sum": 0.06, "coverage": 0.04},
            "derivatives": {"enabled": True, "packs": ["derivatives_funding"]},
        },
        {
            "variant": "funding_basis",
            "model_plugins": {"stage1": "logistic", "stage2": "catboost"},
            "feature_count": 130,
            "duration_seconds": 8.5,
            "train_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.025, "precision_up": 0.33, "precision_down": 0.32, "coverage_end_to_end": 0.52}},
            "validation_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.015, "precision_up": 0.29, "precision_down": 0.28, "coverage_end_to_end": 0.48, "sample_count": 200.0}},
            "walk_forward_summary": {"trade_pnl.pnl_per_sample_mean": 0.013, "trade_pnl.pnl_per_sample_min": 0.002, "precision_up_mean": 0.28, "precision_down_mean": 0.27},
            "overfit_gap": {"pnl_per_sample": 0.01, "precision_sum": 0.08, "coverage": 0.04},
            "derivatives": {"enabled": True, "packs": ["derivatives_funding", "derivatives_basis"]},
        },
    ]

    ranked_results = sorted(
        results,
        key=lambda result: (
            result["walk_forward_summary"]["trade_pnl.pnl_per_sample_mean"],
            result["walk_forward_summary"]["trade_pnl.pnl_per_sample_min"],
            result["walk_forward_summary"]["precision_up_mean"] + result["walk_forward_summary"]["precision_down_mean"],
        ),
        reverse=True,
    )
    leaderboard = _build_leaderboard(ranked_results)
    variant_summary = _build_variant_summary(ranked_results)
    progression = _build_derivatives_progression(variant_summary)

    assert leaderboard[0]["rank"] == 1
    assert leaderboard[0]["variant"] == "funding_basis"
    assert leaderboard[0]["stage1_model_plugin"] == "catboost"
    assert leaderboard[0]["stage2_model_plugin"] == "catboost"

    assert [entry["variant"] for entry in variant_summary] == ["funding_basis", "funding", "baseline"]
    funding_basis_entry = next(entry for entry in variant_summary if entry["variant"] == "funding_basis")
    assert funding_basis_entry["best_model_plugins"]["stage1"] == "catboost"
    assert funding_basis_entry["delta_vs_baseline"]["pnl_per_sample"] == 0.032

    assert [entry["variant"] for entry in progression] == ["baseline", "funding", "funding_basis"]
    assert progression[1]["delta_vs_previous"]["pnl_per_sample"] == 0.012
    assert progression[2]["delta_vs_previous"]["pnl_per_sample"] == 0.02


def test_render_summary_markdown_includes_leaderboard_and_progression_sections() -> None:
    summary = {
        "results": [{"variant": "funding", "model_plugin": "catboost"}],
        "best_variant": "funding",
        "best_model_plugins": {"stage1": "catboost", "stage2": "lightgbm_stage2"},
        "validation_window_days": 30,
        "leaderboard": [
            {
                "rank": 1,
                "variant": "funding",
                "stage1_model_plugin": "catboost",
                "stage2_model_plugin": "lightgbm_stage2",
                "feature_count": 130,
                "validation_pnl_per_sample": 0.04,
                "validation_precision_sum": 0.71,
                "validation_coverage": 0.55,
                "walk_forward_pnl_per_sample_mean": 0.03,
                "walk_forward_pnl_per_sample_min": 0.01,
                "duration_seconds": 25.5,
            }
        ],
        "variant_summary": [
            {
                "variant": "funding",
                "best_model_plugins": {"stage1": "catboost", "stage2": "lightgbm_stage2"},
                "validation_metrics": {"pnl_per_sample": 0.04, "precision_sum": 0.71, "coverage": 0.55, "sample_count": 200},
                "delta_vs_baseline": {"pnl_per_sample": 0.02, "precision_sum": 0.03, "coverage": 0.05},
            }
        ],
        "derivatives_progression": [
            {
                "variant": "funding",
                "best_model_plugins": {"stage1": "catboost", "stage2": "lightgbm_stage2"},
                "validation_pnl_per_sample": 0.04,
                "validation_precision_sum": 0.71,
                "validation_coverage": 0.55,
                "delta_vs_previous": {"pnl_per_sample": 0.02, "precision_sum": 0.03, "coverage": 0.05},
            }
        ],
    }

    markdown = _render_summary_markdown(summary)

    assert "# Model Experiment Summary" in markdown
    assert "## Leaderboard" in markdown
    assert "## Best By Variant" in markdown
    assert "## Derivatives Progression" in markdown
    assert "`funding`" in markdown
    assert "`catboost`" in markdown
    assert "`lightgbm_stage2`" in markdown


def test_collect_existing_results_and_write_summary_support_resume_flow(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline" / "catboost"
    baseline_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding" / "lightgbm_stage1__lightgbm_stage2"
    funding_dir.mkdir(parents=True)

    baseline_result = {
        "variant": "baseline",
        "model_plugins": {"stage1": "catboost", "stage2": "catboost"},
        "feature_count": 126,
        "duration_seconds": 10.0,
        "train_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.02, "precision_up": 0.27, "precision_down": 0.26, "coverage_end_to_end": 0.50}},
        "validation_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.01, "precision_up": 0.26, "precision_down": 0.26, "coverage_end_to_end": 0.48, "sample_count": 200.0}},
        "walk_forward_summary": {"trade_pnl.pnl_per_sample_mean": 0.008, "trade_pnl.pnl_per_sample_min": 0.002, "precision_up_mean": 0.26, "precision_down_mean": 0.26},
        "overfit_gap": {"pnl_per_sample": 0.01, "precision_sum": 0.01, "coverage": 0.02},
        "derivatives": {"enabled": False, "packs": []},
    }
    funding_result = {
        "variant": "funding",
        "model_plugins": {"stage1": "lightgbm_stage1", "stage2": "lightgbm_stage2"},
        "feature_count": 130,
        "duration_seconds": 8.0,
        "train_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.03, "precision_up": 0.31, "precision_down": 0.31, "coverage_end_to_end": 0.54}},
        "validation_metrics": {"end_to_end": {"trade_pnl.pnl_per_sample": 0.02, "precision_up": 0.30, "precision_down": 0.30, "coverage_end_to_end": 0.50, "sample_count": 200.0}},
        "walk_forward_summary": {"trade_pnl.pnl_per_sample_mean": 0.018, "trade_pnl.pnl_per_sample_min": 0.010, "precision_up_mean": 0.30, "precision_down_mean": 0.30},
        "overfit_gap": {"pnl_per_sample": 0.01, "precision_sum": 0.02, "coverage": 0.04},
        "derivatives": {"enabled": True, "packs": ["derivatives_funding"]},
    }
    (baseline_dir / "experiment_report.json").write_text(json.dumps(baseline_result), encoding="utf-8")
    (funding_dir / "experiment_report.json").write_text(json.dumps(funding_result), encoding="utf-8")

    collected = _collect_existing_results(
        tmp_path,
        variants=["baseline", "funding"],
        stage1_model_plugins=["catboost", "lightgbm_stage1"],
        stage2_model_plugins=["catboost", "lightgbm_stage2"],
    )

    assert len(collected) == 2
    assert _ordered_variants(collected) == ["baseline", "funding"]

    summary = _write_summary(
        output_dir=tmp_path,
        variants=_ordered_variants(collected),
        validation_window_days=7,
        results=collected,
    )

    assert summary["best_variant"] == "funding"
    assert summary["best_model_plugins"]["stage1"] == "lightgbm_stage1"
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()
