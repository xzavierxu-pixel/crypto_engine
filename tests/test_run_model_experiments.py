from __future__ import annotations

import json
from pathlib import Path

from scripts.run_model_experiments import (
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
            "model_plugin": "logistic",
            "feature_count": 120,
            "duration_seconds": 10.123,
            "train_metrics": {"roc_auc": 0.54, "log_loss": 0.69, "accuracy": 0.52, "sample_count": 1000.0},
            "validation_metrics": {"roc_auc": 0.53, "log_loss": 0.692, "accuracy": 0.51, "sample_count": 200.0},
            "overfit_gap": {"roc_auc": 0.01, "log_loss": 0.002, "accuracy": 0.01},
            "derivatives": {"enabled": False, "packs": []},
        },
        {
            "variant": "funding_basis",
            "model_plugin": "catboost",
            "feature_count": 130,
            "duration_seconds": 25.5,
            "train_metrics": {"roc_auc": 0.59, "log_loss": 0.685, "accuracy": 0.55, "sample_count": 1000.0},
            "validation_metrics": {"roc_auc": 0.56, "log_loss": 0.689, "accuracy": 0.53, "sample_count": 200.0},
            "overfit_gap": {"roc_auc": 0.03, "log_loss": 0.004, "accuracy": 0.02},
            "derivatives": {"enabled": True, "packs": ["derivatives_funding", "derivatives_basis"]},
        },
        {
            "variant": "funding",
            "model_plugin": "lightgbm",
            "feature_count": 125,
            "duration_seconds": 18.0,
            "train_metrics": {"roc_auc": 0.57, "log_loss": 0.688, "accuracy": 0.54, "sample_count": 1000.0},
            "validation_metrics": {"roc_auc": 0.55, "log_loss": 0.690, "accuracy": 0.525, "sample_count": 200.0},
            "overfit_gap": {"roc_auc": 0.02, "log_loss": 0.002, "accuracy": 0.015},
            "derivatives": {"enabled": True, "packs": ["derivatives_funding"]},
        },
        {
            "variant": "funding_basis",
            "model_plugin": "logistic",
            "feature_count": 130,
            "duration_seconds": 8.5,
            "train_metrics": {"roc_auc": 0.56, "log_loss": 0.689, "accuracy": 0.53, "sample_count": 1000.0},
            "validation_metrics": {"roc_auc": 0.54, "log_loss": 0.691, "accuracy": 0.52, "sample_count": 200.0},
            "overfit_gap": {"roc_auc": 0.02, "log_loss": 0.002, "accuracy": 0.01},
            "derivatives": {"enabled": True, "packs": ["derivatives_funding", "derivatives_basis"]},
        },
    ]

    ranked_results = sorted(results, key=lambda result: (result["validation_metrics"]["roc_auc"], -result["validation_metrics"]["log_loss"], result["validation_metrics"]["accuracy"]), reverse=True)
    leaderboard = _build_leaderboard(ranked_results)
    variant_summary = _build_variant_summary(ranked_results)
    progression = _build_derivatives_progression(variant_summary)

    assert leaderboard[0]["rank"] == 1
    assert leaderboard[0]["variant"] == "funding_basis"
    assert leaderboard[0]["model_plugin"] == "catboost"

    assert [entry["variant"] for entry in variant_summary] == ["funding_basis", "funding", "baseline"]
    funding_basis_entry = next(entry for entry in variant_summary if entry["variant"] == "funding_basis")
    assert funding_basis_entry["best_model_plugin"] == "catboost"
    assert funding_basis_entry["delta_vs_baseline"]["roc_auc"] == 0.03

    assert [entry["variant"] for entry in progression] == ["baseline", "funding", "funding_basis"]
    assert progression[1]["delta_vs_previous"]["roc_auc"] == 0.02
    assert progression[2]["delta_vs_previous"]["roc_auc"] == 0.01


def test_render_summary_markdown_includes_leaderboard_and_progression_sections() -> None:
    summary = {
        "results": [{"variant": "funding", "model_plugin": "catboost"}],
        "best_variant": "funding",
        "best_model_plugin": "catboost",
        "validation_window_days": 30,
        "leaderboard": [
            {
                "rank": 1,
                "variant": "funding",
                "model_plugin": "catboost",
                "feature_count": 130,
                "validation_roc_auc": 0.56,
                "validation_log_loss": 0.689,
                "validation_accuracy": 0.53,
                "train_roc_auc": 0.59,
                "overfit_gap_roc_auc": 0.03,
                "overfit_gap_log_loss": 0.004,
                "duration_seconds": 25.5,
            }
        ],
        "variant_summary": [
            {
                "variant": "funding",
                "best_model_plugin": "catboost",
                "validation_metrics": {"roc_auc": 0.56, "log_loss": 0.689, "accuracy": 0.53, "sample_count": 200},
                "delta_vs_baseline": {"roc_auc": 0.02, "log_loss": -0.001, "accuracy": 0.01},
            }
        ],
        "derivatives_progression": [
            {
                "variant": "funding",
                "best_model_plugin": "catboost",
                "validation_roc_auc": 0.56,
                "validation_log_loss": 0.689,
                "validation_accuracy": 0.53,
                "delta_vs_previous": {"roc_auc": 0.02, "log_loss": -0.001, "accuracy": 0.01},
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


def test_collect_existing_results_and_write_summary_support_resume_flow(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline" / "catboost"
    baseline_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding" / "lightgbm"
    funding_dir.mkdir(parents=True)

    baseline_result = {
        "variant": "baseline",
        "model_plugin": "catboost",
        "feature_count": 126,
        "duration_seconds": 10.0,
        "train_metrics": {"roc_auc": 0.58, "log_loss": 0.686, "accuracy": 0.55, "sample_count": 1000.0},
        "validation_metrics": {"roc_auc": 0.537, "log_loss": 0.691, "accuracy": 0.525, "sample_count": 200.0},
        "overfit_gap": {"roc_auc": 0.043, "log_loss": 0.005, "accuracy": 0.025},
        "derivatives": {"enabled": False, "packs": []},
    }
    funding_result = {
        "variant": "funding",
        "model_plugin": "lightgbm",
        "feature_count": 130,
        "duration_seconds": 8.0,
        "train_metrics": {"roc_auc": 0.59, "log_loss": 0.684, "accuracy": 0.56, "sample_count": 1000.0},
        "validation_metrics": {"roc_auc": 0.545, "log_loss": 0.690, "accuracy": 0.528, "sample_count": 200.0},
        "overfit_gap": {"roc_auc": 0.045, "log_loss": 0.006, "accuracy": 0.032},
        "derivatives": {"enabled": True, "packs": ["derivatives_funding"]},
    }
    (baseline_dir / "experiment_report.json").write_text(json.dumps(baseline_result), encoding="utf-8")
    (funding_dir / "experiment_report.json").write_text(json.dumps(funding_result), encoding="utf-8")

    collected = _collect_existing_results(tmp_path, variants=["baseline", "funding"], model_plugins=["catboost", "lightgbm"])

    assert len(collected) == 2
    assert _ordered_variants(collected) == ["baseline", "funding"]

    summary = _write_summary(
        output_dir=tmp_path,
        variants=_ordered_variants(collected),
        validation_window_days=7,
        results=collected,
    )

    assert summary["best_variant"] == "funding"
    assert summary["best_model_plugin"] == "lightgbm"
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()
