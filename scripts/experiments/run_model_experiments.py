from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'src').is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import FeaturesConfig, Settings, load_settings
from src.data.dataset_builder import build_training_frame
from src.data.derivatives.feature_store import load_derivatives_frame_from_settings
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.model.train import train_two_stage_model


CANONICAL_VARIANT_ORDER = [
    "baseline",
    "funding",
    "funding_basis",
    "funding_basis_oi",
    "funding_basis_oi_options",
    "funding_basis_book_ticker",
    "funding_basis_book_ticker_oi",
    "funding_basis_book_ticker_oi_options",
]


def _load_input(path: Path):
    if path.suffix.lower() == ".csv":
        return load_ohlcv_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return load_ohlcv_parquet(path)
    if path.suffix.lower() in {".feather", ".ft"}:
        return load_ohlcv_feather(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _select_metrics(metrics: dict[str, float]) -> dict[str, float]:
    allowed = {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "multiclass_precision_up",
        "multiclass_precision_down",
        "multiclass_recall_up",
        "multiclass_recall_down",
        "trade_precision_up",
        "trade_precision_down",
        "trade_recall_up",
        "trade_recall_down",
        "up_auc",
        "down_auc",
        "macro_f1",
        "log_loss",
        "roc_auc",
        "sample_count",
        "coverage",
        "coverage_end_to_end",
        "stage1_selected_count",
        "stage1_selected_ratio",
        "stage2_trade_count",
        "trade_pnl.pnl_per_trade",
        "trade_pnl.pnl_per_sample",
        "class_pnl.up",
        "class_pnl.down",
        "support_up",
        "support_down",
        "support_flat",
    }
    return {key: value for key, value in metrics.items() if key in allowed}


def _rank_key(result: dict) -> tuple[float, float, float]:
    return (
        result["walk_forward_summary"].get("trade_pnl.pnl_per_sample_mean", float("-inf")),
        result["walk_forward_summary"].get("trade_pnl.pnl_per_sample_min", float("-inf")),
        (
            result["walk_forward_summary"].get("trade_precision_up_mean", float("-inf"))
            + result["walk_forward_summary"].get("trade_precision_down_mean", float("-inf"))
        ),
    )


def _round_metric(value: float | int, digits: int = 6) -> float:
    return round(float(value), digits)


def _build_leaderboard(results: list[dict]) -> list[dict]:
    leaderboard: list[dict] = []
    for rank, result in enumerate(results, start=1):
        leaderboard.append(
            {
                "rank": rank,
                "variant": result["variant"],
                "stage1_model_plugin": result["model_plugins"]["stage1"],
                "stage2_model_plugin": result["model_plugins"]["stage2"],
                "feature_count": result["feature_count"],
                "validation_pnl_per_sample": _round_metric(result["validation_metrics"]["end_to_end"]["trade_pnl.pnl_per_sample"]),
                "validation_precision_sum": _round_metric(
                    result["validation_metrics"]["end_to_end"].get("trade_precision_up", 0.0)
                    + result["validation_metrics"]["end_to_end"].get("trade_precision_down", 0.0)
                ),
                "validation_coverage": _round_metric(result["validation_metrics"]["end_to_end"]["coverage_end_to_end"]),
                "walk_forward_pnl_per_sample_mean": _round_metric(
                    result["walk_forward_summary"].get("trade_pnl.pnl_per_sample_mean", 0.0)
                ),
                "walk_forward_pnl_per_sample_min": _round_metric(
                    result["walk_forward_summary"].get("trade_pnl.pnl_per_sample_min", 0.0)
                ),
                "duration_seconds": _round_metric(result["duration_seconds"], digits=2),
            }
        )
    return leaderboard


def _build_variant_summary(results: list[dict]) -> list[dict]:
    best_by_variant: dict[str, dict] = {}
    for result in results:
        existing = best_by_variant.get(result["variant"])
        if existing is None or _rank_key(result) > _rank_key(existing):
            best_by_variant[result["variant"]] = result

    baseline = best_by_variant.get("baseline")
    summary: list[dict] = []
    for variant, best_result in sorted(best_by_variant.items(), key=lambda item: _rank_key(item[1]), reverse=True):
        validation = best_result["validation_metrics"]["end_to_end"]
        train = best_result["train_metrics"]["end_to_end"]
        entry = {
            "variant": variant,
            "best_model_plugins": best_result["model_plugins"],
            "feature_count": best_result["feature_count"],
            "validation_metrics": {
                "pnl_per_sample": _round_metric(validation["trade_pnl.pnl_per_sample"]),
                "precision_sum": _round_metric(validation.get("trade_precision_up", 0.0) + validation.get("trade_precision_down", 0.0)),
                "coverage": _round_metric(validation["coverage_end_to_end"]),
                "sample_count": int(validation.get("sample_count", 0)),
            },
            "train_metrics": {
                "pnl_per_sample": _round_metric(train["trade_pnl.pnl_per_sample"]),
                "precision_sum": _round_metric(train.get("trade_precision_up", 0.0) + train.get("trade_precision_down", 0.0)),
                "coverage": _round_metric(train["coverage_end_to_end"]),
            },
            "overfit_gap": {
                "pnl_per_sample": _round_metric(best_result["overfit_gap"]["pnl_per_sample"]),
                "precision_sum": _round_metric(best_result["overfit_gap"]["precision_sum"]),
                "coverage": _round_metric(best_result["overfit_gap"]["coverage"]),
            },
            "walk_forward_summary": best_result["walk_forward_summary"],
            "derivatives": best_result["derivatives"],
        }
        if baseline is not None:
            baseline_validation = baseline["validation_metrics"]["end_to_end"]
            entry["delta_vs_baseline"] = {
                "pnl_per_sample": _round_metric(
                    validation["trade_pnl.pnl_per_sample"] - baseline_validation["trade_pnl.pnl_per_sample"]
                ),
                "precision_sum": _round_metric(
                    (validation.get("trade_precision_up", 0.0) + validation.get("trade_precision_down", 0.0))
                    - (baseline_validation.get("trade_precision_up", 0.0) + baseline_validation.get("trade_precision_down", 0.0))
                ),
                "coverage": _round_metric(validation["coverage_end_to_end"] - baseline_validation["coverage_end_to_end"]),
            }
        summary.append(entry)
    return summary


def _build_derivatives_progression(variant_summary: list[dict]) -> list[dict]:
    entries_by_variant = {entry["variant"]: entry for entry in variant_summary}
    progression: list[dict] = []
    previous_entry: dict | None = None

    for variant in CANONICAL_VARIANT_ORDER:
        entry = entries_by_variant.get(variant)
        if entry is None:
            continue

        metrics = entry["validation_metrics"]
        progression_entry = {
            "variant": variant,
            "best_model_plugins": entry["best_model_plugins"],
            "validation_pnl_per_sample": metrics["pnl_per_sample"],
            "validation_precision_sum": metrics["precision_sum"],
            "validation_coverage": metrics["coverage"],
        }
        if previous_entry is not None:
            previous_metrics = previous_entry["validation_metrics"]
            progression_entry["delta_vs_previous"] = {
                "pnl_per_sample": _round_metric(metrics["pnl_per_sample"] - previous_metrics["pnl_per_sample"]),
                "precision_sum": _round_metric(metrics["precision_sum"] - previous_metrics["precision_sum"]),
                "coverage": _round_metric(metrics["coverage"] - previous_metrics["coverage"]),
            }
        progression.append(progression_entry)
        previous_entry = entry

    return progression


def _render_summary_markdown(summary: dict) -> str:
    lines = [
        "# Model Experiment Summary",
        "",
        "## Best Result",
        "",
    ]

    if not summary["results"]:
        lines.extend(["No experiment results were produced.", ""])
        return "\n".join(lines)

    lines.extend(
        [
            f"- Best variant: `{summary['best_variant']}`",
            f"- Best model plugins: `stage1={summary['best_model_plugins']['stage1']}`, `stage2={summary['best_model_plugins']['stage2']}`",
            f"- Validation window days: `{summary['validation_window_days']}`",
            "",
            "## Leaderboard",
            "",
            "| Rank | Variant | Stage1 | Stage2 | PnL/Sample | Precision Sum | Coverage | WF Mean | WF Min | Features | Duration (s) |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in summary["leaderboard"]:
        lines.append(
            f"| {entry['rank']} | `{entry['variant']}` | `{entry['stage1_model_plugin']}` | `{entry['stage2_model_plugin']}` | "
            f"{entry['validation_pnl_per_sample']:.6f} | {entry['validation_precision_sum']:.6f} | "
            f"{entry['validation_coverage']:.6f} | {entry['walk_forward_pnl_per_sample_mean']:.6f} | "
            f"{entry['walk_forward_pnl_per_sample_min']:.6f} | {entry['feature_count']} | {entry['duration_seconds']:.2f} |"
        )

    lines.extend(["", "## Best By Variant", ""])
    lines.append("| Variant | Best Stage1 | Best Stage2 | PnL/Sample | Precision Sum | Coverage | dPnL vs Baseline |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for entry in summary["variant_summary"]:
        delta = entry.get("delta_vs_baseline", {})
        lines.append(
            f"| `{entry['variant']}` | `{entry['best_model_plugins']['stage1']}` | `{entry['best_model_plugins']['stage2']}` | "
            f"{entry['validation_metrics']['pnl_per_sample']:.6f} | {entry['validation_metrics']['precision_sum']:.6f} | "
            f"{entry['validation_metrics']['coverage']:.6f} | {float(delta.get('pnl_per_sample', 0.0)):.6f} |"
        )

    if summary["derivatives_progression"]:
        lines.extend(["", "## Derivatives Progression", ""])
        lines.append("| Variant | Stage1 | Stage2 | PnL/Sample | dPnL vs Prev | Precision Sum | dPrec vs Prev | Coverage | dCov vs Prev |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for entry in summary["derivatives_progression"]:
            delta = entry.get("delta_vs_previous", {})
            lines.append(
                f"| `{entry['variant']}` | `{entry['best_model_plugins']['stage1']}` | `{entry['best_model_plugins']['stage2']}` | "
                f"{entry['validation_pnl_per_sample']:.6f} | {float(delta.get('pnl_per_sample', 0.0)):.6f} | "
                f"{entry['validation_precision_sum']:.6f} | {float(delta.get('precision_sum', 0.0)):.6f} | "
                f"{entry['validation_coverage']:.6f} | {float(delta.get('coverage', 0.0)):.6f} |"
            )

    return "\n".join(lines) + "\n"


def _load_existing_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(
    output_dir: Path,
    variants: list[str],
    validation_window_days: int,
    results: list[dict],
) -> dict:
    ranked_results = sorted(results, key=_rank_key, reverse=True)
    leaderboard = _build_leaderboard(ranked_results)
    variant_summary = _build_variant_summary(ranked_results)
    derivatives_progression = _build_derivatives_progression(variant_summary)
    summary = {
        "ranking_metric_priority": [
            "trade_pnl.pnl_per_sample_mean",
            "trade_pnl.pnl_per_sample_min",
            "trade_precision_up_mean + trade_precision_down_mean",
        ],
        "validation_window_days": validation_window_days,
        "variants": variants,
        "results": ranked_results,
        "leaderboard": leaderboard,
        "variant_summary": variant_summary,
        "derivatives_progression": derivatives_progression,
        "best_model_plugins": ranked_results[0]["model_plugins"] if ranked_results else None,
        "best_variant": ranked_results[0]["variant"] if ranked_results else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    return summary


def _collect_existing_results(
    output_dir: Path,
    variants: list[str] | None = None,
    stage1_model_plugins: list[str] | None = None,
    stage2_model_plugins: list[str] | None = None,
) -> list[dict]:
    allowed_variants = set(variants) if variants is not None else None
    allowed_stage1_model_plugins = set(stage1_model_plugins) if stage1_model_plugins is not None else None
    allowed_stage2_model_plugins = set(stage2_model_plugins) if stage2_model_plugins is not None else None
    results: list[dict] = []
    for report_path in sorted(output_dir.rglob("experiment_report.json")):
        result = _load_existing_result(report_path)
        if allowed_variants is not None and result["variant"] not in allowed_variants:
            continue
        if allowed_stage1_model_plugins is not None and result.get("model_plugins", {}).get("stage1") not in allowed_stage1_model_plugins:
            continue
        if allowed_stage2_model_plugins is not None and result.get("model_plugins", {}).get("stage2") not in allowed_stage2_model_plugins:
            continue
        results.append(result)
    return results


def _ordered_variants(results: list[dict]) -> list[str]:
    discovered = {result["variant"] for result in results}
    ordered = [variant for variant in CANONICAL_VARIANT_ORDER if variant in discovered]
    extras = sorted(discovered - set(CANONICAL_VARIANT_ORDER))
    return [*ordered, *extras]


def _available_derivatives_sources(derivatives_frame) -> set[str]:
    if derivatives_frame is None or derivatives_frame.empty:
        return set()

    available: set[str] = set()
    columns = set(derivatives_frame.columns)

    if "raw_funding_rate" in columns or "funding_rate" in columns:
        available.add("funding")
    if (
        {"raw_mark_price", "raw_index_price", "raw_premium_index"}.issubset(columns)
        or {"mark_price", "index_price", "premium_index"}.issubset(columns)
    ):
        available.add("basis")
    if (
        {"raw_bid_price", "raw_bid_qty", "raw_ask_price", "raw_ask_qty"}.issubset(columns)
        or {"bid_price", "bid_qty", "ask_price", "ask_qty"}.issubset(columns)
    ):
        available.add("book_ticker")
    if "raw_open_interest" in columns or "open_interest" in columns:
        available.add("oi")
    if "raw_atm_iv_near" in columns or "atm_iv_near" in columns:
        available.add("options")

    return available


def _build_settings_variant(
    settings: Settings,
    horizon_name: str,
    variant_name: str,
) -> Settings:
    profile_name = settings.horizons.get_active_spec(horizon_name).feature_profile
    profile = settings.features.get_profile(profile_name)
    base_packs = [pack for pack in profile.packs if not pack.startswith("derivatives_")]

    if variant_name == "baseline":
        derivative_packs: list[str] = []
        derivatives = replace(
            settings.derivatives,
            enabled=False,
            funding=replace(settings.derivatives.funding, enabled=False),
            basis=replace(settings.derivatives.basis, enabled=False),
            book_ticker=replace(settings.derivatives.book_ticker, enabled=False),
            oi=replace(settings.derivatives.oi, enabled=False),
            options=replace(settings.derivatives.options, enabled=False),
        )
    elif variant_name == "funding":
        derivative_packs = ["derivatives_funding"]
        derivatives = replace(
            settings.derivatives,
            enabled=True,
            funding=replace(settings.derivatives.funding, enabled=True),
            basis=replace(settings.derivatives.basis, enabled=False),
            book_ticker=replace(settings.derivatives.book_ticker, enabled=False),
            oi=replace(settings.derivatives.oi, enabled=False),
            options=replace(settings.derivatives.options, enabled=False),
        )
    elif variant_name == "funding_basis":
        derivative_packs = ["derivatives_funding", "derivatives_basis"]
        derivatives = replace(
            settings.derivatives,
            enabled=True,
            funding=replace(settings.derivatives.funding, enabled=True),
            basis=replace(settings.derivatives.basis, enabled=True),
            book_ticker=replace(settings.derivatives.book_ticker, enabled=False),
            oi=replace(settings.derivatives.oi, enabled=False),
            options=replace(settings.derivatives.options, enabled=False),
        )
    elif variant_name == "funding_basis_book_ticker":
        derivative_packs = ["derivatives_funding", "derivatives_basis", "derivatives_book_ticker"]
        derivatives = replace(
            settings.derivatives,
            enabled=True,
            funding=replace(settings.derivatives.funding, enabled=True),
            basis=replace(settings.derivatives.basis, enabled=True),
            book_ticker=replace(settings.derivatives.book_ticker, enabled=True),
            oi=replace(settings.derivatives.oi, enabled=False),
            options=replace(settings.derivatives.options, enabled=False),
        )
    elif variant_name == "funding_basis_book_ticker_oi":
        derivative_packs = [
            "derivatives_funding",
            "derivatives_basis",
            "derivatives_book_ticker",
            "derivatives_oi",
        ]
        derivatives = replace(
            settings.derivatives,
            enabled=True,
            funding=replace(settings.derivatives.funding, enabled=True),
            basis=replace(settings.derivatives.basis, enabled=True),
            book_ticker=replace(settings.derivatives.book_ticker, enabled=True),
            oi=replace(settings.derivatives.oi, enabled=True),
            options=replace(settings.derivatives.options, enabled=False),
        )
    elif variant_name == "funding_basis_oi":
        derivative_packs = [
            "derivatives_funding",
            "derivatives_basis",
            "derivatives_oi",
        ]
        derivatives = replace(
            settings.derivatives,
            enabled=True,
            funding=replace(settings.derivatives.funding, enabled=True),
            basis=replace(settings.derivatives.basis, enabled=True),
            book_ticker=replace(settings.derivatives.book_ticker, enabled=False),
            oi=replace(settings.derivatives.oi, enabled=True),
            options=replace(settings.derivatives.options, enabled=False),
        )
    elif variant_name == "funding_basis_oi_options":
        derivative_packs = [
            "derivatives_funding",
            "derivatives_basis",
            "derivatives_oi",
            "derivatives_options",
        ]
        derivatives = replace(
            settings.derivatives,
            enabled=True,
            funding=replace(settings.derivatives.funding, enabled=True),
            basis=replace(settings.derivatives.basis, enabled=True),
            book_ticker=replace(settings.derivatives.book_ticker, enabled=False),
            oi=replace(settings.derivatives.oi, enabled=True),
            options=replace(settings.derivatives.options, enabled=True),
        )
    elif variant_name == "funding_basis_book_ticker_oi_options":
        derivative_packs = [
            "derivatives_funding",
            "derivatives_basis",
            "derivatives_book_ticker",
            "derivatives_oi",
            "derivatives_options",
        ]
        derivatives = replace(
            settings.derivatives,
            enabled=True,
            funding=replace(settings.derivatives.funding, enabled=True),
            basis=replace(settings.derivatives.basis, enabled=True),
            book_ticker=replace(settings.derivatives.book_ticker, enabled=True),
            oi=replace(settings.derivatives.oi, enabled=True),
            options=replace(settings.derivatives.options, enabled=True),
        )
    elif variant_name == "full":
        derivative_packs = [pack for pack in profile.packs if pack.startswith("derivatives_")]
        derivatives = settings.derivatives
    else:
        raise ValueError(f"Unknown ablation variant '{variant_name}'.")

    variant_profile = replace(profile, packs=[*base_packs, *derivative_packs])
    variant_features = FeaturesConfig(
        profiles={
            **settings.features.profiles,
            profile_name: variant_profile,
        }
    )
    return replace(settings, features=variant_features, derivatives=derivatives)


def _resolve_variants(
    requested: str,
    derivatives_frame_present: bool,
    available_derivatives_sources: set[str] | None = None,
) -> list[str]:
    if requested != "auto":
        return [name.strip() for name in requested.split(",") if name.strip()]

    if not derivatives_frame_present:
        return ["baseline"]

    available = set(available_derivatives_sources) if available_derivatives_sources is not None else {
        "funding",
        "basis",
        "book_ticker",
        "oi",
        "options",
    }
    variants = ["baseline"]

    if "funding" not in available:
        return variants

    variants.append("funding")
    if "basis" not in available:
        return variants

    variants.append("funding_basis")
    if "book_ticker" in available:
        variants.append("funding_basis_book_ticker")
        if "oi" in available:
            variants.append("funding_basis_book_ticker_oi")
            if "options" in available:
                variants.append("funding_basis_book_ticker_oi_options")
        return variants

    if "oi" in available:
        variants.append("funding_basis_oi")
        if "options" in available:
            variants.append("funding_basis_oi_options")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model families on the same BTC/USDT dataset split.")
    parser.add_argument("--input", help="Path to OHLCV input.")
    parser.add_argument("--output-dir", required=True, help="Directory for experiment outputs.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--horizon", default="5m", help="Horizon name to train.")
    parser.add_argument("--funding-input", help="Optional funding raw input override.")
    parser.add_argument("--basis-input", help="Optional basis raw input override.")
    parser.add_argument("--oi-input", help="Optional OI raw input override.")
    parser.add_argument("--options-input", help="Optional options raw input override.")
    parser.add_argument("--book-ticker-input", help="Optional bookTicker raw input override.")
    parser.add_argument(
        "--derivatives-path-mode",
        choices=["latest", "archive"],
        default=None,
        help="Override derivatives path mode. Defaults to settings.derivatives.path_mode.",
    )
    parser.add_argument(
        "--stage1-model-plugins",
        default="logistic,catboost,lightgbm_stage1",
        help="Comma-separated stage1 model plugin names to compare.",
    )
    parser.add_argument(
        "--stage2-model-plugins",
        default="logistic,catboost,lightgbm_stage2",
        help="Comma-separated stage2 model plugin names to compare.",
    )
    parser.add_argument(
        "--ablation-variants",
        default="auto",
        help="Comma-separated variants from baseline,funding,funding_basis,funding_basis_oi,funding_basis_oi_options,funding_basis_book_ticker,funding_basis_book_ticker_oi,funding_basis_book_ticker_oi_options,full. Default auto.",
    )
    parser.add_argument(
        "--validation-window-days",
        type=int,
        default=None,
        help="Validation window in days. Defaults to config value.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing experiment_report.json files instead of retraining those variant/model combinations.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip training and rebuild summary files from existing experiment_report.json files only.",
    )
    parser.add_argument("--purge-rows", type=int, default=1, help="Rows purged between train and validation.")
    args = parser.parse_args()
    if not args.summary_only and not args.input:
        parser.error("--input is required unless --summary-only is used.")

    settings = load_settings(args.config)
    validation_window_days = (
        args.validation_window_days
        if args.validation_window_days is not None
        else settings.dataset.validation_window_days
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_model_plugins = [name.strip() for name in args.stage1_model_plugins.split(",") if name.strip()]
    stage2_model_plugins = [name.strip() for name in args.stage2_model_plugins.split(",") if name.strip()]

    if args.summary_only:
        requested_variants = None if args.ablation_variants == "auto" else _resolve_variants(
            args.ablation_variants,
            derivatives_frame_present=True,
        )
        results = _collect_existing_results(
            output_dir=output_dir,
            variants=requested_variants,
            stage1_model_plugins=stage1_model_plugins,
            stage2_model_plugins=stage2_model_plugins,
        )
        if not results:
            raise ValueError("No existing experiment reports matched the requested filters.")
        summary = _write_summary(
            output_dir=output_dir,
            variants=requested_variants or _ordered_variants(results),
            validation_window_days=validation_window_days,
            results=results,
        )
        print(json.dumps(summary, indent=2))
        return

    source = _load_input(Path(args.input))
    derivatives_frame = load_derivatives_frame_from_settings(
        settings,
        funding_path=args.funding_input,
        basis_path=args.basis_input,
        oi_path=args.oi_input,
        options_path=args.options_input,
        book_ticker_path=args.book_ticker_input,
        path_mode=args.derivatives_path_mode,
    )
    results: list[dict] = []
    variants = _resolve_variants(
        args.ablation_variants,
        derivatives_frame_present=derivatives_frame is not None,
        available_derivatives_sources=_available_derivatives_sources(derivatives_frame),
    )
    for variant_name in variants:
        variant_settings: Settings | None = None
        training = None

        for stage1_plugin in stage1_model_plugins:
            for stage2_plugin in stage2_model_plugins:
                combo_name = f"{stage1_plugin}__{stage2_plugin}"
                plugin_output_dir = output_dir / variant_name / combo_name
                plugin_output_dir.mkdir(parents=True, exist_ok=True)
                report_path = plugin_output_dir / "experiment_report.json"

                if report_path.exists() and (args.skip_existing or args.summary_only):
                    results.append(_load_existing_result(report_path))
                    continue
                if args.summary_only:
                    continue

                if variant_settings is None:
                    variant_settings = _build_settings_variant(settings, args.horizon, variant_name)
                    training = build_training_frame(
                        source,
                        variant_settings,
                        horizon_name=args.horizon,
                        derivatives_frame=derivatives_frame,
                    )
                experiment_settings = replace(
                    variant_settings,
                    model=replace(
                        variant_settings.model,
                        active_plugins={"stage1": stage1_plugin, "stage2": stage2_plugin},
                    ),
                )
                started_at = time.perf_counter()
                artifacts = train_two_stage_model(
                    training,
                    experiment_settings,
                    validation_window_days=validation_window_days,
                    purge_rows=args.purge_rows,
                )
                duration_seconds = time.perf_counter() - started_at

                result = {
                    "variant": variant_name,
                    "model_plugins": {"stage1": stage1_plugin, "stage2": stage2_plugin},
                    "feature_count": len(artifacts.feature_columns),
                    "feature_counts": {
                        "stage1": len(artifacts.feature_columns),
                        "stage2": len(artifacts.stage2_feature_columns),
                    },
                    "duration_seconds": duration_seconds,
                    "stage1_threshold": artifacts.stage1_threshold,
                    "up_threshold": artifacts.up_threshold,
                    "down_threshold": artifacts.down_threshold,
                    "margin_threshold": artifacts.margin_threshold,
                    "buy_threshold": artifacts.up_threshold,
                    "base_rate": artifacts.base_rate,
                    "train_metrics": {
                        name: _select_metrics(values) for name, values in artifacts.train_metrics.items()
                    },
                    "validation_metrics": {
                        name: _select_metrics(values) for name, values in artifacts.validation_metrics.items()
                    },
                    "overfit_gap": {
                        "pnl_per_sample": artifacts.train_metrics["end_to_end"]["trade_pnl.pnl_per_sample"]
                        - artifacts.validation_metrics["end_to_end"]["trade_pnl.pnl_per_sample"],
                        "precision_sum": (
                            artifacts.train_metrics["end_to_end"].get("trade_precision_up", 0.0)
                            + artifacts.train_metrics["end_to_end"].get("trade_precision_down", 0.0)
                            - artifacts.validation_metrics["end_to_end"].get("trade_precision_up", 0.0)
                            - artifacts.validation_metrics["end_to_end"].get("trade_precision_down", 0.0)
                        ),
                        "coverage": artifacts.train_metrics["end_to_end"]["coverage_end_to_end"]
                        - artifacts.validation_metrics["end_to_end"]["coverage_end_to_end"],
                    },
                    "train_window": artifacts.train_window,
                    "validation_window": artifacts.validation_window,
                    "walk_forward_summary": artifacts.walk_forward_summary,
                    "threshold_search_best": {
                        "stage1": artifacts.threshold_search["stage1_threshold_search"]["best"],
                        "stage2": artifacts.threshold_search["stage2_threshold_search"]["best"],
                    },
                    "stage1_probability_summary": artifacts.stage1_probability_summary,
                    "derivatives": {
                        "enabled": variant_settings.derivatives.enabled,
                        "funding_enabled": variant_settings.derivatives.funding.enabled,
                        "basis_enabled": variant_settings.derivatives.basis.enabled,
                        "book_ticker_enabled": variant_settings.derivatives.book_ticker.enabled,
                        "oi_enabled": variant_settings.derivatives.oi.enabled,
                        "options_enabled": variant_settings.derivatives.options.enabled,
                        "packs": [
                            pack
                            for pack in variant_settings.features.get_profile(
                                variant_settings.horizons.get_active_spec(args.horizon).feature_profile
                            ).packs
                            if pack.startswith("derivatives_")
                        ],
                    },
                }
                results.append(result)

                artifacts.stage1_model.save(plugin_output_dir / f"{stage1_plugin}.stage1.pkl")
                artifacts.stage2_model.save(plugin_output_dir / f"{stage2_plugin}.stage2.pkl")
                artifacts.stage1_calibrator.save(
                    plugin_output_dir / f"{artifacts.stage1_calibrator.name}.stage1.pkl"
                )
                artifacts.stage2_calibrator.save(
                    plugin_output_dir / f"{artifacts.stage2_calibrator.name}.stage2.pkl"
                )
                report_path.write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )

    if not results:
        raise ValueError("No experiment results were produced. Disable --summary-only or provide existing reports.")

    summary = _write_summary(
        output_dir=output_dir,
        variants=variants,
        validation_window_days=validation_window_days,
        results=results,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
