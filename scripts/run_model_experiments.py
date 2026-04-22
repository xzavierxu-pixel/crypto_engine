from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import FeaturesConfig, Settings, load_settings
from src.data.dataset_builder import build_training_frame
from src.data.derivatives.feature_store import load_derivatives_frame_from_settings
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.model.train import train_model


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
    return {
        "accuracy": metrics["accuracy"],
        "log_loss": metrics["log_loss"],
        "roc_auc": metrics["roc_auc"],
        "sample_count": metrics["sample_count"],
    }


def _rank_key(result: dict) -> tuple[float, float, float]:
    return (
        result["validation_metrics"]["roc_auc"],
        -result["validation_metrics"]["log_loss"],
        result["validation_metrics"]["accuracy"],
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
                "model_plugin": result["model_plugin"],
                "feature_count": result["feature_count"],
                "validation_roc_auc": _round_metric(result["validation_metrics"]["roc_auc"]),
                "validation_log_loss": _round_metric(result["validation_metrics"]["log_loss"]),
                "validation_accuracy": _round_metric(result["validation_metrics"]["accuracy"]),
                "train_roc_auc": _round_metric(result["train_metrics"]["roc_auc"]),
                "overfit_gap_roc_auc": _round_metric(result["overfit_gap"]["roc_auc"]),
                "overfit_gap_log_loss": _round_metric(result["overfit_gap"]["log_loss"]),
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
        validation = best_result["validation_metrics"]
        train = best_result["train_metrics"]
        entry = {
            "variant": variant,
            "best_model_plugin": best_result["model_plugin"],
            "feature_count": best_result["feature_count"],
            "validation_metrics": {
                "roc_auc": _round_metric(validation["roc_auc"]),
                "log_loss": _round_metric(validation["log_loss"]),
                "accuracy": _round_metric(validation["accuracy"]),
                "sample_count": int(validation["sample_count"]),
            },
            "train_metrics": {
                "roc_auc": _round_metric(train["roc_auc"]),
                "log_loss": _round_metric(train["log_loss"]),
                "accuracy": _round_metric(train["accuracy"]),
            },
            "overfit_gap": {
                "roc_auc": _round_metric(best_result["overfit_gap"]["roc_auc"]),
                "log_loss": _round_metric(best_result["overfit_gap"]["log_loss"]),
                "accuracy": _round_metric(best_result["overfit_gap"]["accuracy"]),
            },
            "derivatives": best_result["derivatives"],
        }
        if baseline is not None:
            baseline_validation = baseline["validation_metrics"]
            entry["delta_vs_baseline"] = {
                "roc_auc": _round_metric(validation["roc_auc"] - baseline_validation["roc_auc"]),
                "log_loss": _round_metric(validation["log_loss"] - baseline_validation["log_loss"]),
                "accuracy": _round_metric(validation["accuracy"] - baseline_validation["accuracy"]),
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
            "best_model_plugin": entry["best_model_plugin"],
            "validation_roc_auc": metrics["roc_auc"],
            "validation_log_loss": metrics["log_loss"],
            "validation_accuracy": metrics["accuracy"],
        }
        if previous_entry is not None:
            previous_metrics = previous_entry["validation_metrics"]
            progression_entry["delta_vs_previous"] = {
                "roc_auc": _round_metric(metrics["roc_auc"] - previous_metrics["roc_auc"]),
                "log_loss": _round_metric(metrics["log_loss"] - previous_metrics["log_loss"]),
                "accuracy": _round_metric(metrics["accuracy"] - previous_metrics["accuracy"]),
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
            f"- Best model plugin: `{summary['best_model_plugin']}`",
            f"- Validation window days: `{summary['validation_window_days']}`",
            "",
            "## Leaderboard",
            "",
            "| Rank | Variant | Model | ROC AUC | Log Loss | Accuracy | Features | ROC AUC Gap | Duration (s) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in summary["leaderboard"]:
        lines.append(
            f"| {entry['rank']} | `{entry['variant']}` | `{entry['model_plugin']}` | "
            f"{entry['validation_roc_auc']:.6f} | {entry['validation_log_loss']:.6f} | "
            f"{entry['validation_accuracy']:.6f} | {entry['feature_count']} | "
            f"{entry['overfit_gap_roc_auc']:.6f} | {entry['duration_seconds']:.2f} |"
        )

    lines.extend(["", "## Best By Variant", ""])
    lines.append("| Variant | Best Model | ROC AUC | Log Loss | Accuracy | dAUC vs Baseline | dAcc vs Baseline |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for entry in summary["variant_summary"]:
        delta = entry.get("delta_vs_baseline", {})
        lines.append(
            f"| `{entry['variant']}` | `{entry['best_model_plugin']}` | "
            f"{entry['validation_metrics']['roc_auc']:.6f} | {entry['validation_metrics']['log_loss']:.6f} | "
            f"{entry['validation_metrics']['accuracy']:.6f} | "
            f"{float(delta.get('roc_auc', 0.0)):.6f} | {float(delta.get('accuracy', 0.0)):.6f} |"
        )

    if summary["derivatives_progression"]:
        lines.extend(["", "## Derivatives Progression", ""])
        lines.append("| Variant | Best Model | ROC AUC | dAUC vs Prev | Log Loss | dLogLoss vs Prev | Accuracy | dAcc vs Prev |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for entry in summary["derivatives_progression"]:
            delta = entry.get("delta_vs_previous", {})
            lines.append(
                f"| `{entry['variant']}` | `{entry['best_model_plugin']}` | "
                f"{entry['validation_roc_auc']:.6f} | {float(delta.get('roc_auc', 0.0)):.6f} | "
                f"{entry['validation_log_loss']:.6f} | {float(delta.get('log_loss', 0.0)):.6f} | "
                f"{entry['validation_accuracy']:.6f} | {float(delta.get('accuracy', 0.0)):.6f} |"
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
        "ranking_metric_priority": ["roc_auc", "log_loss", "accuracy"],
        "validation_window_days": validation_window_days,
        "variants": variants,
        "results": ranked_results,
        "leaderboard": leaderboard,
        "variant_summary": variant_summary,
        "derivatives_progression": derivatives_progression,
        "best_model_plugin": ranked_results[0]["model_plugin"] if ranked_results else None,
        "best_variant": ranked_results[0]["variant"] if ranked_results else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    return summary


def _collect_existing_results(
    output_dir: Path,
    variants: list[str] | None = None,
    model_plugins: list[str] | None = None,
) -> list[dict]:
    allowed_variants = set(variants) if variants is not None else None
    allowed_model_plugins = set(model_plugins) if model_plugins is not None else None
    results: list[dict] = []
    for report_path in sorted(output_dir.rglob("experiment_report.json")):
        result = _load_existing_result(report_path)
        if allowed_variants is not None and result["variant"] not in allowed_variants:
            continue
        if allowed_model_plugins is not None and result["model_plugin"] not in allowed_model_plugins:
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
        "--model-plugins",
        default="logistic,catboost,lightgbm",
        help="Comma-separated model plugin names to compare.",
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
    parser.add_argument("--calibration-fraction", type=float, default=0.15, help="Calibration fraction.")
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
    model_plugins = [name.strip() for name in args.model_plugins.split(",") if name.strip()]

    if args.summary_only:
        requested_variants = None if args.ablation_variants == "auto" else _resolve_variants(
            args.ablation_variants,
            derivatives_frame_present=True,
        )
        results = _collect_existing_results(
            output_dir=output_dir,
            variants=requested_variants,
            model_plugins=model_plugins,
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

        for plugin_name in model_plugins:
            plugin_output_dir = output_dir / variant_name / plugin_name
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
            started_at = time.perf_counter()
            artifacts = train_model(
                training,
                variant_settings,
                validation_window_days=validation_window_days,
                calibration_fraction=args.calibration_fraction,
                purge_rows=args.purge_rows,
                model_plugin_name=plugin_name,
            )
            duration_seconds = time.perf_counter() - started_at

            result = {
                "variant": variant_name,
                "model_plugin": plugin_name,
                "feature_count": len(artifacts.feature_columns),
                "duration_seconds": duration_seconds,
                "train_metrics": _select_metrics(artifacts.train_metrics),
                "validation_metrics": _select_metrics(artifacts.validation_metrics),
                "overfit_gap": {
                    "roc_auc": artifacts.train_metrics["roc_auc"] - artifacts.validation_metrics["roc_auc"],
                    "log_loss": artifacts.validation_metrics["log_loss"] - artifacts.train_metrics["log_loss"],
                    "accuracy": artifacts.train_metrics["accuracy"] - artifacts.validation_metrics["accuracy"],
                },
                "train_window": artifacts.train_window,
                "validation_window": artifacts.validation_window,
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

            artifacts.model.save(plugin_output_dir / f"{plugin_name}.pkl")
            artifacts.calibrator.save(plugin_output_dir / f"{artifacts.calibrator.name}.pkl")
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
