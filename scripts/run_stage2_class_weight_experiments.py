from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import Settings, load_settings
from src.data.dataset_builder import build_training_frame
from src.data.derivatives.feature_store import load_derivatives_frame_from_settings
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.model.train import load_cached_training_split, split_training_frame, train_two_stage_model_from_split


def _load_input(path: Path):
    if path.suffix.lower() == ".csv":
        return load_ohlcv_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return load_ohlcv_parquet(path)
    if path.suffix.lower() in {".feather", ".ft"}:
        return load_ohlcv_feather(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _load_cached_split(cache_dir: Path):
    development_path = cache_dir / "development_frame.parquet"
    validation_path = cache_dir / "validation_frame.parquet"
    if not development_path.exists() or not validation_path.exists():
        raise FileNotFoundError(f"Missing cached split parquet files under: {cache_dir}")
    return load_cached_training_split(
        development_frame=pd.read_parquet(development_path),
        validation_frame=pd.read_parquet(validation_path),
    )


def _parse_weight_spec(spec: str) -> tuple[str, str | dict[int, float] | None]:
    normalized = spec.strip()
    if normalized == "balanced":
        return normalized, "balanced"
    if normalized == "none":
        return normalized, None
    parts = normalized.split(",")
    if len(parts) != 3:
        raise ValueError(
            "Custom weight spec must be 'down,flat,up', for example '1.5,1.0,1.5'."
        )
    down, flat, up = (float(value) for value in parts)
    label = f"d{down:g}_f{flat:g}_u{up:g}"
    return label, {0: down, 1: flat, 2: up}


def _replace_stage2_class_weight(settings: Settings, class_weight: str | dict[int, float] | None) -> Settings:
    model = replace(settings.model, stage2_class_weight=class_weight)
    return replace(settings, model=model)


def _result_payload(label: str, class_weight: str | dict[int, float] | None, artifacts, duration_seconds: float) -> dict[str, Any]:
    validation_stage2 = artifacts.validation_metrics["stage2"]
    validation_end = artifacts.validation_metrics["end_to_end"]
    return {
        "label": label,
        "stage2_class_weight": class_weight,
        "duration_seconds": round(float(duration_seconds), 2),
        "stage1_threshold": artifacts.stage1_threshold,
        "up_threshold": artifacts.up_threshold,
        "down_threshold": artifacts.down_threshold,
        "margin_threshold": artifacts.margin_threshold,
        "validation_stage2": {
            "macro_f1": validation_stage2.get("macro_f1", 0.0),
            "precision_up": validation_stage2.get("precision_up", 0.0),
            "precision_down": validation_stage2.get("precision_down", 0.0),
            "recall_up": validation_stage2.get("recall_up", 0.0),
            "recall_down": validation_stage2.get("recall_down", 0.0),
            "trade_pnl.pnl_per_trade": validation_stage2.get("trade_pnl.pnl_per_trade", 0.0),
            "trade_pnl.pnl_per_sample": validation_stage2.get("trade_pnl.pnl_per_sample", 0.0),
            "coverage": validation_stage2.get("coverage", 0.0),
            "stage2_trade_count": validation_stage2.get("stage2_trade_count", 0.0),
        },
        "validation_end_to_end": {
            "precision_up": validation_end.get("precision_up", 0.0),
            "precision_down": validation_end.get("precision_down", 0.0),
            "recall_up": validation_end.get("recall_up", 0.0),
            "recall_down": validation_end.get("recall_down", 0.0),
            "coverage_end_to_end": validation_end.get("coverage_end_to_end", 0.0),
            "stage2_trade_count": validation_end.get("stage2_trade_count", 0.0),
            "trade_pnl.pnl_per_trade": validation_end.get("trade_pnl.pnl_per_trade", 0.0),
            "trade_pnl.pnl_per_sample": validation_end.get("trade_pnl.pnl_per_sample", 0.0),
        },
        "threshold_search_best": {
            "stage1": artifacts.threshold_search["stage1_threshold_search"]["best"],
            "stage2": artifacts.threshold_search["stage2_threshold_search"]["best"],
        },
        "validation_window": artifacts.validation_window,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage2 Class Weight Experiments",
        "",
        f"- Horizon: `{summary['horizon']}`",
        f"- Validation window days: `{summary['validation_window_days']}`",
        "",
        "| Label | Class Weight | Prec Up | Prec Down | Recall Up | Recall Down | Coverage | Trades | PnL/Trade | PnL/Sample | Stage2 Macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in summary["results"]:
        end_metrics = entry["validation_end_to_end"]
        stage2_metrics = entry["validation_stage2"]
        lines.append(
            f"| `{entry['label']}` | `{entry['stage2_class_weight']}` | "
            f"{end_metrics['precision_up']:.6f} | {end_metrics['precision_down']:.6f} | "
            f"{end_metrics['recall_up']:.6f} | {end_metrics['recall_down']:.6f} | "
            f"{end_metrics['coverage_end_to_end']:.6f} | {end_metrics['stage2_trade_count']:.0f} | "
            f"{end_metrics['trade_pnl.pnl_per_trade']:.6f} | {end_metrics['trade_pnl.pnl_per_sample']:.6f} | "
            f"{stage2_metrics['macro_f1']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 class-weight experiments on a fixed split.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to OHLCV input.")
    input_group.add_argument("--cached-split-dir", help="Directory containing development_frame.parquet and validation_frame.parquet.")
    parser.add_argument("--output-dir", required=True, help="Directory for experiment outputs.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--horizon", default="5m", help="Horizon name to train.")
    parser.add_argument("--validation-window-days", type=int, default=None, help="Validation window size in days.")
    parser.add_argument("--purge-rows", type=int, default=1, help="Rows purged between train and validation.")
    parser.add_argument(
        "--weight-specs",
        default="balanced;1.25,1.0,1.25;1.5,1.0,1.5;2.0,1.0,2.0",
        help="Semicolon-separated specs. Each spec is 'balanced', 'none', or 'down,flat,up'.",
    )
    parser.add_argument("--funding-input", help="Optional funding raw input override.")
    parser.add_argument("--basis-input", help="Optional basis raw input override.")
    parser.add_argument("--oi-input", help="Optional OI raw input override.")
    parser.add_argument("--options-input", help="Optional options raw input override.")
    parser.add_argument("--book-ticker-input", help="Optional bookTicker raw input override.")
    parser.add_argument(
        "--derivatives-path-mode",
        choices=["latest", "archive"],
        default=None,
        help="Override derivatives path mode.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_window_days = args.validation_window_days or settings.dataset.validation_window_days

    if args.cached_split_dir:
        development, validation = _load_cached_split(Path(args.cached_split_dir))
    else:
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
        training = build_training_frame(
            source,
            settings,
            horizon_name=args.horizon,
            derivatives_frame=derivatives_frame,
        )
        development, validation = split_training_frame(
            training,
            validation_window_days=validation_window_days,
            purge_rows=args.purge_rows,
        )

    results: list[dict[str, Any]] = []
    specs = [spec.strip() for spec in args.weight_specs.split(";") if spec.strip()]
    for spec in specs:
        label, class_weight = _parse_weight_spec(spec)
        experiment_settings = _replace_stage2_class_weight(settings, class_weight)
        started_at = time.perf_counter()
        artifacts = train_two_stage_model_from_split(
            development=development,
            validation=validation,
            settings=experiment_settings,
        )
        duration_seconds = time.perf_counter() - started_at
        result = _result_payload(label, class_weight, artifacts, duration_seconds)
        results.append(result)
        (output_dir / f"{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {
        "horizon": args.horizon,
        "validation_window_days": validation_window_days,
        "results": sorted(
            results,
            key=lambda entry: (
                entry["validation_end_to_end"]["trade_pnl.pnl_per_sample"],
                entry["validation_end_to_end"]["precision_up"] + entry["validation_end_to_end"]["precision_down"],
                entry["validation_end_to_end"]["coverage_end_to_end"],
            ),
            reverse=True,
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
