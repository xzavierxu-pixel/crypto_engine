from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.core.constants import DERIVATIVES_SCHEMA_VERSION
from src.core.versioning import hash_config
from src.data.dataset_builder import build_training_frame
from src.data.derivatives.feature_store import (
    load_derivatives_frame_from_settings,
    resolve_derivatives_paths,
)
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


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _load_cached_split(cache_dir: Path):
    development_path = cache_dir / "development_frame.parquet"
    validation_path = cache_dir / "validation_frame.parquet"
    if not development_path.exists():
        raise FileNotFoundError(f"Cached development split not found: {development_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Cached validation split not found: {validation_path}")
    logging.info("Loading cached split data from %s", cache_dir)
    return load_cached_training_split(
        development_frame=pd.read_parquet(development_path),
        validation_frame=pd.read_parquet(validation_path),
    )


def _write_cached_split(output_dir: Path, development, validation) -> None:
    development_path = output_dir / "development_frame.parquet"
    validation_path = output_dir / "validation_frame.parquet"
    development.frame.to_parquet(development_path, index=False)
    validation.frame.to_parquet(validation_path, index=False)
    logging.info("Cached split data written to %s and %s", development_path, validation_path)


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Train the two-stage BTC Polymarket model.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to OHLCV CSV or parquet input.")
    input_group.add_argument(
        "--cached-split-dir",
        help="Directory containing development_frame.parquet and validation_frame.parquet.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for model artifacts.")
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
        "--validation-window-days",
        type=int,
        default=None,
        help="Validation window size in days. Defaults to the value in config/settings.yaml.",
    )
    parser.add_argument("--purge-rows", type=int, default=1, help="Grid rows removed between train and validation splits.")
    args = parser.parse_args()
    logging.info("Loading settings from %s", args.config)

    settings = load_settings(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Resolving derivatives inputs")
    derivatives_paths = resolve_derivatives_paths(
        settings,
        funding_path=args.funding_input,
        basis_path=args.basis_input,
        oi_path=args.oi_input,
        options_path=args.options_input,
        book_ticker_path=args.book_ticker_input,
        path_mode=args.derivatives_path_mode,
    )
    validation_window_days = (
        args.validation_window_days
        if args.validation_window_days is not None
        else settings.dataset.validation_window_days
    )

    if args.cached_split_dir:
        development, validation = _load_cached_split(Path(args.cached_split_dir))
        training_row_count = len(development.frame) + len(validation.frame)
        train_start = str(development.frame["timestamp"].min()) if not development.frame.empty else None
        train_end = str(validation.frame["timestamp"].max()) if not validation.frame.empty else None
        sample_weight = development.sample_weight
        logging.info(
            "Training from cached split: development_rows=%s, validation_rows=%s",
            len(development.frame),
            len(validation.frame),
        )
    else:
        logging.info("Loading input data from %s", args.input)
        source = _load_input(Path(args.input))
        logging.info("Loading derivatives frame")
        derivatives_frame = load_derivatives_frame_from_settings(
            settings,
            funding_path=args.funding_input,
            basis_path=args.basis_input,
            oi_path=args.oi_input,
            options_path=args.options_input,
            book_ticker_path=args.book_ticker_input,
            path_mode=args.derivatives_path_mode,
        )
        logging.info("Building training frame for horizon=%s", args.horizon)
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
        _write_cached_split(output_dir, development, validation)
        training_row_count = len(training.frame)
        train_start = str(training.frame["timestamp"].min()) if not training.frame.empty else None
        train_end = str(training.frame["timestamp"].max()) if not training.frame.empty else None
        sample_weight = training.sample_weight
        logging.info(
            "Training two-stage model: rows=%s, validation_window_days=%s, purge_rows=%s",
            len(training.frame),
            validation_window_days,
            args.purge_rows,
        )

    logging.info("Starting validation-based threshold search with configured coverage constraint")
    artifacts = train_two_stage_model_from_split(
        development=development,
        validation=validation,
        settings=settings,
    )

    logging.info("Writing artifacts to %s", output_dir)
    stage1_model_path = output_dir / f"{settings.model.resolve_plugin(stage='stage1')}.stage1.pkl"
    stage2_model_path = output_dir / f"{settings.model.resolve_plugin(stage='stage2')}.stage2.pkl"
    stage1_calibrator_path = output_dir / f"{artifacts.stage1_calibrator.name}.stage1.pkl"
    stage2_calibrator_path = output_dir / f"{artifacts.stage2_calibrator.name}.stage2.pkl"
    report_path = output_dir / "training_report.json"
    threshold_search_path = output_dir / "threshold_search.json"
    stage1_probability_reference_path = output_dir / "stage1_probability_reference.json"
    artifacts.stage1_model.save(stage1_model_path)
    artifacts.stage2_model.save(stage2_model_path)
    artifacts.stage1_calibrator.save(stage1_calibrator_path)
    artifacts.stage2_calibrator.save(stage2_calibrator_path)
    threshold_search_path.write_text(json.dumps(artifacts.threshold_search, indent=2), encoding="utf-8")
    stage1_probability_reference_path.write_text(
        json.dumps(artifacts.stage1_probability_reference, indent=2),
        encoding="utf-8",
    )
    report_payload = {
        "project": settings.project.name,
        "market": settings.market.pair,
        "exchange": settings.market.exchange,
        "horizon": args.horizon,
        "feature_counts": {
            "stage1": len(artifacts.feature_columns),
            "stage2": len(artifacts.stage2_feature_columns),
        },
        "stage2_feature_columns": artifacts.stage2_feature_columns,
        "model_plugins": {
            "stage1": settings.model.resolve_plugin(stage="stage1"),
            "stage2": settings.model.resolve_plugin(stage="stage2"),
        },
        "calibration_plugins": {
            "stage1": artifacts.stage1_calibrator.name,
            "stage2": artifacts.stage2_calibrator.name,
        },
        "config_hash": hash_config(settings),
        "train_row_count": training_row_count,
        "train_start": train_start,
        "train_end": train_end,
        "sample_quality_filter": settings.dataset.sample_quality_filter,
        "sample_weighting": settings.dataset.sample_weighting,
        "sample_weight_summary": {
            "enabled": sample_weight is not None,
            "min": float(sample_weight.min()) if sample_weight is not None else None,
            "max": float(sample_weight.max()) if sample_weight is not None else None,
            "mean": float(sample_weight.mean()) if sample_weight is not None else None,
        },
        "derivatives": {
            "enabled": settings.derivatives.enabled,
            "schema_version": DERIVATIVES_SCHEMA_VERSION if settings.derivatives.enabled else None,
            "path_mode": derivatives_paths["path_mode"],
            "funding_enabled": settings.derivatives.funding.enabled,
            "basis_enabled": settings.derivatives.basis.enabled,
            "oi_enabled": settings.derivatives.oi.enabled,
            "options_enabled": settings.derivatives.options.enabled,
            "book_ticker_enabled": settings.derivatives.book_ticker.enabled,
            "funding_path": derivatives_paths["funding_path"],
            "basis_path": derivatives_paths["basis_path"],
            "oi_path": derivatives_paths["oi_path"],
            "options_path": derivatives_paths["options_path"],
            "book_ticker_path": derivatives_paths["book_ticker_path"],
        },
        "validation_window_days": validation_window_days,
        "purge_rows": args.purge_rows,
        "stage1_threshold": artifacts.stage1_threshold,
        "buy_threshold": artifacts.buy_threshold,
        "base_rate": artifacts.base_rate,
        "threshold_selection_data": artifacts.threshold_search.get("selection_data"),
        "threshold_search_constraints": {
            "stage1_coverage_constraint": artifacts.threshold_search.get("stage1_coverage_constraint"),
            "constraint_applied": artifacts.threshold_search.get("constraint_applied"),
            "constraint_satisfied": artifacts.threshold_search.get("constraint_satisfied"),
            "fallback_reason": artifacts.threshold_search.get("fallback_reason"),
        },
        "threshold_search_path": threshold_search_path.name,
        "stage1_probability_summary": artifacts.stage1_probability_summary,
        "stage1_probability_reference_path": stage1_probability_reference_path.name,
        "train_metrics": artifacts.train_metrics,
        "train_window": artifacts.train_window,
        "validation_window": artifacts.validation_window,
        "validation_metrics": artifacts.validation_metrics,
        "walk_forward_summary": artifacts.walk_forward_summary,
        "walk_forward_folds": artifacts.walk_forward_fold_details,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    logging.info(
        "Training finished: stage1_threshold=%.4f, buy_threshold=%.4f, base_rate=%.4f, stage1_valid_precision=%.4f, stage1_valid_recall=%.4f, stage2_valid_accuracy=%.4f",
        artifacts.stage1_threshold,
        artifacts.buy_threshold,
        artifacts.base_rate,
        artifacts.validation_metrics["stage1"].get("precision", 0.0),
        artifacts.validation_metrics["stage1"].get("recall", 0.0),
        artifacts.validation_metrics["stage2"].get("accuracy", 0.0),
    )


if __name__ == "__main__":
    main()
