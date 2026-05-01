from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.data.derivatives.feature_store import load_derivatives_frame_from_settings, resolve_derivatives_paths
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.model.rolling import build_recent_rolling_splits, summarize_binary_rolling_results
from src.model.train import load_cached_training_split, train_binary_selective_model_from_split


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _load_input(path: Path):
    if path.suffix.lower() == ".csv":
        return load_ohlcv_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return load_ohlcv_parquet(path)
    if path.suffix.lower() in {".feather", ".ft"}:
        return load_ohlcv_feather(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _parse_days_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one train-days value is required.")
    return values


def _load_training_frame(args, settings):
    if args.cached_frame:
        logging.info("Loading cached full training frame from %s", args.cached_frame)
        frame = pd.read_parquet(args.cached_frame)
        development, _ = load_cached_training_split(development_frame=frame, validation_frame=frame.tail(1).copy())
        return development

    logging.info("Loading spot input from %s", args.input)
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
    logging.info("Building full training frame for rolling validation")
    return build_training_frame(source, settings, horizon_name=args.horizon, derivatives_frame=derivatives_frame)


def _fold_result_record(*, train_days: int, fold_index: int, artifacts, window: dict, duration_seconds: float) -> dict:
    metrics = artifacts.validation_metrics
    side_guarded = artifacts.threshold_search.get("side_guarded_best", {})
    best = artifacts.threshold_search.get("best", {})
    return {
        "train_days": int(train_days),
        "fold_index": int(fold_index),
        "train_start": window["train_start"],
        "train_end": window["train_end"],
        "validation_start": window["validation_start"],
        "validation_end": window["validation_end"],
        "train_row_count": window["train_row_count"],
        "validation_row_count": window["validation_row_count"],
        "weighted": bool(artifacts.weighted),
        "t_up": float(artifacts.t_up),
        "t_down": float(artifacts.t_down),
        "balanced_precision": float(metrics.get("balanced_precision", 0.0)),
        "coverage": float(metrics.get("coverage", 0.0)),
        "precision_up": float(metrics.get("precision_up", 0.0)),
        "precision_down": float(metrics.get("precision_down", 0.0)),
        "accepted_sample_accuracy": float(metrics.get("accepted_sample_accuracy", 0.0)),
        "roc_auc": float(metrics.get("roc_auc", 0.0)),
        "share_up_predictions": float(metrics.get("share_up_predictions", 0.0)),
        "share_down_predictions": float(metrics.get("share_down_predictions", 0.0)),
        "constraint_satisfied": bool(best.get("constraint_satisfied", False)),
        "side_guardrail_constraint_satisfied": bool(side_guarded.get("constraint_satisfied", False)),
        "side_guardrail_balanced_precision": float(side_guarded.get("balanced_precision", 0.0)),
        "duration_seconds": float(duration_seconds),
    }


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Run BTC 5m binary selective rolling validation.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to OHLCV CSV/parquet input.")
    input_group.add_argument("--cached-frame", help="Full cached training frame parquet.")
    parser.add_argument("--output-dir", required=True, help="Directory for rolling validation reports.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--horizon", default="5m", help="Horizon name.")
    parser.add_argument("--train-days-list", default="30,60,90", help="Comma-separated training window sizes.")
    parser.add_argument("--validation-days", type=int, default=None, help="Validation window days.")
    parser.add_argument("--fold-count", type=int, default=3, help="Number of recent folds per train window.")
    parser.add_argument("--step-days", type=int, default=30, help="Days between validation fold ends.")
    parser.add_argument("--purge-rows", type=int, default=1, help="Rows removed between train and validation.")
    parser.add_argument("--unweighted", action="store_true", help="Disable sample weights.")
    parser.add_argument("--funding-input", help="Optional funding raw input override.")
    parser.add_argument("--basis-input", help="Optional basis raw input override.")
    parser.add_argument("--oi-input", help="Optional OI raw input override.")
    parser.add_argument("--options-input", help="Optional options raw input override.")
    parser.add_argument("--book-ticker-input", help="Optional bookTicker raw input override.")
    parser.add_argument("--derivatives-path-mode", choices=["latest", "archive"], default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_days = args.validation_days or settings.validation.validation_days
    train_days_list = _parse_days_list(args.train_days_list)
    derivatives_paths = resolve_derivatives_paths(
        settings,
        funding_path=args.funding_input,
        basis_path=args.basis_input,
        oi_path=args.oi_input,
        options_path=args.options_input,
        book_ticker_path=args.book_ticker_input,
        path_mode=args.derivatives_path_mode,
    )

    training = _load_training_frame(args, settings)
    splits = build_recent_rolling_splits(
        training,
        train_days_list=train_days_list,
        validation_days=validation_days,
        fold_count=args.fold_count,
        step_days=args.step_days,
        purge_rows=args.purge_rows,
    )
    if not splits:
        raise ValueError("No rolling splits were produced. Check data range and window settings.")

    records = []
    for split in splits:
        logging.info(
            "Training rolling fold: train_days=%s fold=%s train_rows=%s valid_rows=%s",
            split.train_days,
            split.fold_index,
            split.window["train_row_count"],
            split.window["validation_row_count"],
        )
        start = time.perf_counter()
        artifacts = train_binary_selective_model_from_split(
            development=split.development,
            validation=split.validation,
            settings=settings,
            weighted=not args.unweighted,
        )
        records.append(
            _fold_result_record(
                train_days=split.train_days,
                fold_index=split.fold_index,
                artifacts=artifacts,
                window=split.window,
                duration_seconds=time.perf_counter() - start,
            )
        )

    fold_metrics_path = output_dir / "fold_metrics.csv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "rolling_manifest.json"
    pd.DataFrame.from_records(records).to_csv(fold_metrics_path, index=False)
    summary = summarize_binary_rolling_results(records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "objective": "weighted_binary_selective_direction",
                "horizon": args.horizon,
                "weighted": not args.unweighted,
                "train_days_list": train_days_list,
                "validation_days": validation_days,
                "fold_count": args.fold_count,
                "step_days": args.step_days,
                "purge_rows": args.purge_rows,
                "data_start": str(training.frame["timestamp"].min()) if not training.frame.empty else None,
                "data_end": str(training.frame["timestamp"].max()) if not training.frame.empty else None,
                "row_count": int(len(training.frame)),
                "feature_count": int(len(training.feature_columns)),
                "derivatives_paths": derivatives_paths,
                "fold_metrics_path": fold_metrics_path.name,
                "summary_path": summary_path.name,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Rolling validation finished. Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
