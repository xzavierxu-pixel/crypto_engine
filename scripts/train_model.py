from __future__ import annotations

import argparse
import json
import logging
import subprocess
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
from src.data.dataset_builder import RAW_METADATA_FEATURE_COLUMNS
from src.data.derivatives.feature_store import (
    load_derivatives_frame_from_settings,
    resolve_derivatives_paths,
)
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.data.second_level_features import build_second_level_feature_frame, load_second_level_frame
from src.model.train import (
    load_cached_training_split,
    split_recent_train_validation_frame,
    train_binary_selective_model_from_split,
)


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


def _run_split_data_quality_report(output_dir: Path) -> None:
    script_path = REPO_ROOT / "src" / "quality check" / "data_quality_report.py"
    if not script_path.exists():
        logging.warning("Data quality report script not found at %s; skipping split DQC.", script_path)
        return
    development_path = output_dir / "development_frame.parquet"
    validation_path = output_dir / "validation_frame.parquet"
    dqc_output_dir = output_dir / "data_quality"
    command = [
        sys.executable,
        str(script_path),
        "--train",
        str(development_path),
        "--valid",
        str(validation_path),
        "--output-dir",
        str(dqc_output_dir),
    ]
    logging.info("Generating split data quality report in %s", dqc_output_dir)
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def _feature_presence(feature_columns: list[str], prefixes: tuple[str, ...], names: tuple[str, ...] = ()) -> bool:
    feature_set = set(feature_columns)
    return any(column in feature_set for column in names) or any(
        column.startswith(prefix) for column in feature_columns for prefix in prefixes
    )


def _build_data_availability_report(feature_columns: list[str], derivatives_paths: dict) -> dict:
    return {
        "funding_features_available": _feature_presence(feature_columns, ("funding_",), ("funding_abs",)),
        "basis_features_available": _feature_presence(feature_columns, ("basis_",), ("premium_index",)),
        "book_ticker_features_available": _feature_presence(
            feature_columns,
            ("book_",),
            ("spread_bps", "mid_price", "microprice", "bid_ask_qty_imbalance"),
        ),
        "flow_proxy_features_available": _feature_presence(
            feature_columns,
            ("taker_",),
            ("signed_dollar_flow", "positive_taker_imbalance", "negative_taker_imbalance"),
        ),
        "funding_path_exists": bool(derivatives_paths.get("funding_path") and Path(derivatives_paths["funding_path"]).exists()),
        "basis_path_exists": bool(derivatives_paths.get("basis_path") and Path(derivatives_paths["basis_path"]).exists()),
        "book_ticker_path_exists": bool(
            derivatives_paths.get("book_ticker_path") and Path(derivatives_paths["book_ticker_path"]).exists()
        ),
    }


def _threshold_constraint_report(threshold_search: dict) -> dict:
    best = threshold_search.get("best", {})
    side_guarded = threshold_search.get("side_guarded_best", {})
    return {
        "threshold_constraint_satisfied": bool(best.get("constraint_satisfied", False)),
        "threshold_fallback_reason": best.get("fallback_reason"),
        "side_guardrail_constraint_satisfied": bool(side_guarded.get("constraint_satisfied", False)),
        "side_guardrail_fallback_reason": side_guarded.get("fallback_reason"),
        "side_guardrail_t_up": side_guarded.get("t_up"),
        "side_guardrail_t_down": side_guarded.get("t_down"),
        "side_guardrail_balanced_precision": side_guarded.get("balanced_precision"),
        "side_guardrail_coverage": side_guarded.get("coverage"),
    }


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Train the BTC 5m weighted binary selective direction model.")
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
    parser.add_argument("--agg-trades-input", help="Optional second-level aggregate trades input.")
    parser.add_argument("--trades-input", help="Optional second-level raw trades input.")
    parser.add_argument("--second-book-ticker-input", help="Optional second-level bookTicker input.")
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
        help="Validation window size in days. Defaults to validation.validation_days in config/settings.yaml.",
    )
    parser.add_argument(
        "--train-window-days",
        type=int,
        default=None,
        help="Training window size in days. Defaults to validation.train_days in config/settings.yaml.",
    )
    parser.add_argument("--unweighted", action="store_true", help="Disable sample weights for an unweighted control run.")
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
        else settings.validation.validation_days
    )
    train_window_days = (
        args.train_window_days
        if args.train_window_days is not None
        else settings.validation.train_days
    )

    if args.cached_split_dir:
        development, validation = _load_cached_split(Path(args.cached_split_dir))
        training_row_count = len(development.frame) + len(validation.frame)
        train_start = str(development.frame["timestamp"].min()) if not development.frame.empty else None
        train_end = str(validation.frame["timestamp"].max()) if not validation.frame.empty else None
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
        second_level_features_frame = None
        if args.agg_trades_input or args.trades_input or args.second_book_ticker_input:
            trade_frames = []
            if args.agg_trades_input:
                logging.info("Loading second-level aggTrades from %s", args.agg_trades_input)
                trade_frames.append(load_second_level_frame(args.agg_trades_input))
            if args.trades_input:
                logging.info("Loading second-level trades from %s", args.trades_input)
                trade_frames.append(load_second_level_frame(args.trades_input))
            book_frame = None
            if args.second_book_ticker_input:
                logging.info("Loading second-level bookTicker from %s", args.second_book_ticker_input)
                book_frame = load_second_level_frame(args.second_book_ticker_input)
            trades_frame = pd.concat(trade_frames, ignore_index=True) if trade_frames else None
            logging.info("Aggregating second-level features to decision timestamps")
            second_level_features_frame = build_second_level_feature_frame(
                source,
                trades_frame=trades_frame,
                book_frame=book_frame,
            )
        logging.info("Building training frame for horizon=%s", args.horizon)
        training = build_training_frame(
            source,
            settings,
            horizon_name=args.horizon,
            derivatives_frame=derivatives_frame,
            second_level_features_frame=second_level_features_frame,
        )
        development, validation = split_recent_train_validation_frame(
            training,
            train_days=train_window_days,
            validation_days=validation_window_days,
            purge_rows=args.purge_rows,
        )
        _write_cached_split(output_dir, development, validation)
        _run_split_data_quality_report(output_dir)
        training_row_count = len(training.frame)
        train_start = str(training.frame["timestamp"].min()) if not training.frame.empty else None
        train_end = str(training.frame["timestamp"].max()) if not training.frame.empty else None
        logging.info(
            "Training binary selective model: rows=%s, train_window_days=%s, validation_window_days=%s, purge_rows=%s",
            len(training.frame),
            train_window_days,
            validation_window_days,
            args.purge_rows,
        )

    logging.info("Starting asymmetric p_up threshold search")
    artifacts = train_binary_selective_model_from_split(
        development=development,
        validation=validation,
        settings=settings,
        weighted=not args.unweighted,
    )

    logging.info("Writing artifacts to %s", output_dir)
    model_name = settings.model.resolve_plugin(stage="binary")
    model_path = output_dir / f"{model_name}.binary.pkl"
    calibrator_path = output_dir / f"{artifacts.calibrator.name}.binary.pkl"
    manifest_path = output_dir / "artifact_manifest.json"
    metrics_path = output_dir / "metrics.json"
    threshold_search_path = output_dir / "threshold_search.json"
    threshold_frontier_path = output_dir / "threshold_frontier.csv"
    boundary_slices_path = output_dir / "boundary_slices.csv"
    regime_slices_path = output_dir / "regime_slices.csv"
    feature_importance_path = output_dir / "feature_importance.csv"
    probability_deciles_path = output_dir / "probability_deciles.csv"
    false_up_slices_path = output_dir / "false_up_slices.csv"
    false_down_slices_path = output_dir / "false_down_slices.csv"
    probability_reference_path = output_dir / "probability_reference.json"
    artifacts.model.save(model_path)
    artifacts.calibrator.save(calibrator_path)
    threshold_search_path.write_text(json.dumps(artifacts.threshold_search, indent=2), encoding="utf-8")
    artifacts.threshold_frontier.to_csv(threshold_frontier_path, index=False)
    artifacts.boundary_slices.to_csv(boundary_slices_path, index=False)
    artifacts.regime_slices.to_csv(regime_slices_path, index=False)
    artifacts.feature_importance.to_csv(feature_importance_path, index=False)
    artifacts.probability_deciles.to_csv(probability_deciles_path, index=False)
    artifacts.false_up_slices.to_csv(false_up_slices_path, index=False)
    artifacts.false_down_slices.to_csv(false_down_slices_path, index=False)
    probability_reference_path.write_text(json.dumps(artifacts.probability_reference, indent=2), encoding="utf-8")
    metrics_payload = {
        "train": artifacts.train_metrics,
        "validation": artifacts.validation_metrics,
        "thresholds": {"t_up": artifacts.t_up, "t_down": artifacts.t_down},
        "threshold_search": artifacts.threshold_search["best"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    manifest_payload = {
        "project": settings.project.name,
        "market": settings.market.pair,
        "exchange": settings.market.exchange,
        "horizon": args.horizon,
        "objective": "weighted_binary_selective_direction",
        "feature_count": len(artifacts.feature_columns),
        "feature_columns": artifacts.feature_columns,
        "raw_metadata_feature_count": sum(1 for column in artifacts.feature_columns if column in RAW_METADATA_FEATURE_COLUMNS),
        "data_availability": _build_data_availability_report(artifacts.feature_columns, derivatives_paths),
        "model_plugin": model_name,
        "calibration_plugin": artifacts.calibrator.name,
        "config_hash": hash_config(settings),
        "train_row_count": training_row_count,
        "train_start": train_start,
        "train_end": train_end,
        "second_level": {
            "agg_trades_input": args.agg_trades_input,
            "trades_input": args.trades_input,
            "book_ticker_input": args.second_book_ticker_input,
            "feature_count": sum(1 for column in artifacts.feature_columns if column.startswith("sl_")),
        },
        "weighted": artifacts.weighted,
        "sample_weighting": settings.sample_weighting.__dict__,
        "sample_quality_filter": settings.dataset.sample_quality_filter,
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
        "train_window_days": train_window_days,
        "validation_window_days": validation_window_days,
        "purge_rows": args.purge_rows,
        "t_up": artifacts.t_up,
        "t_down": artifacts.t_down,
        "base_rate": artifacts.base_rate,
        "threshold_constraint_report": _threshold_constraint_report(artifacts.threshold_search),
        "threshold_search_constraints": artifacts.threshold_search,
        "metrics_path": metrics_path.name,
        "threshold_search_path": threshold_search_path.name,
        "threshold_frontier_path": threshold_frontier_path.name,
        "boundary_slices_path": boundary_slices_path.name,
        "regime_slices_path": regime_slices_path.name,
        "feature_importance_path": feature_importance_path.name,
        "probability_deciles_path": probability_deciles_path.name,
        "false_up_slices_path": false_up_slices_path.name,
        "false_down_slices_path": false_down_slices_path.name,
        "probability_summary": artifacts.probability_summary,
        "probability_reference_path": probability_reference_path.name,
        "train_metrics": artifacts.train_metrics,
        "train_window": artifacts.train_window,
        "validation_window": artifacts.validation_window,
        "validation_metrics": artifacts.validation_metrics,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    logging.info(
        "Training finished: t_up=%.4f, t_down=%.4f, coverage=%.4f, balanced_precision=%.4f, precision_up=%.4f, precision_down=%.4f",
        artifacts.t_up,
        artifacts.t_down,
        artifacts.validation_metrics.get("coverage", 0.0),
        artifacts.validation_metrics.get("balanced_precision", 0.0),
        artifacts.validation_metrics.get("precision_up", 0.0),
        artifacts.validation_metrics.get("precision_down", 0.0),
    )


if __name__ == "__main__":
    main()
