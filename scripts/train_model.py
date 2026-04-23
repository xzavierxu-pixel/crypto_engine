from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.core.versioning import hash_config
from src.data.dataset_builder import build_training_frame
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.model.evaluation import build_threshold_scan
from src.model.train import train_model


def _load_input(path: Path):
    if path.suffix.lower() == ".csv":
        return load_ohlcv_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return load_ohlcv_parquet(path)
    if path.suffix.lower() in {".feather", ".ft"}:
        return load_ohlcv_feather(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _model_input_frame(frame, feature_columns: list[str], target_column: str, sample_weight_column: str | None):
    columns = ["timestamp", target_column, *feature_columns]
    if sample_weight_column and sample_weight_column in frame.columns:
        columns.insert(2, sample_weight_column)
    return frame.loc[:, columns].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline BTC Polymarket model.")
    parser.add_argument("--input", required=True, help="Path to OHLCV CSV or parquet input.")
    parser.add_argument("--output-dir", required=True, help="Directory for model artifacts.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--horizon", default="5m", help="Horizon name to train.")
    parser.add_argument("--model-plugin", default=None, help="Optional model plugin override.")
    parser.add_argument(
        "--validation-window-days",
        type=int,
        default=None,
        help="Validation window size in days. Defaults to the value in config/settings.yaml.",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.15, help="Fraction of the development window reserved for probability calibration.")
    parser.add_argument("--purge-rows", type=int, default=1, help="Grid rows removed between train and validation splits.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    source = _load_input(Path(args.input))
    training = build_training_frame(source, settings, horizon_name=args.horizon)
    validation_window_days = (
        args.validation_window_days
        if args.validation_window_days is not None
        else settings.dataset.validation_window_days
    )
    artifacts = train_model(
        training,
        settings,
        validation_window_days=validation_window_days,
        calibration_fraction=args.calibration_fraction,
        purge_rows=args.purge_rows,
        model_plugin_name=args.model_plugin,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_plugin_name = args.model_plugin or settings.model.active_plugin
    model_path = output_dir / f"{model_plugin_name}.pkl"
    calibrator_path = output_dir / f"{artifacts.calibrator.name}.pkl"
    report_path = output_dir / "training_report.json"
    artifacts.model.save(model_path)
    artifacts.calibrator.save(calibrator_path)
    threshold_scan = build_threshold_scan(
        artifacts.validation_frame[training.target_column].astype(int),
        artifacts.validation_probabilities,
    )
    best_threshold_row = threshold_scan.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    threshold_scan_path = output_dir / "threshold_scan_validation.csv"
    train_processed_path = output_dir / "train_model_input.parquet"
    valid_processed_path = output_dir / "valid_model_input.parquet"
    threshold_scan.to_csv(threshold_scan_path, index=False)
    _model_input_frame(
        artifacts.development_frame,
        artifacts.feature_columns,
        training.target_column,
        training.sample_weight_column,
    ).to_parquet(train_processed_path, index=False)
    _model_input_frame(
        artifacts.validation_frame,
        artifacts.feature_columns,
        training.target_column,
        training.sample_weight_column,
    ).to_parquet(valid_processed_path, index=False)
    report_payload = {
        "project": settings.project.name,
        "market": settings.market.pair,
        "exchange": settings.market.exchange,
        "horizon": args.horizon,
        "feature_columns": artifacts.feature_columns,
        "feature_count": len(artifacts.feature_columns),
        "model_plugin": model_plugin_name,
        "calibration_plugin": artifacts.calibrator.name,
        "config_hash": hash_config(settings),
        "train_row_count": len(training.frame),
        "train_start": str(training.frame["timestamp"].min()) if not training.frame.empty else None,
        "train_end": str(training.frame["timestamp"].max()) if not training.frame.empty else None,
        "sample_quality_filter": settings.dataset.sample_quality_filter,
        "sample_weighting": settings.dataset.sample_weighting,
        "sample_weight_summary": {
            "enabled": training.sample_weight is not None,
            "min": float(training.sample_weight.min()) if training.sample_weight is not None else None,
            "max": float(training.sample_weight.max()) if training.sample_weight is not None else None,
            "mean": float(training.sample_weight.mean()) if training.sample_weight is not None else None,
        },
        "validation_window_days": validation_window_days,
        "calibration_fraction": args.calibration_fraction,
        "purge_rows": args.purge_rows,
        "train_metrics": artifacts.train_metrics,
        "train_window": artifacts.train_window,
        "validation_window": artifacts.validation_window,
        "threshold_scan": {
            "path": str(threshold_scan_path.resolve()),
            "best_by_f1": {
                "threshold": float(best_threshold_row["threshold"]),
                "precision": float(best_threshold_row["precision"]),
                "recall": float(best_threshold_row["recall"]),
                "f1": float(best_threshold_row["f1"]),
                "accuracy": float(best_threshold_row["accuracy"]),
                "balanced_accuracy": float(best_threshold_row["balanced_accuracy"]),
                "predicted_positive_count": int(best_threshold_row["predicted_positive_count"]),
                "predicted_positive_rate": float(best_threshold_row["predicted_positive_rate"]),
            },
        },
        "processed_data": {
            "train_model_input_path": str(train_processed_path.resolve()),
            "valid_model_input_path": str(valid_processed_path.resolve()),
            "columns": ["timestamp", training.target_column, training.sample_weight_column, *artifacts.feature_columns]
            if training.sample_weight_column
            else ["timestamp", training.target_column, *artifacts.feature_columns],
        },
        "raw_validation_metrics": artifacts.raw_validation_metrics,
        "validation_metrics": artifacts.validation_metrics,
        "walk_forward_summary": artifacts.walk_forward_summary,
        "walk_forward_folds": [
            {
                "fold_index": fold.fold_index,
                "train_start": fold.split.train_start,
                "train_end": fold.split.train_end,
                "valid_start": fold.split.valid_start,
                "valid_end": fold.split.valid_end,
                "purge_rows": fold.split.purge_rows,
                "metrics": fold.metrics,
            }
            for fold in artifacts.walk_forward_results
        ],
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
