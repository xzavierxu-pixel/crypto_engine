from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.model.evaluation import compute_classification_metrics, purged_chronological_time_window_split
from src.model.train import train_model


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model families on the same BTC/USDT dataset split.")
    parser.add_argument("--input", required=True, help="Path to OHLCV input.")
    parser.add_argument("--output-dir", required=True, help="Directory for experiment outputs.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--horizon", default="5m", help="Horizon name to train.")
    parser.add_argument(
        "--model-plugins",
        default="logistic,catboost,lightgbm",
        help="Comma-separated model plugin names to compare.",
    )
    parser.add_argument(
        "--validation-window-days",
        type=int,
        default=None,
        help="Validation window in days. Defaults to config value.",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.15, help="Calibration fraction.")
    parser.add_argument("--purge-rows", type=int, default=1, help="Rows purged between train and validation.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    validation_window_days = (
        args.validation_window_days
        if args.validation_window_days is not None
        else settings.dataset.validation_window_days
    )
    source = _load_input(Path(args.input))
    training = build_training_frame(source, settings, horizon_name=args.horizon)
    _, _, _, _, split = purged_chronological_time_window_split(
        training,
        validation_window_days=validation_window_days,
        purge_rows=args.purge_rows,
    )
    development = training.frame.iloc[split.train_slice].reset_index(drop=True)
    validation = training.frame.iloc[split.valid_slice].reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for plugin_name in [name.strip() for name in args.model_plugins.split(",") if name.strip()]:
        plugin_output_dir = output_dir / plugin_name
        plugin_output_dir.mkdir(parents=True, exist_ok=True)

        started_at = time.perf_counter()
        artifacts = train_model(
            training,
            settings,
            validation_window_days=validation_window_days,
            calibration_fraction=args.calibration_fraction,
            purge_rows=args.purge_rows,
            model_plugin_name=plugin_name,
        )
        duration_seconds = time.perf_counter() - started_at

        train_raw_probabilities = artifacts.model.predict_proba(development[artifacts.feature_columns])
        train_probabilities = artifacts.calibrator.transform(train_raw_probabilities)
        train_metrics = compute_classification_metrics(
            development[training.target_column].astype(int),
            train_probabilities,
        )

        result = {
            "model_plugin": plugin_name,
            "feature_count": len(artifacts.feature_columns),
            "duration_seconds": duration_seconds,
            "train_metrics": _select_metrics(train_metrics),
            "validation_metrics": _select_metrics(artifacts.validation_metrics),
            "overfit_gap": {
                "roc_auc": train_metrics["roc_auc"] - artifacts.validation_metrics["roc_auc"],
                "log_loss": artifacts.validation_metrics["log_loss"] - train_metrics["log_loss"],
                "accuracy": train_metrics["accuracy"] - artifacts.validation_metrics["accuracy"],
            },
            "train_window": {
                "row_count": len(development),
                "start": str(development["timestamp"].min()),
                "end": str(development["timestamp"].max()),
            },
            "validation_window": {
                "row_count": len(validation),
                "start": str(validation["timestamp"].min()),
                "end": str(validation["timestamp"].max()),
            },
        }
        results.append(result)

        artifacts.model.save(plugin_output_dir / f"{plugin_name}.pkl")
        artifacts.calibrator.save(plugin_output_dir / f"{artifacts.calibrator.name}.pkl")
        (plugin_output_dir / "experiment_report.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    results.sort(key=_rank_key, reverse=True)
    summary = {
        "ranking_metric_priority": ["roc_auc", "log_loss", "accuracy"],
        "validation_window_days": validation_window_days,
        "results": results,
        "best_model_plugin": results[0]["model_plugin"] if results else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
