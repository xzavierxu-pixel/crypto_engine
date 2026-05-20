from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.model.train_model import (
    _build_data_availability_report,
    _configured_label_store_path,
    _label_metadata_report,
    _load_cached_split,
    _threshold_constraint_report,
    _with_signal_aliases,
)
from src.core.config import load_settings
from src.core.constants import DERIVATIVES_SCHEMA_VERSION
from src.core.versioning import hash_config
from src.data.dataset_builder import RAW_METADATA_FEATURE_COLUMNS
from src.model.train import load_cached_training_split, train_binary_selective_model_full_train


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_threshold_source(artifact_dir: Path) -> tuple[float, float, dict, dict]:
    manifest_path = artifact_dir / "artifact_manifest.json"
    report_path = artifact_dir / "report.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Accepted artifact manifest not found: {manifest_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Accepted artifact report not found: {report_path}")
    manifest = _read_json(manifest_path)
    report = _read_json(report_path)
    t_up = manifest.get("t_up")
    t_down = manifest.get("t_down")
    if t_up is None or t_down is None:
        raise ValueError(f"Accepted artifact does not define t_up/t_down: {manifest_path}")
    return float(t_up), float(t_down), manifest, report


def _load_full_training_from_split(split_dir: Path):
    development, validation = _load_cached_split(split_dir)
    full_frame = pd.concat([development.frame, validation.frame], ignore_index=True)
    return load_cached_training_split(development_frame=full_frame, validation_frame=full_frame)[0]


def _load_full_training_from_frame(frame_path: Path):
    frame = pd.read_parquet(frame_path)
    return load_cached_training_split(development_frame=frame, validation_frame=frame)[0]


def _write_artifacts(
    *,
    output_dir: Path,
    artifacts,
    settings,
    horizon: str,
    accepted_artifact_dir: Path,
    accepted_manifest: dict,
    accepted_report: dict,
    full_frame: pd.DataFrame,
    weighted: bool,
    config_path: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_metrics = _with_signal_aliases(artifacts.train_metrics)
    full_train_metrics = _with_signal_aliases(artifacts.validation_metrics)
    offline_validation_metrics = _with_signal_aliases(accepted_report.get("validation_metrics", {}))
    label_store_path = _configured_label_store_path(settings, horizon)
    label_metadata = _label_metadata_report(
        full_frame,
        settings,
        horizon_name=horizon,
        label_store_path=label_store_path,
    )
    model_name = settings.model.resolve_plugin(stage="binary")
    model_path = output_dir / f"{model_name}.binary.pkl"
    calibrator_path = output_dir / f"{artifacts.calibrator.name}.binary.pkl"
    manifest_path = output_dir / "artifact_manifest.json"
    report_path = output_dir / "report.json"
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

    accepted_validation_window = accepted_report.get("validation_window", {})
    threshold_source = {
        "artifact_dir": str(accepted_artifact_dir),
        "config_hash": accepted_manifest.get("config_hash"),
        "t_up": artifacts.t_up,
        "t_down": artifacts.t_down,
        "selection_data": "offline_validation",
        "validation_window": accepted_validation_window,
        "validation_metrics": offline_validation_metrics,
    }
    decision_alignment = {
        "mode": settings.decision_alignment.mode,
        "enabled": settings.decision_alignment.enabled,
        "feature_offset_minutes": settings.decision_alignment.feature_offset_minutes,
        "row_policy": settings.decision_alignment.row_policy,
        "coverage_constraint_min": float(settings.objective.min_coverage),
        "coverage_constraint_satisfied": bool(
            offline_validation_metrics.get("coverage", 0.0) >= float(settings.objective.min_coverage)
        ),
        "validation_threshold_tuned": False,
        "validation_result_optimistic": False,
        "offline_validation_artifact_required": True,
    }
    data_availability = _build_data_availability_report(artifacts.feature_columns, {})
    manifest_payload = {
        "project": settings.project.name,
        "market": settings.market.pair,
        "exchange": settings.market.exchange,
        "horizon": horizon,
        "objective": "weighted_binary_selective_direction",
        "training_mode": "online_full_train",
        "full_train_uses_all_offline_split_rows": True,
        "config_path": config_path,
        "feature_count": len(artifacts.feature_columns),
        "feature_columns": artifacts.feature_columns,
        "raw_metadata_feature_count": sum(1 for column in artifacts.feature_columns if column in RAW_METADATA_FEATURE_COLUMNS),
        "data_availability": data_availability,
        **label_metadata,
        "label_metadata": label_metadata,
        "model_plugin": model_name,
        "calibration_plugin": artifacts.calibrator.name,
        "config_hash": hash_config(settings),
        "accepted_offline_config_hash": accepted_manifest.get("config_hash"),
        "train_row_count": len(full_frame),
        "train_start": str(full_frame["timestamp"].min()) if not full_frame.empty else None,
        "train_end": str(full_frame["timestamp"].max()) if not full_frame.empty else None,
        "full_train_window": artifacts.train_window,
        "accepted_validation_window": accepted_validation_window,
        "second_level": {
            "enabled": settings.second_level.enabled,
            "feature_store_path": settings.second_level.feature_store_path,
            "feature_store_version": getattr(settings.second_level, "feature_store_version", None),
            "feature_count": sum(1 for column in artifacts.feature_columns if column.startswith("sl_")),
        },
        "weighted": weighted,
        "sample_weighting": settings.sample_weighting.__dict__,
        "sample_quality_filter": settings.dataset.sample_quality_filter,
        "derivatives": {
            "enabled": settings.derivatives.enabled,
            "schema_version": DERIVATIVES_SCHEMA_VERSION if settings.derivatives.enabled else None,
            "path_mode": settings.derivatives.path_mode,
            "funding_enabled": settings.derivatives.funding.enabled,
            "basis_enabled": settings.derivatives.basis.enabled,
            "oi_enabled": settings.derivatives.oi.enabled,
            "options_enabled": settings.derivatives.options.enabled,
            "book_ticker_enabled": settings.derivatives.book_ticker.enabled,
            "funding_path": settings.derivatives.funding.path,
            "basis_path": settings.derivatives.basis.path,
            "oi_path": settings.derivatives.oi.path,
            "options_path": settings.derivatives.options.path,
            "book_ticker_path": settings.derivatives.book_ticker.path,
        },
        "t_up": artifacts.t_up,
        "t_down": artifacts.t_down,
        "base_rate": artifacts.base_rate,
        "threshold_source": threshold_source,
        "threshold_constraint_report": _threshold_constraint_report(artifacts.threshold_search),
        "threshold_search_constraints": artifacts.threshold_search,
        "decision_alignment": decision_alignment,
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
        "train_metrics": train_metrics,
        "full_train_metrics": full_train_metrics,
        "offline_validation_metrics": offline_validation_metrics,
        "train_window": artifacts.train_window,
    }
    report_payload = {
        "training_mode": "online_full_train",
        "train_metrics": train_metrics,
        "train_window": artifacts.train_window,
        "full_train_metrics": full_train_metrics,
        "full_train_window": artifacts.train_window,
        "offline_validation_metrics": offline_validation_metrics,
        "offline_validation_window": accepted_validation_window,
        "threshold_search": artifacts.threshold_search,
        "threshold_source": threshold_source,
        "thresholds": {"t_up": artifacts.t_up, "t_down": artifacts.t_down},
        "decision_alignment": decision_alignment,
        **label_metadata,
        "label_metadata": label_metadata,
        "config_hash": manifest_payload["config_hash"],
        "feature_count": manifest_payload["feature_count"],
        "feature_columns": manifest_payload["feature_columns"],
    }
    metrics_payload = {
        "train": train_metrics,
        "full_train": full_train_metrics,
        "offline_validation": offline_validation_metrics,
        "decision_alignment": decision_alignment,
        "thresholds": {"t_up": artifacts.t_up, "t_down": artifacts.t_down},
        "threshold_source": threshold_source,
        "threshold_search": artifacts.threshold_search["best"],
        "label_metadata": label_metadata,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Retrain the accepted offline binary model on all split rows for deploy.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--cached-split-dir", help="Directory containing development_frame.parquet and validation_frame.parquet.")
    input_group.add_argument("--input-frame", help="Full cached training frame parquet.")
    parser.add_argument("--accepted-artifact-dir", required=True, help="Accepted offline split artifact directory.")
    parser.add_argument("--output-dir", default="execution_engine/deploy/baseline", help="Deploy artifact output directory.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML used by the accepted artifact.")
    parser.add_argument("--horizon", default="5m", help="Horizon name to train.")
    parser.add_argument("--unweighted", action="store_true", help="Disable sample weights.")
    parser.add_argument("--allow-config-mismatch", action="store_true", help="Allow current config hash to differ from accepted artifact.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    accepted_artifact_dir = Path(args.accepted_artifact_dir)
    t_up, t_down, accepted_manifest, accepted_report = _load_threshold_source(accepted_artifact_dir)
    current_config_hash = hash_config(settings)
    accepted_config_hash = accepted_manifest.get("config_hash")
    if accepted_config_hash and accepted_config_hash != current_config_hash and not args.allow_config_mismatch:
        raise ValueError(
            f"Config hash mismatch: accepted artifact has {accepted_config_hash}, current config has {current_config_hash}. "
            "Use the same config or pass --allow-config-mismatch."
        )

    if args.cached_split_dir:
        training = _load_full_training_from_split(Path(args.cached_split_dir))
    else:
        training = _load_full_training_from_frame(Path(args.input_frame))

    logging.info(
        "Training online full artifact: rows=%s, t_up=%.4f, t_down=%.4f, output=%s",
        len(training.frame),
        t_up,
        t_down,
        args.output_dir,
    )
    artifacts = train_binary_selective_model_full_train(
        training,
        settings,
        t_up=t_up,
        t_down=t_down,
        weighted=not args.unweighted,
    )
    _write_artifacts(
        output_dir=Path(args.output_dir),
        artifacts=artifacts,
        settings=settings,
        horizon=args.horizon,
        accepted_artifact_dir=accepted_artifact_dir,
        accepted_manifest=accepted_manifest,
        accepted_report=accepted_report,
        full_frame=training.frame,
        weighted=not args.unweighted,
        config_path=args.config,
    )
    logging.info(
        "Online full train finished: rows=%s, offline_validation_score=%.4f, full_train_score=%.4f",
        len(training.frame),
        accepted_report.get("validation_metrics", {}).get("selection_score", 0.0),
        artifacts.validation_metrics.get("selection_score", 0.0),
    )


if __name__ == "__main__":
    main()
