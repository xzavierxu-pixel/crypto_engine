from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.data.dataset_builder import TrainingFrame, compute_sample_weight, infer_feature_columns
from src.model.evaluation import compute_selective_binary_metrics
from src.model.train import train_binary_selective_model_from_split


def _slice_frame(frame: TrainingFrame, df: pd.DataFrame) -> TrainingFrame:
    return TrainingFrame(
        frame=df.reset_index(drop=True),
        feature_columns=list(frame.feature_columns),
        target_column=frame.target_column,
        sample_weight_column=frame.sample_weight_column,
    )


def _split_by_time(
    frame: TrainingFrame,
    *,
    dev_days: int,
    validation_days: int,
    purge_rows: int,
) -> tuple[TrainingFrame, TrainingFrame, dict[str, Any]]:
    df = frame.frame.sort_values("timestamp").reset_index(drop=True)
    start = pd.Timestamp(df["timestamp"].min())
    dev_end = start + pd.Timedelta(days=dev_days)
    validation_end = dev_end + pd.Timedelta(days=validation_days)

    dev = df[df["timestamp"] < dev_end]
    validation = df[(df["timestamp"] >= dev_end) & (df["timestamp"] < validation_end)]

    if purge_rows > 0:
        dev = dev.iloc[:-purge_rows] if len(dev) > purge_rows else dev.iloc[0:0]
        validation = validation.iloc[purge_rows:-purge_rows] if len(validation) > purge_rows * 2 else validation.iloc[0:0]

    if dev.empty or validation.empty:
        raise ValueError(
            "Time split produced an empty split: "
            f"dev={len(dev)}, validation={len(validation)}"
        )

    split_info = {
        "source_start": pd.Timestamp(df["timestamp"].min()).isoformat(),
        "source_end": pd.Timestamp(df["timestamp"].max()).isoformat(),
        "dev_days": dev_days,
        "validation_days": validation_days,
        "purge_rows": purge_rows,
        "development": _range_info(dev),
        "validation": _range_info(validation),
    }
    return _slice_frame(frame, dev), _slice_frame(frame, validation), split_info


def _range_info(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "start": pd.Timestamp(df["timestamp"].min()).isoformat(),
        "end": pd.Timestamp(df["timestamp"].max()).isoformat(),
        "target_mean": float(df["target"].mean()),
    }


def _predict_proba(artifacts: Any, frame: TrainingFrame) -> Any:
    raw_proba = artifacts.model.predict_proba(frame.X)
    return artifacts.calibrator.transform(raw_proba)


def _metric_dict(
    y_true: Any,
    proba: Any,
    *,
    t_up: float,
    t_down: float,
    settings: Any,
) -> dict[str, Any]:
    metrics = compute_selective_binary_metrics(y_true, proba, t_up=t_up, t_down=t_down)
    return {
        "precision_up": metrics["precision_up"],
        "precision_down": metrics["precision_down"],
        "balanced_precision": metrics["balanced_precision"],
        "up_signal_count": int(metrics["up_prediction_count"]),
        "down_signal_count": int(metrics["down_prediction_count"]),
        "total_signal_count": int(metrics["accepted_count"]),
        "signal_coverage": metrics["coverage"],
        "overall_signal_accuracy": metrics["accepted_sample_accuracy"],
        "sample_count": int(metrics["sample_count"]),
        "all_sample_accuracy": metrics["all_sample_accuracy"],
        "constraints_satisfied": _constraints_satisfied(metrics, settings=settings),
    }


def _constraints_satisfied(metrics: Any, *, settings: Any) -> bool:
    threshold_cfg = settings.threshold_search
    return (
        metrics["up_prediction_count"] >= threshold_cfg.min_up_signals
        and metrics["down_prediction_count"] >= threshold_cfg.min_down_signals
        and metrics["accepted_count"] >= threshold_cfg.min_total_signals
        and metrics["coverage"] >= settings.objective.min_coverage
    )


def _search_thresholds(settings: Any, y_true: Any, proba: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    threshold_cfg = settings.threshold_search
    objective_cfg = settings.objective
    rows: list[dict[str, Any]] = []

    up_values = _threshold_values(threshold_cfg.t_up_min, threshold_cfg.t_up_max, threshold_cfg.step)
    down_values = _threshold_values(threshold_cfg.t_down_min, threshold_cfg.t_down_max, threshold_cfg.step)
    for t_up in up_values:
        for t_down in down_values:
            metrics = compute_selective_binary_metrics(y_true, proba, t_up=t_up, t_down=t_down)
            row = {
                "t_up": float(t_up),
                "t_down": float(t_down),
                "precision_up": metrics["precision_up"],
                "precision_down": metrics["precision_down"],
                "balanced_precision": metrics["balanced_precision"],
                "up_signal_count": int(metrics["up_prediction_count"]),
                "down_signal_count": int(metrics["down_prediction_count"]),
                "total_signal_count": int(metrics["accepted_count"]),
                "signal_coverage": metrics["coverage"],
                "overall_signal_accuracy": metrics["accepted_sample_accuracy"],
                "side_count_gap": int(abs(metrics["up_prediction_count"] - metrics["down_prediction_count"])),
                "constraints_satisfied": _constraints_satisfied(metrics, settings=settings),
                "objective_min_coverage_satisfied": metrics["coverage"] >= objective_cfg.min_coverage,
            }
            rows.append(row)

    frontier = pd.DataFrame(rows)
    eligible = frontier[
        (frontier["constraints_satisfied"])
        & (frontier["objective_min_coverage_satisfied"])
        & frontier["balanced_precision"].notna()
    ].copy()
    if eligible.empty:
        fallback = frontier[frontier["balanced_precision"].notna()].copy()
        if fallback.empty:
            raise ValueError("No threshold candidate produced a valid balanced_precision.")
        ranked = fallback.sort_values(
            ["balanced_precision", "signal_coverage", "total_signal_count"],
            ascending=[False, False, False],
        )
        selected = ranked.iloc[0].to_dict()
        selected["selection_reason"] = "fallback_no_candidate_met_required_constraints"
        return selected, frontier

    eligible["threshold_distance_from_neutral"] = (
        (eligible["t_up"] - 0.5).abs() + (eligible["t_down"] - 0.5).abs()
    )
    ranked = eligible.sort_values(
        [
            "balanced_precision",
            "signal_coverage",
            "total_signal_count",
            "side_count_gap",
            "threshold_distance_from_neutral",
        ],
        ascending=[False, False, False, True, True],
    )
    selected = ranked.iloc[0].to_dict()
    selected["selection_reason"] = "max_balanced_precision_with_required_constraints"
    return selected, frontier


def _threshold_values(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = float(start)
    epsilon = step / 10.0
    while current <= float(stop) + epsilon:
        values.append(round(current, 10))
        current += float(step)
    return values


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    thresholds = report["thresholds"]
    lines = [
        "# Balanced Precision Validation Experiment",
        "",
        f"- experiment_id: `{report['experiment_id']}`",
        f"- thresholds: `t_up={thresholds['t_up']:.4f}`, `t_down={thresholds['t_down']:.4f}`",
        f"- selection_reason: `{thresholds['selection_reason']}`",
        f"- feature_count: `{report['feature_count']}`",
        f"- git_commit: `{report['git']['head']}`",
        f"- git_dirty: `{report['git']['dirty']}`",
        f"- config_path: `{report['config_copy']}`",
        f"- report_path: `{report['artifacts']['report']}`",
        f"- primary_metric: `validation.balanced_precision={metrics['validation']['balanced_precision']:.6f}`",
        f"- signal_coverage: `validation={metrics['validation']['signal_coverage']:.6f}`",
        f"- coverage_constraint_satisfied: `{metrics['validation']['signal_coverage'] >= 0.60}`",
        "",
        "## Metrics",
        "",
        "| split | balanced_precision | precision_up | precision_down | signal_coverage | up_signals | down_signals | total_signals | overall_signal_accuracy | constraints |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split in ("development", "validation"):
        m = metrics[split]
        lines.append(
            "| {split} | {balanced_precision:.6f} | {precision_up:.6f} | {precision_down:.6f} | "
            "{signal_coverage:.6f} | {up_signal_count} | {down_signal_count} | {total_signal_count} | "
            "{overall_signal_accuracy:.6f} | {constraints_satisfied} |".format(split=split, **m)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_info() -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return ""
        return result.stdout.strip()

    status = run_git(["status", "--short"])
    return {
        "head": run_git(["rev-parse", "HEAD"]) or None,
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "experiment_commit_created": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-frame", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-copy", type=Path, required=True)
    parser.add_argument("--dev-days", type=int, default=30)
    parser.add_argument("--validation-days", type=int, default=15)
    parser.add_argument("--purge-rows", type=int, default=1)
    parser.add_argument("--experiment-id", default="validation_balanced_precision")
    args = parser.parse_args()

    settings = load_settings(args.config)
    df = pd.read_parquet(args.training_frame)
    if "abs_return" in df.columns and "stage1_sample_weight" in df.columns:
        df = df.copy()
        df["stage1_sample_weight"] = compute_sample_weight(df["abs_return"], settings=settings)
    feature_columns = infer_feature_columns(df)
    sample_weight_column = "stage1_sample_weight" if "stage1_sample_weight" in df.columns else None
    frame = TrainingFrame(
        frame=df,
        feature_columns=feature_columns,
        target_column="target",
        sample_weight_column=sample_weight_column,
    )

    development, validation, split_info = _split_by_time(
        frame,
        dev_days=args.dev_days,
        validation_days=args.validation_days,
        purge_rows=args.purge_rows,
    )

    artifacts = train_binary_selective_model_from_split(
        development=development,
        validation=validation,
        settings=settings,
        weighted=sample_weight_column is not None and settings.sample_weighting.enabled,
    )
    validation_proba = _predict_proba(artifacts, validation)
    selected, frontier = _search_thresholds(settings, validation.y, validation_proba)
    t_up = float(selected["t_up"])
    t_down = float(selected["t_down"])

    metrics = {
        "development": _metric_dict(
            development.y,
            _predict_proba(artifacts, development),
            t_up=t_up,
            t_down=t_down,
            settings=settings,
        ),
        "validation": _metric_dict(
            validation.y,
            validation_proba,
            t_up=t_up,
            t_down=t_down,
            settings=settings,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.config_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.config_copy)

    frontier_path = args.output_dir / "threshold_frontier.csv"
    frontier.to_csv(frontier_path, index=False)
    feature_set_path = args.output_dir / "feature_set.json"
    feature_set_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    model_path = args.output_dir / "model_artifact.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(artifacts, fh)

    if not artifacts.feature_importance.empty:
        artifacts.feature_importance.to_csv(args.output_dir / "feature_importance.csv", index=False)

    report_path = args.output_dir / "report.json"
    report = {
        "experiment_id": args.experiment_id,
        "training_frame": str(args.training_frame),
        "config_source": str(args.config),
        "config_copy": str(args.config_copy),
        "feature_count": len(feature_columns),
        "model_settings": asdict(settings.model),
        "objective_settings": asdict(settings.objective),
        "threshold_search_settings": asdict(settings.threshold_search),
        "required_constraints": {
            "min_up_signals": settings.threshold_search.min_up_signals,
            "min_down_signals": settings.threshold_search.min_down_signals,
            "min_total_signals": settings.threshold_search.min_total_signals,
            "min_signal_coverage": settings.objective.min_coverage,
        },
        "split_info": split_info,
        "thresholds": {
            "t_up": t_up,
            "t_down": t_down,
            "selection_reason": selected["selection_reason"],
            "validation_selected_row": selected,
        },
        "metrics": metrics,
        "git": _git_info(),
        "artifacts": {
            "report": str(report_path),
            "threshold_frontier": str(frontier_path),
            "feature_set": str(feature_set_path),
            "model_artifact": str(model_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary(args.output_dir / "summary.md", report)

    print(json.dumps({"report": str(report_path), "metrics": metrics, "thresholds": report["thresholds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
