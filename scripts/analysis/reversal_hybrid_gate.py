from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.reversal_hybrid import (
    apply_four_bucket_abstain_gate,
    compute_decision_metrics,
    compute_reversal_continuation_metrics,
    p_follow_from_direction_probability,
    search_four_bucket_gate,
    summarize_buckets,
)


DEFAULT_BASELINE_EXPERIMENT = Path("artifacts/data_v2/experiments/20260521_regime_reversal_second_agg_features")
DEFAULT_FOLLOW_EXPERIMENT = Path("artifacts/data_v2/experiments/20260611_first_minute_follow_reversal_weight_utility")
DEFAULT_OUTPUT_DIR = Path("artifacts/data_v2/reports/reversal_hybrid/20260611_four_bucket_abstain_gate")
DEFAULT_CONFIG_PATH = Path("experiments/configs/20260611_four_bucket_abstain_gate.yaml")
DEFAULT_CUTOFFS = [0.35, 0.40, 0.45, 0.50]
DEFAULT_BASE_BANDS = [0.03, 0.05, 0.07, 0.10]


def _load_predictions(experiment_dir: Path, split: str) -> pd.DataFrame:
    path = experiment_dir / f"{split}_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{split} predictions not found: {path}")
    return pd.read_parquet(path)


def _load_report(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"hybrid gate config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"hybrid gate config must be a mapping: {path}")
    return payload


def _prepare_hybrid_frame(baseline: pd.DataFrame, follow: pd.DataFrame) -> pd.DataFrame:
    join_keys = ["timestamp"]
    if "grid_id" in baseline.columns and "grid_id" in follow.columns:
        join_keys = ["grid_id"]
    base_columns = [
        *join_keys,
        "target",
        "p_up",
        "decision",
        "first_minute_side",
        "resolved_side",
        "post_first_minute_reversal",
        "selected_t_up",
        "selected_t_down",
    ]
    follow_columns = [*join_keys, "p_up", "first_minute_side"]
    missing_base = sorted(set(base_columns).difference(baseline.columns))
    missing_follow = sorted(set(follow_columns).difference(follow.columns))
    if missing_base:
        raise ValueError(f"baseline predictions missing required columns: {missing_base}")
    if missing_follow:
        raise ValueError(f"follow predictions missing required columns: {missing_follow}")

    base = baseline[base_columns].copy()
    base = base.rename(columns={"p_up": "p_base", "decision": "base_decision"})
    follow_work = follow[follow_columns].copy()
    follow_work["p_follow"] = p_follow_from_direction_probability(follow_work)
    follow_work = follow_work[[*join_keys, "p_follow"]]
    merged = base.merge(follow_work, on=join_keys, how="inner", validate="one_to_one")
    if len(merged) != len(base):
        raise ValueError(f"merged hybrid frame has {len(merged)} rows, expected {len(base)}.")
    return merged


def _window_summary(predictions: pd.DataFrame) -> dict[str, str | int]:
    timestamps = pd.to_datetime(predictions["timestamp"], utc=True)
    return {
        "row_count": int(len(predictions)),
        "start": str(timestamps.min()) if len(timestamps) else "",
        "end": str(timestamps.max()) if len(timestamps) else "",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _training_target(report: dict[str, Any], default: str | None = None) -> str | None:
    semantics = report.get("target_semantics")
    if isinstance(semantics, dict):
        return semantics.get("training_target", default)
    return default


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline_train_predictions = _load_predictions(args.baseline_experiment, "train")
    follow_train_predictions = _load_predictions(args.follow_experiment, "train")
    baseline_predictions = _load_predictions(args.baseline_experiment, "validation")
    follow_predictions = _load_predictions(args.follow_experiment, "validation")
    baseline_report = _load_report(args.baseline_experiment)
    follow_report = _load_report(args.follow_experiment)
    frame = _prepare_hybrid_frame(baseline_predictions, follow_predictions)
    train_frame = _prepare_hybrid_frame(baseline_train_predictions, follow_train_predictions)

    baseline_metrics = compute_decision_metrics(
        frame["target"],
        frame["p_base"],
        frame["base_decision"],
        selected_t_up=float(frame["selected_t_up"].iloc[0]),
        selected_t_down=float(frame["selected_t_down"].iloc[0]),
    )
    baseline_metrics.update(compute_reversal_continuation_metrics(frame, frame["base_decision"]))
    frontier, best = search_four_bucket_gate(
        frame,
        p_follow_cutoffs=args.p_follow_cutoffs,
        base_bands=args.base_bands,
        min_coverage=args.min_coverage,
    )
    gated = apply_four_bucket_abstain_gate(
        frame,
        p_follow_cutoff=float(best["p_follow_cutoff"]),
        base_band=float(best["base_band"]),
    )
    gated_train = apply_four_bucket_abstain_gate(
        train_frame,
        p_follow_cutoff=float(best["p_follow_cutoff"]),
        base_band=float(best["base_band"]),
    )
    train_metrics = compute_decision_metrics(
        gated_train["target"],
        gated_train["p_base"],
        gated_train["hybrid_decision"],
        selected_t_up=float(best["p_follow_cutoff"]),
        selected_t_down=float(best["base_band"]),
    )
    train_metrics.update(compute_reversal_continuation_metrics(gated_train, gated_train["hybrid_decision"]))
    hybrid_metrics = compute_decision_metrics(
        gated["target"],
        gated["p_base"],
        gated["hybrid_decision"],
        selected_t_up=float(best["p_follow_cutoff"]),
        selected_t_down=float(best["base_band"]),
    )
    hybrid_metrics.update(compute_reversal_continuation_metrics(gated, gated["hybrid_decision"]))
    bucket_summary = summarize_buckets(gated, gated["hybrid_decision"], gated["hybrid_bucket"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frontier_path = args.output_dir / "gate_frontier.csv"
    predictions_path = args.output_dir / "hybrid_validation_predictions.parquet"
    report_path = args.output_dir / "report.json"
    frontier.to_csv(frontier_path, index=False)
    gated.to_parquet(predictions_path, index=False)
    payload = {
        "experiment_id": args.output_dir.name,
        "mode": "four_bucket_abstain_only",
        "primary_metric": "validation selection_score with coverage >= min_coverage",
        "objective": {"min_coverage": float(args.min_coverage)},
        "threshold_search": {"hard_constraint": "coverage_only"},
        "config_path": str(args.config),
        "baseline_experiment": str(args.baseline_experiment),
        "follow_experiment": str(args.follow_experiment),
        "baseline_training_target": _training_target(baseline_report, "polymarket_direction"),
        "follow_training_target": _training_target(follow_report),
        "train_window": _window_summary(baseline_train_predictions),
        "validation_window": _window_summary(baseline_predictions),
        "train_metrics": train_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "validation_metrics": hybrid_metrics,
        "best_gate": best,
        "bucket_summary": bucket_summary,
        "frontier_path": str(frontier_path),
        "validation_predictions_path": str(predictions_path),
        "accepted": bool(
            hybrid_metrics["selection_score"] > baseline_metrics["selection_score"]
            and hybrid_metrics["coverage"] >= args.min_coverage
            and hybrid_metrics["accepted_sample_accuracy"] > 0.50
            and hybrid_metrics["utility"] > 0.0
            and hybrid_metrics["continuation_accepted_accuracy"] >= args.min_continuation_accuracy
        ),
        "acceptance_criteria": {
            "selection_score_gt_baseline": hybrid_metrics["selection_score"] > baseline_metrics["selection_score"],
            "utility_gt_baseline": hybrid_metrics["utility"] > baseline_metrics["utility"],
            "utility_gt_0": hybrid_metrics["utility"] > 0.0,
            "coverage_gte_min": hybrid_metrics["coverage"] >= args.min_coverage,
            "accepted_sample_accuracy_gt_050": hybrid_metrics["accepted_sample_accuracy"] > 0.50,
            "continuation_accepted_accuracy_gte_min": hybrid_metrics["continuation_accepted_accuracy"] >= args.min_continuation_accuracy,
            "reversal_accepted_accuracy_gt_min": hybrid_metrics["reversal_accepted_accuracy"] > args.min_reversal_accuracy,
        },
    }
    _write_json(report_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the 2026-06-11 four-bucket reversal hybrid abstain gate.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--baseline-experiment", type=Path)
    parser.add_argument("--follow-experiment", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-coverage", type=float)
    parser.add_argument("--min-continuation-accuracy", type=float)
    parser.add_argument("--min-reversal-accuracy", type=float)
    parser.add_argument("--p-follow-cutoffs", type=float, nargs="+")
    parser.add_argument("--base-bands", type=float, nargs="+")
    args = parser.parse_args()
    config = _load_config(args.config)
    gate = config.get("gate", {}) if isinstance(config.get("gate", {}), dict) else {}
    objective = config.get("objective", {}) if isinstance(config.get("objective", {}), dict) else {}
    acceptance = config.get("acceptance", {}) if isinstance(config.get("acceptance", {}), dict) else {}
    args.baseline_experiment = args.baseline_experiment or Path(
        config.get("baseline_experiment", DEFAULT_BASELINE_EXPERIMENT)
    )
    args.follow_experiment = args.follow_experiment or Path(config.get("follow_experiment", DEFAULT_FOLLOW_EXPERIMENT))
    args.output_dir = args.output_dir or Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    args.min_coverage = float(args.min_coverage if args.min_coverage is not None else objective.get("min_coverage", 0.70))
    args.min_continuation_accuracy = float(
        args.min_continuation_accuracy
        if args.min_continuation_accuracy is not None
        else acceptance.get("min_continuation_accepted_accuracy", 0.958)
    )
    args.min_reversal_accuracy = float(
        args.min_reversal_accuracy
        if args.min_reversal_accuracy is not None
        else acceptance.get("min_reversal_accepted_accuracy", 0.08)
    )
    args.p_follow_cutoffs = args.p_follow_cutoffs or gate.get("p_follow_cutoffs", DEFAULT_CUTOFFS)
    args.base_bands = args.base_bands or gate.get("base_bands", DEFAULT_BASE_BANDS)
    return args


def main() -> None:
    payload = run(parse_args())
    metrics = payload["validation_metrics"]
    baseline = payload["baseline_validation_metrics"]
    print(json.dumps({
        "report_path": str(Path(payload["frontier_path"]).with_name("report.json")),
        "accepted": payload["accepted"],
        "baseline_selection_score": baseline["selection_score"],
        "hybrid_selection_score": metrics["selection_score"],
        "baseline_utility": baseline["utility"],
        "hybrid_utility": metrics["utility"],
        "baseline_coverage": baseline["coverage"],
        "hybrid_coverage": metrics["coverage"],
        "best_gate": payload["best_gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
