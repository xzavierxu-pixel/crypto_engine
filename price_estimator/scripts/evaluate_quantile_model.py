#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_estimator_common import load_config, pinball_loss, resolve_path, write_json


QUANTILE_COLS = {0.70: "pred_q70", 0.80: "pred_q80", 0.90: "pred_q90"}


def quantile_metrics(df: pd.DataFrame, prefix: str = "", epsilon: float = 0.0) -> dict[str, float]:
    y = df["target_raw"].to_numpy(dtype=float)
    out: dict[str, float] = {"sample_count": float(len(df))}
    for alpha, col in QUANTILE_COLS.items():
        pred = df[col].to_numpy(dtype=float)
        key = f"q{int(alpha * 100)}"
        gap = pred - y
        violation = np.maximum(y + epsilon - pred, 0.0)
        out[f"{prefix}coverage_{key}"] = float(np.mean(y + epsilon <= pred)) if len(y) else float("nan")
        out[f"{prefix}pinball_{key}"] = pinball_loss(y, pred, alpha) if len(y) else float("nan")
        out[f"{prefix}mean_pred_{key}"] = float(np.mean(pred)) if len(y) else float("nan")
        out[f"{prefix}mean_gap_{key}"] = float(np.mean(gap)) if len(y) else float("nan")
        out[f"{prefix}median_gap_{key}"] = float(np.median(gap)) if len(y) else float("nan")
        gap_p05 = float(np.quantile(gap, 0.05)) if len(y) else float("nan")
        gap_p95 = float(np.quantile(gap, 0.95)) if len(y) else float("nan")
        out[f"{prefix}gap_p05_{key}"] = gap_p05
        out[f"{prefix}gap_p95_{key}"] = gap_p95
        out[f"{prefix}gap_p95_p05_range_{key}"] = gap_p95 - gap_p05 if len(y) else float("nan")
        out[f"{prefix}p99_violation_{key}"] = float(np.quantile(violation, 0.99)) if len(y) else float("nan")
        out[f"{prefix}max_violation_{key}"] = float(np.max(violation)) if len(y) else float("nan")
    out[f"{prefix}crossing_rate_raw"] = float(df.get("raw_crossing", pd.Series(dtype=bool)).mean()) if len(df) else float("nan")
    return out


def conditional_metrics(df: pd.DataFrame, group_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value, part in df.groupby(group_col, dropna=False):
        row: dict[str, object] = {"group": group_col, "value": str(value)}
        row.update(quantile_metrics(part, epsilon=0.0))
        rows.append(row)
    return rows


def evaluate_predictions(config: dict, predictions_path: Path, split: str = "validation") -> dict:
    df = pd.read_parquet(predictions_path)
    epsilon = float(config.get("evaluation", {}).get("coverage_epsilon", config.get("target", {}).get("eps", 0.0)))
    metrics = quantile_metrics(df, epsilon=epsilon)
    metrics["crossing_rate_postprocessed"] = float(
        ((df["pred_q70"] > df["pred_q80"]) | (df["pred_q80"] > df["pred_q90"])).mean()
    )
    strata = []
    for col in config["evaluation"]["strata"]:
        if col in df.columns:
            strata.extend(conditional_metrics(df, col))
    return {
        "experiment_id": config["experiment_id"],
        "split": split,
        "prediction_path": str(predictions_path),
        "metrics": metrics,
        "conditional_metrics": strata,
    }


def write_markdown(report: dict, path: Path) -> None:
    m = report["metrics"]
    lines = [
        "# Price Estimator Quantile Baseline",
        "",
        f"experiment_id: `{report['experiment_id']}`",
        f"prediction_path: `{report['prediction_path']}`",
        "",
        "## Validation Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in m.items():
        lines.append(f"| {key} | {value:.8f} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines.extend(["", "## Conditional Metrics", ""])
    for row in report["conditional_metrics"]:
        lines.append(
            f"- {row['group']}={row['value']}: "
            f"n={row['sample_count']:.0f}, "
            f"cov70={row['coverage_q70']:.4f}, cov80={row['coverage_q80']:.4f}, cov90={row['coverage_q90']:.4f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/configs/catboost_quantile_baseline.yaml")
    parser.add_argument("--predictions", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    predictions = resolve_path(args.predictions or config["paths"]["predictions_validation"])
    report = evaluate_predictions(config, predictions)
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    write_json(reports_dir / "quantile_metrics.json", report)
    write_markdown(report, reports_dir / "quantile_baseline_report.md")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
