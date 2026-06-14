#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from price_estimator_common import apply_sample_filter, load_config, load_deploy_manifest, resolve_path


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=resolve_path("."), text=True).strip()
    except Exception:
        return None


def metrics(y: pd.Series, pred: pd.Series, epsilon: float) -> dict[str, float]:
    yy = y.to_numpy(dtype=float)
    pp = pred.to_numpy(dtype=float)
    gap = pp - yy
    violation = np.maximum(yy + epsilon - pp, 0.0)
    gap_p05 = float(np.quantile(gap, 0.05)) if len(yy) else float("nan")
    gap_p95 = float(np.quantile(gap, 0.95)) if len(yy) else float("nan")
    return {
        "sample_count": float(len(yy)),
        "coverage": float(np.mean(pp >= yy + epsilon)) if len(yy) else float("nan"),
        "mean_gap": float(np.mean(gap)) if len(yy) else float("nan"),
        "median_gap": float(np.median(gap)) if len(yy) else float("nan"),
        "gap_p05": gap_p05,
        "gap_p95": gap_p95,
        "gap_p95_p05_range": gap_p95 - gap_p05 if len(yy) else float("nan"),
        "p95_violation": float(np.quantile(violation, 0.95)) if len(yy) else float("nan"),
        "p99_violation": float(np.quantile(violation, 0.99)) if len(yy) else float("nan"),
        "max_violation": float(np.max(violation)) if len(yy) else float("nan"),
    }


def make_pred(df: pd.DataFrame, base_col: str, margin: float, cap_slack: float) -> pd.Series:
    pred = pd.to_numeric(df[base_col], errors="coerce") + margin
    cap = pd.to_numeric(df["p_side"], errors="coerce") + cap_slack
    return pred.clip(lower=0.0, upper=cap.clip(upper=1.0))


def by_price_bin(df: pd.DataFrame, pred: pd.Series, epsilon: float) -> list[dict[str, Any]]:
    bins = [i / 10 for i in range(11)]
    labels = [f"{bins[i]:.2f}_{bins[i + 1]:.2f}" for i in range(10)]
    work = pd.DataFrame(
        {
            "target_raw": df["target_raw"].astype(float),
            "pred": pred.astype(float),
        }
    )
    work["price_bin"] = pd.cut(work["target_raw"], bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for value, part in work.groupby("price_bin", observed=True):
        row = {"value": str(value)}
        row.update(metrics(part["target_raw"], part["pred"], epsilon))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_deploy_manifest(config)
    valid = pd.read_parquet(resolve_path(config["paths"]["predictions_validation"]))
    train = pd.read_parquet(resolve_path(config["paths"]["predictions_train"]))
    valid, validation_filter = apply_sample_filter(valid, config, manifest)
    train, train_filter = apply_sample_filter(train, config, manifest)
    epsilon = float(config.get("target", {}).get("epsilon", config.get("target", {}).get("eps", 0.01)))

    candidates = []
    for base_col in config["calibration"]["base_columns"]:
        for margin in [float(v) for v in config["calibration"]["margins"]]:
            for cap_slack in [float(v) for v in config["calibration"]["cap_slacks"]]:
                pred = make_pred(valid, base_col, margin, cap_slack)
                row = {
                    "base_col": base_col,
                    "margin": margin,
                    "cap_slack": cap_slack,
                }
                row.update(metrics(valid["target_raw"], pred, epsilon))
                candidates.append(row)
    result = pd.DataFrame(candidates).sort_values(["mean_gap", "p99_violation"], ascending=[True, True])
    min_coverage = float(config["objective"].get("min_coverage", 0.0))
    min_mean_gap = float(config["objective"].get("min_mean_gap", 0.0))
    passing = result[(result["coverage"] >= min_coverage) & (result["mean_gap"] >= min_mean_gap)]
    selected = (passing if len(passing) else result).sort_values(["mean_gap", "gap_p95_p05_range", "p99_violation"]).iloc[0].to_dict()

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(reports_dir / "candidate_metrics.csv", index=False)
    selected_pred = make_pred(valid, str(selected["base_col"]), float(selected["margin"]), float(selected["cap_slack"]))
    pred_out = valid.copy()
    pred_out["p_pred"] = selected_pred
    pred_out["gap"] = pred_out["p_pred"] - pred_out["target_raw"]
    pred_out["violation"] = np.maximum(pred_out["target_raw"] + epsilon - pred_out["p_pred"], 0.0)
    pred_out.to_parquet(reports_dir / "predictions_validation.parquet", index=False)
    pd.DataFrame(by_price_bin(valid, selected_pred, epsilon)).to_csv(reports_dir / "diagnostics_validation_by_price_bin.csv", index=False)

    report = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_training": git_commit(),
        "config_path": args.config,
        "objective": config.get("objective", {}),
        "sample_filter": {"train": train_filter, "validation": validation_filter},
        "selected": selected,
        "candidate_count": int(len(result)),
        "artifacts": {
            "candidate_metrics": str(reports_dir / "candidate_metrics.csv"),
            "predictions_validation": str(reports_dir / "predictions_validation.parquet"),
            "diagnostics_validation_by_price_bin": str(reports_dir / "diagnostics_validation_by_price_bin.csv"),
        },
    }
    (reports_dir / "summary_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["selected"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
