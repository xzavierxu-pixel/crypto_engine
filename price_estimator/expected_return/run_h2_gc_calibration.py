#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_distribution

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "upper_bound_mlp"))

from price_estimator_common import load_config, resolve_path  # noqa: E402
from train_upper_bound_mlp import Preprocessor  # noqa: E402

from expected_return_common import git_commit, write_json  # noqa: E402
from run_empirical_pside_bin_cdf import pside_bin_index  # noqa: E402
from train_low_cdf_and_backtest import (  # noqa: E402
    BacktestResult,
    HazardMLP,
    backtest_metrics,
    backtest_with_bid,
    choose_survival_expected_return_bids,
    predict_hazard,
    write_predictions,
)


def split_calibration_halves(frame: pd.DataFrame, timestamp_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(timestamp_column).reset_index(drop=True)
    midpoint = len(ordered) // 2
    if midpoint == 0 or midpoint == len(ordered):
        raise ValueError("Calibration frame is too small to split")
    return ordered.iloc[:midpoint].copy(), ordered.iloc[midpoint:].copy()


def fit_bin_gc_calibrator(
    frame_correct: pd.DataFrame,
    h2_gc_correct: np.ndarray,
    tick_grid: np.ndarray,
    bin_width: float,
    beta_prior: float,
    confidence_levels: list[float],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    indices = pside_bin_index(frame_correct["p_side"].to_numpy(dtype=float), bin_width)
    chosen_low = pd.to_numeric(frame_correct["chosen_low"], errors="coerce").to_numpy(dtype=float)
    if h2_gc_correct.shape != (len(frame_correct), len(tick_grid)):
        raise ValueError("h2_gc_correct shape mismatch")
    bin_count = int(np.ceil(1.0 / bin_width))
    populated = sorted(set(indices.tolist()))
    if not populated:
        raise ValueError("No populated calibration bins")
    empirical = np.empty((bin_count, len(tick_grid)), dtype=float)
    mean_h2 = np.empty_like(empirical)
    upper = {level: np.empty_like(empirical) for level in confidence_levels}
    rows: list[dict[str, Any]] = []
    for index in range(bin_count):
        source_index = index if np.any(indices == index) else min(
            populated, key=lambda value: (abs(value - index), value)
        )
        mask = indices == source_index
        lows = chosen_low[mask]
        hits = (lows[:, None] <= tick_grid[None, :]).sum(axis=0)
        n = int(mask.sum())
        empirical[index] = hits / n
        mean_h2[index] = h2_gc_correct[mask].mean(axis=0)
        for level in confidence_levels:
            values = beta_distribution.ppf(level, hits + beta_prior, n - hits + beta_prior)
            upper[level][index] = np.maximum.accumulate(values)
        rows.append(
            {
                "bin_index": index,
                "p_side_lower": index * bin_width,
                "p_side_upper": min((index + 1) * bin_width, 1.0),
                "sample_count": int((indices == index).sum()),
                "source_bin_index": int(source_index),
                "source_sample_count": n,
                "fallback_used": bool(source_index != index),
            }
        )
    return {"empirical": empirical, "mean_h2": mean_h2, **{f"upper_{k}": v for k, v in upper.items()}}, pd.DataFrame(rows)


def adjust_gc(
    h2_gc: np.ndarray,
    p_side: np.ndarray,
    calibrator: dict[str, np.ndarray],
    bin_width: float,
    method: str,
    parameter: float,
) -> np.ndarray:
    indices = pside_bin_index(p_side, bin_width)
    if method == "blend":
        adjusted = (1.0 - parameter) * h2_gc + parameter * calibrator["empirical"][indices]
    elif method == "upper_cap":
        key = f"upper_{parameter}"
        if key not in calibrator:
            raise ValueError(f"Missing calibrated upper bound for confidence={parameter}")
        adjusted = np.minimum(h2_gc, calibrator[key][indices])
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
    adjusted = np.maximum.accumulate(np.clip(adjusted, 0.0, 1.0), axis=1)
    if np.any(np.diff(adjusted, axis=1) < -1e-12):
        raise AssertionError("Adjusted Gc is not monotone")
    return adjusted


def choose_best_policy(rows: list[dict[str, float]], min_order_count: int) -> dict[str, float]:
    eligible = [row for row in rows if row["order_count"] >= min_order_count]
    if not eligible:
        raise ValueError(f"No policy candidate meets min_order_count={min_order_count}")
    return max(
        eligible,
        key=lambda row: (row["mean_accepted_pnl"], row["order_count"], row["parameter"], row["min_ev"]),
    )


def window(frame: pd.DataFrame) -> dict[str, Any]:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    return {"row_count": int(len(frame)), "start": str(timestamp.min()), "end": str(timestamp.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")

    train_all = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    validation = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    timestamp_column = str(config["split"].get("timestamp_column", "timestamp"))
    timestamp = pd.to_datetime(train_all[timestamp_column], utc=True)
    calibration_days = int(config["split"]["calibration_tail_days"])
    calibration_all = train_all.loc[timestamp >= timestamp.max() - pd.Timedelta(days=calibration_days)].copy()
    gc_fit_all, policy_select_all = split_calibration_halves(calibration_all, timestamp_column)

    checkpoint_path = resolve_path(config["paths"]["h2_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    preprocessor = Preprocessor(**checkpoint["preprocessor"])
    tick_grid = np.asarray(checkpoint["tick_grid"], dtype=float)
    model_config = checkpoint["model"]
    model = HazardMLP(
        input_dim=len(preprocessor.output_columns),
        hidden_dims=[int(v) for v in model_config["hidden_dims"]],
        dropout=[float(v) for v in model_config["dropout"]],
        output_dim=len(tick_grid),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(str(config["evaluation"].get("device", "cpu")))
    model.to(device)
    batch_size = int(config["evaluation"].get("batch_size", 512))

    frames = {
        "train": train_all,
        "gc_fit": gc_fit_all,
        "policy_select": policy_select_all,
        "validation": validation,
    }
    h2_gc = {
        name: predict_hazard(model, preprocessor.transform(frame), device, batch_size)[1]
        for name, frame in frames.items()
    }
    gc_fit_correct_mask = (
        gc_fit_all["correct"].astype(bool) & gc_fit_all["chosen_low"].notna()
    ).to_numpy()
    gc_fit_correct = gc_fit_all.loc[gc_fit_correct_mask].copy()
    calibration_config = config["gc_calibration"]
    method = str(calibration_config["method"])
    parameters = [float(v) for v in calibration_config["parameter_grid"]]
    confidence_levels = parameters if method == "upper_cap" else []
    calibrator, bin_summary = fit_bin_gc_calibrator(
        gc_fit_correct,
        h2_gc["gc_fit"][gc_fit_correct_mask],
        tick_grid,
        float(calibration_config["p_side_bin_width"]),
        float(calibration_config.get("beta_prior", 1.0)),
        confidence_levels,
    )
    bin_summary_path = reports_dir / "gc_calibration_bin_summary.csv"
    bin_summary.to_csv(bin_summary_path, index=False)

    long_rows = []
    for index in range(len(calibrator["empirical"])):
        for tick_index, price in enumerate(tick_grid):
            row = {
                "bin_index": index,
                "limit_price": float(price),
                "empirical_gc": float(calibrator["empirical"][index, tick_index]),
                "mean_h2_gc": float(calibrator["mean_h2"][index, tick_index]),
            }
            for level in confidence_levels:
                row[f"upper_{level}"] = float(calibrator[f"upper_{level}"][index, tick_index])
            long_rows.append(row)
    calibration_table_path = reports_dir / "gc_calibration_table.csv"
    pd.DataFrame(long_rows).to_csv(calibration_table_path, index=False)

    min_ev_grid = [float(v) for v in config["target"]["min_ev_grid"]]
    tick_size = float(config["target"]["tick_size"])
    min_bid = float(config["target"].get("min_bid", tick_size))
    bin_width = float(calibration_config["p_side_bin_width"])

    def accepted(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        mask = frame["threshold_accepted"].astype(bool).to_numpy()
        return frame.loc[mask].copy(), mask

    def run(frame: pd.DataFrame, base_gc: np.ndarray, parameter: float, min_ev: float) -> BacktestResult:
        q = frame["p_side"].to_numpy(dtype=float)
        gc = adjust_gc(base_gc, q, calibrator, bin_width, method, parameter)
        bid, ev, fill_prob = choose_survival_expected_return_bids(q, gc, tick_grid, min_bid, min_ev)
        return backtest_with_bid(frame, bid, ev, fill_prob)

    policy_select, policy_select_mask = accepted(policy_select_all)
    search_rows: list[dict[str, float]] = []
    search_results: dict[tuple[float, float], BacktestResult] = {}
    for parameter in parameters:
        for min_ev in min_ev_grid:
            result = run(
                policy_select,
                h2_gc["policy_select"][policy_select_mask],
                parameter,
                min_ev,
            )
            search_results[(parameter, min_ev)] = result
            search_rows.append(
                {
                    "parameter": parameter,
                    "min_ev": min_ev,
                    **backtest_metrics(policy_select, result, len(policy_select_all)),
                }
            )
    selected = choose_best_policy(
        search_rows, int(config["target"].get("min_ev_min_order_count", 100))
    )
    selected_parameter = float(selected["parameter"])
    selected_min_ev = float(selected["min_ev"])

    results: dict[str, BacktestResult] = {}
    accepted_frames: dict[str, pd.DataFrame] = {}
    accepted_masks: dict[str, np.ndarray] = {}
    for name in ["train", "gc_fit", "policy_select", "validation"]:
        accepted_frame, mask = accepted(frames[name])
        accepted_frames[name] = accepted_frame
        accepted_masks[name] = mask
        results[name] = run(
            accepted_frame,
            h2_gc[name][mask],
            selected_parameter,
            selected_min_ev,
        )

    validation_frontier = []
    for parameter in parameters:
        for min_ev in min_ev_grid:
            result = run(
                accepted_frames["validation"],
                h2_gc["validation"][accepted_masks["validation"]],
                parameter,
                min_ev,
            )
            validation_frontier.append(
                {
                    "parameter": parameter,
                    "min_ev": min_ev,
                    **backtest_metrics(accepted_frames["validation"], result, len(validation)),
                }
            )
    pd.DataFrame(search_rows).to_csv(reports_dir / "policy_search.csv", index=False)
    pd.DataFrame(validation_frontier).to_csv(reports_dir / "validation_frontier.csv", index=False)

    for name, prediction_key in [
        ("train", "predictions_train"),
        ("policy_select", "predictions_policy_select"),
        ("validation", "predictions_validation"),
    ]:
        write_predictions(
            accepted_frames[name],
            results[name],
            resolve_path(config["paths"][prediction_key]),
            accepted_frames[name]["p_side"].to_numpy(dtype=float),
        )

    model_payload = {
        "family": "h2_gc_calibration",
        "method": method,
        "source_h2_checkpoint": str(checkpoint_path),
        "p_side_bin_width": bin_width,
        "beta_prior": float(calibration_config.get("beta_prior", 1.0)),
        "selected_parameter": selected_parameter,
        "selected_min_ev": selected_min_ev,
        "tick_grid": tick_grid.tolist(),
        "calibrator": {key: value.tolist() for key, value in calibrator.items()},
    }
    calibrated_checkpoint_path = models_dir / "h2_gc_calibration.json"
    write_json(calibrated_checkpoint_path, model_payload)

    validation_metrics = backtest_metrics(
        accepted_frames["validation"], results["validation"], len(validation)
    )
    direction_coverage = float(accepted_masks["validation"].mean())
    report = {
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit(),
        "git_commit_at_evaluation": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "summary_metrics.json"),
        "primary_metric": "validation forced mean_accepted_pnl after H2 Gc calibration",
        "model_family": "h2_hazard_gc_calibration",
        "gc_calibration_method": method,
        "p_side_bin_width": bin_width,
        "selected_parameter": selected_parameter,
        "selected_min_ev": selected_min_ev,
        "selection_source": "second_chronological_half_of_original_calibration",
        "gc_fit_source": "first_chronological_half_of_original_calibration_correct",
        "validation_selection_note": "Validation did not participate in Gc calibration or policy selection.",
        "calibration_limitation": "The frozen H2 checkpoint used the original full calibration tail for early stopping.",
        "policy_search": search_rows,
        "validation_frontier": validation_frontier,
        "train_metrics": backtest_metrics(accepted_frames["train"], results["train"], len(train_all)),
        "train_window": window(accepted_frames["train"]),
        "calibration_metrics": backtest_metrics(
            accepted_frames["policy_select"], results["policy_select"], len(policy_select_all)
        ),
        "calibration_window": window(accepted_frames["policy_select"]),
        "validation_metrics": validation_metrics,
        "validation_window": window(accepted_frames["validation"]),
        "signal_coverage": direction_coverage,
        "coverage_constraint_satisfied": bool(
            direction_coverage >= float(config.get("objective", {}).get("min_coverage", 0.70))
        ),
        "coverage_note": "Direction coverage is unchanged; expected-return order coverage is separate.",
        "deploy_training_mode": "not_applicable_frozen_h2_backtest",
        "offline_validation_metric_source": config["paths"]["validation_dataset"],
        "gc_monotonicity_check": True,
        "artifacts": {
            "source_h2_checkpoint": str(checkpoint_path),
            "calibrated_checkpoint": str(calibrated_checkpoint_path),
            "config_snapshot": str(reports_dir / "config_used.yaml"),
            "bin_summary": str(bin_summary_path),
            "calibration_table": str(calibration_table_path),
            "policy_search": str(reports_dir / "policy_search.csv"),
            "validation_frontier": str(reports_dir / "validation_frontier.csv"),
            "validation_predictions": str(resolve_path(config["paths"]["predictions_validation"])),
        },
    }
    write_json(reports_dir / "summary_metrics.json", report)
    print(
        {
            "method": method,
            "selected_parameter": selected_parameter,
            "selected_min_ev": selected_min_ev,
            "validation_metrics": validation_metrics,
        }
    )


if __name__ == "__main__":
    main()
