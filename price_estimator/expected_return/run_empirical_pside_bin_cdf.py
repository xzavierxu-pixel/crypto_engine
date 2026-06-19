#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))

from price_estimator_common import load_config, resolve_path  # noqa: E402

from expected_return_common import git_commit, write_json  # noqa: E402
from train_low_cdf_and_backtest import (  # noqa: E402
    BacktestResult,
    backtest_metrics,
    backtest_with_bid,
    build_tick_grid,
    choose_survival_expected_return_bids,
    empirical_cdf,
    floor_to_tick,
    select_min_ev,
    split_fit_calibration,
    write_predictions,
)


def pside_bin_index(p_side: np.ndarray, bin_width: float) -> np.ndarray:
    if not 0.0 < bin_width < 1.0:
        raise ValueError("bin_width must be in (0, 1)")
    values = np.asarray(p_side, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_side must be finite and in [0, 1]")
    count = int(np.ceil(1.0 / bin_width))
    return np.clip(np.floor(values / bin_width + 1e-12).astype(int), 0, count - 1)


def fit_empirical_pside_bin_cdf(
    calibration_correct: pd.DataFrame,
    tick_grid: np.ndarray,
    bin_width: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    count = int(np.ceil(1.0 / bin_width))
    indices = pside_bin_index(calibration_correct["p_side"].to_numpy(dtype=float), bin_width)
    chosen_low = pd.to_numeric(calibration_correct["chosen_low"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(chosen_low).all():
        raise ValueError("calibration_correct chosen_low must be finite")
    populated = sorted(set(indices.tolist()))
    if not populated:
        raise ValueError("No populated p_side bins in calibration_correct")
    gc_by_bin = np.empty((count, len(tick_grid)), dtype=float)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        own = chosen_low[indices == index]
        source_index = index if len(own) else min(populated, key=lambda value: (abs(value - index), value))
        source = chosen_low[indices == source_index]
        gc_by_bin[index] = empirical_cdf(source, tick_grid)
        rows.append(
            {
                "bin_index": index,
                "p_side_lower": index * bin_width,
                "p_side_upper": min((index + 1) * bin_width, 1.0),
                "sample_count": int(len(own)),
                "source_bin_index": int(source_index),
                "source_sample_count": int(len(source)),
                "fallback_used": bool(source_index != index),
            }
        )
    if np.any(np.diff(gc_by_bin, axis=1) < -1e-12):
        raise AssertionError("Empirical Gc must be non-decreasing over limit price")
    return gc_by_bin, pd.DataFrame(rows)


def gc_for_frame(frame: pd.DataFrame, gc_by_bin: np.ndarray, bin_width: float) -> np.ndarray:
    indices = pside_bin_index(frame["p_side"].to_numpy(dtype=float), bin_width)
    return gc_by_bin[indices]


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
    fit_all, calibration_all = split_fit_calibration(train_all, config)
    calibration_correct = calibration_all.loc[
        calibration_all["correct"].astype(bool) & calibration_all["chosen_low"].notna()
    ].copy()
    gc_fit_source = str(config["model"].get("gc_fit_source", "calibration_correct"))
    if gc_fit_source == "calibration_correct":
        gc_fit_frame = calibration_correct
    elif gc_fit_source == "fit_correct":
        gc_fit_frame = fit_all.loc[
            fit_all["correct"].astype(bool) & fit_all["chosen_low"].notna()
        ].copy()
    else:
        raise ValueError(f"Unsupported model.gc_fit_source: {gc_fit_source}")

    tick_size = float(config["target"]["tick_size"])
    min_bid = float(config["target"].get("min_bid", tick_size))
    tick_grid = build_tick_grid(tick_size, float(config["model"]["max_price"]))
    bin_width = float(config["model"]["p_side_bin_width"])
    gc_by_bin, bin_summary = fit_empirical_pside_bin_cdf(gc_fit_frame, tick_grid, bin_width)
    bin_summary_path = reports_dir / "p_side_bin_summary.csv"
    bin_summary.to_csv(bin_summary_path, index=False)
    distribution_path = reports_dir / "p_side_bin_gc.csv"
    distribution_rows = []
    for index, gc in enumerate(gc_by_bin):
        distribution_rows.extend(
            {"bin_index": index, "limit_price": float(price), "gc": float(probability)}
            for price, probability in zip(tick_grid, gc)
        )
    pd.DataFrame(distribution_rows).to_csv(distribution_path, index=False)

    train_mask = train_all["threshold_accepted"].astype(bool).to_numpy()
    calibration_mask = calibration_all["threshold_accepted"].astype(bool).to_numpy()
    validation_mask = validation["threshold_accepted"].astype(bool).to_numpy()
    train_accepted = train_all.loc[train_mask].copy()
    calibration_accepted = calibration_all.loc[calibration_mask].copy()
    validation_accepted = validation.loc[validation_mask].copy()
    validation_bin_indices = pside_bin_index(
        validation_accepted["p_side"].to_numpy(dtype=float), bin_width
    )
    used_bins, used_counts = np.unique(validation_bin_indices, return_counts=True)
    fallback_bins = set(
        bin_summary.loc[bin_summary["fallback_used"], "bin_index"].astype(int).tolist()
    )
    validation_bin_diagnostics = {
        "used_bin_count": int(len(used_bins)),
        "used_bins": [int(value) for value in used_bins],
        "fallback_bins_used": [int(value) for value in used_bins if int(value) in fallback_bins],
        "fallback_affected_count": int(
            sum(count for value, count in zip(used_bins, used_counts) if int(value) in fallback_bins)
        ),
        "minimum_fit_samples_in_used_own_bin": int(
            bin_summary.set_index("bin_index").loc[used_bins, "sample_count"].min()
        ),
    }

    def run(frame: pd.DataFrame, min_ev: float) -> BacktestResult:
        q = frame["p_side"].to_numpy(dtype=float)
        gc = gc_for_frame(frame, gc_by_bin, bin_width)
        bid, ev, fill_prob = choose_survival_expected_return_bids(q, gc, tick_grid, min_bid, min_ev)
        return backtest_with_bid(frame, bid, ev, fill_prob)

    min_ev_grid = [float(value) for value in config["target"]["min_ev_grid"]]
    calibration_candidates = {value: run(calibration_accepted, value) for value in min_ev_grid}
    selected_min_ev, min_ev_search = select_min_ev(
        calibration_candidates,
        calibration_accepted,
        len(calibration_all),
        int(config["target"].get("min_ev_min_order_count", 100)),
    )
    train_result = run(train_accepted, selected_min_ev)
    calibration_result = calibration_candidates[selected_min_ev]
    validation_result = run(validation_accepted, selected_min_ev)
    validation_frontier = [
        {"min_ev": value, **backtest_metrics(validation_accepted, run(validation_accepted, value), len(validation))}
        for value in min_ev_grid
    ]
    o0_result = run(validation_accepted, float("-inf"))

    q_validation = validation_accepted["p_side"].to_numpy(dtype=float)
    baselines = {
        name: backtest_metrics(validation_accepted, backtest_with_bid(validation_accepted, bid), len(validation))
        for name, bid in {
            "fixed_0p50_pside": np.maximum(floor_to_tick(0.50 * q_validation, tick_size), min_bid),
            "fixed_0p75_pside": np.maximum(floor_to_tick(0.75 * q_validation, tick_size), min_bid),
            "pay_pside": np.maximum(floor_to_tick(q_validation, tick_size), min_bid),
        }.items()
    }
    baselines["empirical_pside_bin_cdf"] = backtest_metrics(
        validation_accepted, validation_result, len(validation)
    )

    write_predictions(
        train_accepted,
        train_result,
        resolve_path(config["paths"]["predictions_train"]),
        train_accepted["p_side"].to_numpy(dtype=float),
    )
    write_predictions(
        calibration_accepted,
        calibration_result,
        resolve_path(config["paths"]["predictions_calibration"]),
        calibration_accepted["p_side"].to_numpy(dtype=float),
    )
    write_predictions(
        validation_accepted,
        validation_result,
        resolve_path(config["paths"]["predictions_validation"]),
        validation_accepted["p_side"].to_numpy(dtype=float),
    )
    pd.DataFrame(validation_frontier).to_csv(reports_dir / "validation_frontier.csv", index=False)

    model_payload = {
        "family": "empirical_pside_bin_cdf",
        "source_split": gc_fit_source,
        "source_sample_count": int(len(gc_fit_frame)),
        "p_side_bin_width": bin_width,
        "tick_grid": tick_grid.tolist(),
        "gc_by_bin": gc_by_bin.tolist(),
        "selected_min_ev": selected_min_ev,
    }
    checkpoint_path = models_dir / "empirical_pside_bin_cdf.json"
    write_json(checkpoint_path, model_payload)
    min_coverage = float(config.get("objective", {}).get("min_coverage", 0.70))
    direction_coverage = float(validation_mask.mean())
    report = {
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit(),
        "git_commit_at_evaluation": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "summary_metrics.json"),
        "primary_metric": "validation forced mean_accepted_pnl for empirical p_side-bin Gc",
        "model_family": "empirical_pside_bin_cdf",
        "gc_fit_source": gc_fit_source,
        "gc_fit_sample_count": int(len(gc_fit_frame)),
        "validation_selection_note": "Validation did not participate in Gc fitting or min_ev selection.",
        "p_side_bin_width": bin_width,
        "empty_bin_fallback": "nearest_nonempty_fit_source_bin",
        "empty_bin_count": int(bin_summary["fallback_used"].sum()),
        "validation_bin_diagnostics": validation_bin_diagnostics,
        "selected_min_ev": selected_min_ev,
        "min_ev_selection_source": "calibration",
        "min_ev_search": min_ev_search,
        "o0_forced_no_abstain_validation_metrics": backtest_metrics(
            validation_accepted, o0_result, len(validation)
        ),
        "train_metrics": backtest_metrics(train_accepted, train_result, len(train_all)),
        "train_window": window(train_accepted),
        "calibration_metrics": backtest_metrics(
            calibration_accepted, calibration_result, len(calibration_all)
        ),
        "calibration_window": window(calibration_accepted),
        "validation_metrics": backtest_metrics(validation_accepted, validation_result, len(validation)),
        "validation_window": window(validation_accepted),
        "validation_frontier": validation_frontier,
        "validation_baselines": baselines,
        "signal_coverage": direction_coverage,
        "coverage_constraint_satisfied": bool(direction_coverage >= min_coverage),
        "coverage_note": "Direction coverage is unchanged; expected-return order coverage is separate.",
        "deploy_training_mode": "not_applicable_empirical_gc_backtest",
        "offline_validation_metric_source": config["paths"]["validation_dataset"],
        "chosen_low_missing_policy": "excluded_from_gc_fit",
        "gc_monotonicity_check": True,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "config_snapshot": str(reports_dir / "config_used.yaml"),
            "bin_summary": str(bin_summary_path),
            "gc_distribution": str(distribution_path),
            "train_predictions": str(resolve_path(config["paths"]["predictions_train"])),
            "calibration_predictions": str(resolve_path(config["paths"]["predictions_calibration"])),
            "validation_predictions": str(resolve_path(config["paths"]["predictions_validation"])),
        },
    }
    write_json(reports_dir / "summary_metrics.json", report)
    print({"selected_min_ev": selected_min_ev, "validation_metrics": report["validation_metrics"]})


if __name__ == "__main__":
    main()
