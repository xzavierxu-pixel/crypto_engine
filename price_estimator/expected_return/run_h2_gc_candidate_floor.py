#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "upper_bound_mlp"))

from price_estimator_common import load_config, resolve_path  # noqa: E402
from train_upper_bound_mlp import Preprocessor  # noqa: E402

from expected_return_common import git_commit, write_json  # noqa: E402
from train_low_cdf_and_backtest import (  # noqa: E402
    HazardMLP,
    backtest_metrics,
    backtest_with_bid,
    choose_survival_expected_return_bids,
    predict_hazard,
    select_min_ev,
    write_predictions,
)


def window(frame: pd.DataFrame) -> dict[str, object]:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    return {"row_count": len(frame), "start": str(timestamp.min()), "end": str(timestamp.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")

    train_all = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    validation = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    timestamp_column = str(config["split"].get("timestamp_column", "timestamp"))
    timestamp = pd.to_datetime(train_all[timestamp_column], utc=True)
    cutoff = timestamp.max() - pd.Timedelta(days=int(config["split"]["calibration_tail_days"]))
    calibration = train_all.loc[timestamp >= cutoff].copy()

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
    frames = {"train": train_all, "calibration": calibration, "validation": validation}
    gc = {
        name: predict_hazard(model, preprocessor.transform(frame), device, batch_size)[1]
        for name, frame in frames.items()
    }

    min_bid = float(config["target"]["min_bid"])
    gc_floor = float(config["order_policy"]["candidate_gc_strict_floor"])

    def run(frame: pd.DataFrame, gc_values: np.ndarray, min_ev: float):
        mask = frame["threshold_accepted"].astype(bool).to_numpy()
        accepted = frame.loc[mask].copy()
        q = accepted["p_side"].to_numpy(dtype=float)
        bid, ev, fill_prob = choose_survival_expected_return_bids(
            q, gc_values[mask], tick_grid, min_bid, min_ev, min_fill_probability=gc_floor
        )
        return accepted, backtest_with_bid(accepted, bid, ev, fill_prob)

    candidates = {}
    calibration_accepted = None
    for min_ev in [float(value) for value in config["target"]["min_ev_grid"]]:
        calibration_accepted, candidates[min_ev] = run(calibration, gc["calibration"], min_ev)
    selected_min_ev, search_rows = select_min_ev(
        candidates, calibration_accepted, len(calibration), int(config["target"]["min_ev_min_order_count"])
    )
    train_accepted, train_result = run(train_all, gc["train"], selected_min_ev)
    validation_accepted, validation_result = run(validation, gc["validation"], selected_min_ev)
    calibration_result = candidates[selected_min_ev]

    predictions_path = reports_dir / "predictions_validation.parquet"
    write_predictions(
        validation_accepted,
        validation_result,
        predictions_path,
        validation_accepted["p_side"].to_numpy(dtype=float),
    )
    search_path = reports_dir / "min_ev_search.csv"
    pd.DataFrame(search_rows).to_csv(search_path, index=False)
    validation_metrics = backtest_metrics(validation_accepted, validation_result, len(validation))
    metrics = {
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "summary_metrics.json"),
        "primary_metric": "validation forced sum_pnl with H2 Gc candidate floor",
        "policy": {"candidate_gc_operator": ">", "candidate_gc_floor": gc_floor, "selected_min_ev": selected_min_ev},
        "min_ev_selection_source": "calibration",
        "validation_selection_note": "Validation did not participate in checkpoint or min_ev selection.",
        "train_metrics": backtest_metrics(train_accepted, train_result, len(train_all)),
        "train_window": window(train_all),
        "calibration_metrics": backtest_metrics(calibration_accepted, calibration_result, len(calibration)),
        "validation_metrics": validation_metrics,
        "validation_window": window(validation),
        "signal_coverage": validation_metrics["coverage"],
        "coverage_constraint_satisfied": bool(validation_metrics["coverage"] >= float(config["objective"]["min_coverage"])),
        "coverage_note": "Direction coverage is unchanged; expected-return order coverage is separate.",
        "deploy_training_mode": config["metadata"]["deploy_training_mode"],
        "offline_validation_metric_source": config["metadata"]["offline_validation_metric_source"],
        "h2_checkpoint": str(checkpoint_path),
        "artifacts": {"validation_predictions": str(predictions_path), "min_ev_search": str(search_path)},
    }
    write_json(reports_dir / "summary_metrics.json", metrics)


if __name__ == "__main__":
    main()
