#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an expected-return hazard checkpoint for execution.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    experiment = (ROOT / args.experiment).resolve()
    output = (ROOT / args.output).resolve()
    config = yaml.safe_load((experiment / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((experiment / "reports" / "summary_metrics.json").read_text(encoding="utf-8"))
    checkpoint_path = (ROOT / config["paths"]["h2_checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    hidden_indices = sorted({int(key.split(".")[1]) for key in state if key.startswith("trunk.") and key.endswith(".weight")})
    arrays: dict[str, np.ndarray] = {"tick_grid": np.asarray(checkpoint["tick_grid"], dtype=np.float32)}
    for output_index, state_index in enumerate(hidden_indices):
        arrays[f"hidden_{output_index}__weight"] = state[f"trunk.{state_index}.weight"].cpu().numpy()
        arrays[f"hidden_{output_index}__bias"] = state[f"trunk.{state_index}.bias"].cpu().numpy()
    arrays["output__weight"] = state["output.weight"].cpu().numpy()
    arrays["output__bias"] = state["output.bias"].cpu().numpy()

    policy = {
        "min_bid": float(config["target"]["min_bid"]),
        "min_ev": float(report["policy"]["selected_min_ev"]),
        "candidate_gc_strict_floor": float(config["order_policy"]["candidate_gc_strict_floor"]),
    }
    metadata = {
        "preprocessor": checkpoint["preprocessor"],
        "model": {"layer_count": len(hidden_indices)},
        "order_policy": policy,
    }
    output.mkdir(parents=True, exist_ok=True)
    model_file = "expected_return_hazard.npz"
    np.savez_compressed(output / model_file, metadata=json.dumps(metadata), **arrays)
    (output / "feature_columns.json").write_text(
        json.dumps(checkpoint["feature_columns"], indent=2), encoding="utf-8"
    )
    manifest = {
        "artifact_type": "price_estimator_expected_return_hazard",
        "experiment_id": config["experiment_id"],
        "model_format": "expected_return_hazard_numpy",
        "model_file": model_file,
        "feature_columns_file": "feature_columns.json",
        "feature_count": len(checkpoint["feature_columns"]),
        "prediction_column": "expected_return_bid",
        "selected_side_column": "selected_side",
        "yes_value": "UP",
        "no_value": "DOWN",
        "round_decimals": 2,
        "best_ask_offset": 0.01,
        "fallback_price_mode": "skip",
        "order_price_policy": "min(best_ask - 0.01, expected_return_optimal_bid)",
        "order_policy": policy,
        "source_config_path": str(Path(args.experiment) / "config.yaml"),
        "source_report_path": str(Path(args.experiment) / "reports" / "summary_metrics.json"),
        "source_checkpoint_path": config["paths"]["h2_checkpoint"],
        "validation_metrics": report["validation_metrics"],
        "coverage_constraint_satisfied": report["coverage_constraint_satisfied"],
        "offline_validation_metric_source": report["offline_validation_metric_source"],
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
