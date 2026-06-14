#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
ROOT = PRICE_ESTIMATOR_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))

from price_estimator_common import load_config, resolve_path  # noqa: E402


def slug_float(value: float) -> str:
    return str(value).replace(".", "p")


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def experiment_config(base: dict[str, Any], experiment_id: str, loss: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(base)
    config["experiment_id"] = experiment_id
    config["loss"] = loss
    exp_dir = f"price_estimator/experiment/{experiment_id}"
    config["paths"]["models_dir"] = f"{exp_dir}/models"
    config["paths"]["reports_dir"] = f"{exp_dir}/reports"
    config["paths"]["predictions_train"] = f"{exp_dir}/reports/predictions_train.parquet"
    config["paths"]["predictions_validation"] = f"{exp_dir}/reports/predictions_validation.parquet"
    return config


def default_grid(variant: str) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    if variant in {"all", "mean_gap"}:
        for c in [50.0, 20.0, 10.0, 5.0]:
            for scale in [0.03, 0.05, 0.08]:
                rows.append(
                    (
                        f"mean_gap_C{slug_float(c)}_s{slug_float(scale)}",
                        {"type": "mean_gap_soft_violation", "violation_penalty": c, "scale": scale},
                    )
                )
    if variant in {"all", "asymmetric"}:
        for w_under in [20.0, 10.0, 5.0, 3.0]:
            for scale in [0.08, 0.05, 0.03, 0.02]:
                rows.append(
                    (
                        f"asym_wu{slug_float(w_under)}_s{slug_float(scale)}",
                        {"type": "asymmetric_logcosh", "w_under": w_under, "w_over": 1.0, "scale": scale},
                    )
                )
    return rows


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="price_estimator/upper_bound_mlp/configs/upper_bound_mlp_asymmetric_logcosh.yaml")
    parser.add_argument("--variant", choices=["all", "mean_gap", "asymmetric"], default="all")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based index into the selected grid.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    base = load_config(args.base_config)
    stamp = args.tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    grid = default_grid(args.variant)
    if args.start_index:
        grid = grid[args.start_index :]
    if args.max_runs is not None:
        grid = grid[: args.max_runs]

    summaries: list[dict[str, Any]] = []
    for idx, (suffix, loss) in enumerate(grid, start=1):
        experiment_id = f"{stamp}_upper_bound_mlp_{suffix}"
        config = experiment_config(base, experiment_id, loss)
        exp_dir = resolve_path(f"price_estimator/experiment/{experiment_id}")
        config_path = exp_dir / "config.yaml"
        write_config(config_path, config)
        print(json.dumps({"run": idx, "total": len(grid), "experiment_id": experiment_id, "loss": loss}, sort_keys=True))
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "train_upper_bound_mlp.py"), "--config", str(config_path)],
            cwd=ROOT,
            check=True,
        )
        report_path = exp_dir / "reports" / "summary_metrics.json"
        report = load_summary(report_path)
        summaries.append(
            {
                "experiment_id": experiment_id,
                "config_path": str(config_path.relative_to(ROOT)),
                "report_path": str(report_path.relative_to(ROOT)),
                "loss": loss,
                "validation_metrics": report["validation_metrics"],
                "success_criteria": report["success_criteria"],
            }
        )

    summary_path = resolve_path(f"price_estimator/experiment/{stamp}_upper_bound_mlp_grid_summary.json")
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"grid_summary": str(summary_path.relative_to(ROOT)), "runs": len(summaries)}, sort_keys=True))


if __name__ == "__main__":
    main()
