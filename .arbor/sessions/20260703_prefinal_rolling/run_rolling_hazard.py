#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
SOURCE = ROOT / "price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet"
BASE_CONFIG = ROOT / ".arbor/sessions/20260703_sum_pnl_no_leak/bdev_h2.yaml"
TRAINER = ROOT / "price_estimator/expected_return/train_low_cdf_and_backtest.py"

FOLDS = [
    ("w1", "2026-02-27", "2026-03-06"),
    ("w2", "2026-03-06", "2026-03-13"),
    ("w3", "2026-03-13", "2026-03-20"),
    ("w4", "2026-03-20", "2026-03-27"),
    ("w5", "2026-03-27", "2026-04-03"),
    ("w6", "2026-04-03", "2026-04-11"),
]


def main() -> None:
    frame = pd.read_parquet(SOURCE)
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    summaries = []
    for name, start, end in FOLDS:
        fold_dir = SESSION / "folds" / name
        data_dir = fold_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        train = frame.loc[ts < start_ts].copy()
        dev = frame.loc[(ts >= start_ts) & (ts < end_ts)].copy()
        if len(train) < 3000 or len(dev) < 500:
            raise RuntimeError(f"invalid fold {name}: train={len(train)}, dev={len(dev)}")
        train_path, dev_path = data_dir/"train.parquet", data_dir/"dev.parquet"
        train.to_parquet(train_path, index=False)
        dev.to_parquet(dev_path, index=False)

        cfg = copy.deepcopy(base)
        cfg["experiment_id"] = f"20260703_prefinal_rolling_{name}"
        cfg["paths"].update({
            "train_dataset": str(train_path.relative_to(ROOT)).replace("\\", "/"),
            "validation_dataset": str(dev_path.relative_to(ROOT)).replace("\\", "/"),
            "models_dir": str((fold_dir/"models").relative_to(ROOT)).replace("\\", "/"),
            "reports_dir": str((fold_dir/"reports").relative_to(ROOT)).replace("\\", "/"),
            "predictions_train": str((fold_dir/"reports/predictions_train.parquet").relative_to(ROOT)).replace("\\", "/"),
            "predictions_calibration": str((fold_dir/"reports/predictions_calibration.parquet").relative_to(ROOT)).replace("\\", "/"),
            "predictions_validation": str((fold_dir/"reports/predictions_validation.parquet").relative_to(ROOT)).replace("\\", "/"),
            "target_summary_source": str((fold_dir/"nonexistent_target_summary.json").relative_to(ROOT)).replace("\\", "/"),
        })
        cfg["training"].update({"epochs": 30, "early_stop_patience": 6, "device": "cuda", "random_seed": 20260703})
        forbidden = set(cfg["features"].get("forbidden_columns", []))
        forbidden.add("stage1_sample_weight")
        cfg["features"]["forbidden_columns"] = sorted(forbidden)
        config_path = fold_dir/"config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        subprocess.run([sys.executable, str(TRAINER), "--config", str(config_path)], cwd=ROOT, check=True)
        report = json.loads((fold_dir/"reports/summary_metrics.json").read_text(encoding="utf-8"))
        summaries.append({"fold": name, "start": start, "end": end, "train_rows": len(train), "dev_rows": len(dev),
                          "sum_pnl": report["validation_metrics"]["sum_pnl"],
                          "order_coverage": report["validation_metrics"]["order_coverage"],
                          "brier": report["validation_metrics"]["brier_score"],
                          "fill_gap": report["validation_metrics"]["submitted_fill_calibration_gap"]})
        (SESSION/"rolling_hazard_progress.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(SESSION/"rolling_hazard_summary.csv", index=False)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
