from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from price_estimator.expected_return.expected_return_common import (  # noqa: E402
    derive_selected_side_l2_features,
    p_side_bucket,
    session_label,
)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _read_partitioned(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(path)
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def _selected(values_up: pd.Series, values_down: pd.Series, side: pd.Series) -> np.ndarray:
    return np.where(side.eq("UP"), values_up, values_down)


def build_variant(
    variant_dir: Path,
    baseline: pd.DataFrame,
    l2: pd.DataFrame,
    references: pd.DataFrame,
    lows: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    report = json.loads((variant_dir / "report.json").read_text(encoding="utf-8"))
    threshold_up = float(report["validation_metrics"]["selected_t_up"])
    threshold_down = float(report["validation_metrics"]["selected_t_down"])
    frames = []
    split_summaries = {}
    l2_columns = [column for column in l2 if column.startswith("pm_l2_1m_")]
    l2_join = l2[["market_t0", "feature_cutoff_time", *l2_columns]]
    reference_join = references[
        ["market_t0", "up_last_trade_price_1m", "down_last_trade_price_1m"]
    ]
    low_join = lows[
        [
            "market_t0", "up_future_low_4m", "down_future_low_4m",
            "up_future_low_time_4m", "down_future_low_time_4m",
        ]
    ]
    for split in ("train", "calibration", "validation"):
        predictions = pd.read_parquet(variant_dir / f"predictions_{split}.parquet")
        predictions = predictions.rename(columns={"selected_side": "threshold_selected_side"})
        frame = predictions.merge(baseline, on="market_t0", how="left", validate="one_to_one", suffixes=("_prediction", ""))
        frame = frame.merge(l2_join, on="market_t0", how="left", validate="one_to_one")
        frame = frame.merge(reference_join, on="market_t0", how="left", validate="one_to_one")
        frame = frame.merge(low_join, on="market_t0", how="left", validate="one_to_one")
        frame["timestamp"] = frame["market_t0"]
        frame["feature_timestamp"] = pd.to_datetime(
            frame.get("feature_timestamp", frame["market_t0"] + pd.Timedelta(minutes=1)), utc=True
        ).fillna(pd.to_datetime(frame["market_t0"], utc=True) + pd.Timedelta(minutes=1))
        frame["decision_time"] = pd.to_datetime(
            frame.get("decision_time", frame["feature_timestamp"]), utc=True
        ).fillna(frame["feature_timestamp"])
        frame["calibrated_p_up"] = pd.to_numeric(frame["calibrated_p_up"], errors="coerce")
        frame["p_up"] = frame["calibrated_p_up"]
        frame["threshold_accepted"] = frame["threshold_selected_side"].isin(["UP", "DOWN"])
        # Preserve the complete source universe for correct direction coverage.
        # The price runner masks all orders to threshold_accepted rows. An argmax
        # side is retained for non-signals only so CDF tensors keep row alignment.
        frame["selected_side"] = frame["threshold_selected_side"].where(
            frame["threshold_accepted"],
            np.where(frame["p_up"] >= 0.5, "UP", "DOWN"),
        ).astype("string")
        frame["selected_outcome"] = frame["selected_side"].str.lower()
        frame["selected_t_up"] = threshold_up
        frame["selected_t_down"] = threshold_down
        frame["p_side"] = np.where(frame["selected_side"].eq("UP"), frame["p_up"], 1.0 - frame["p_up"])
        target_values = pd.to_numeric(frame.get("target"), errors="coerce")
        if "target_prediction" in frame:
            target_values = target_values.fillna(
                pd.to_numeric(frame["target_prediction"], errors="coerce")
            )
        if target_values.isna().any():
            raise ValueError("resolved direction target missing after prediction join")
        frame["target"] = target_values.astype(int)
        frame["correct"] = (
            (frame["target"].eq(1) & frame["selected_side"].eq("UP"))
            | (frame["target"].eq(0) & frame["selected_side"].eq("DOWN"))
        )
        frame["chosen_low"] = _selected(
            pd.to_numeric(frame["up_future_low_4m"], errors="coerce"),
            pd.to_numeric(frame["down_future_low_4m"], errors="coerce"),
            frame["selected_side"],
        )
        frame["chosen_low_trade_time"] = pd.to_datetime(
            _selected(frame["up_future_low_time_4m"], frame["down_future_low_time_4m"], frame["selected_side"]),
            utc=True,
        )
        frame["time_to_chosen_low_sec"] = (
            frame["chosen_low_trade_time"] - pd.to_datetime(frame["decision_time"], utc=True)
        ).dt.total_seconds()
        frame["target_raw"] = frame["chosen_low"]
        frame["price_reference"] = _selected(
            frame["up_last_trade_price_1m"], frame["down_last_trade_price_1m"], frame["selected_side"]
        )
        frame["direction_confidence"] = (frame["p_up"] - 0.5).abs()
        frame["p_bin"] = p_side_bucket(frame["p_side"])
        frame["p_side_bucket"] = p_side_bucket(frame["p_side"])
        frame["market_time_bucket"] = session_label(frame["decision_time"]).to_numpy()
        frame = derive_selected_side_l2_features(frame)
        frame["split"] = split
        frame["chosen_low_reason"] = np.where(frame["chosen_low"].notna(), "l2_future_trade", "no_l2_future_trade")
        if (frame["chosen_low_trade_time"].notna() & (frame["chosen_low_trade_time"] <= frame["feature_cutoff_time"])).any():
            raise ValueError("future-low target crosses feature cutoff")
        frames.append(frame)
        split_summaries[split] = {
            "source_prediction_rows": len(predictions),
            "output_rows": len(frame),
            "accepted_signal_count": int(frame["threshold_accepted"].sum()),
            "accepted_coverage": float(frame["threshold_accepted"].mean()),
            "accepted_accuracy": float(frame.loc[frame["threshold_accepted"], "correct"].mean()),
            "future_low_coverage": float(frame["chosen_low"].notna().mean()),
            "start": frame["market_t0"].min(),
            "end": frame["market_t0"].max(),
        }
    combined_train = pd.concat([frames[0], frames[1]], ignore_index=True).sort_values("market_t0")
    validation = frames[2].sort_values("market_t0")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_train.to_parquet(output_dir / "expected_return_train.parquet", index=False)
    validation.to_parquet(output_dir / "expected_return_validation.parquet", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classifier_experiment_id": report["experiment_id"],
        "classifier_artifact_hash": report["feature_schema_hash"],
        "probability_calibrator": report["probability_calibrator"],
        "probability_calibration_window": report["windows"]["calibration"],
        "probability_column": "calibrated_p_up",
        "raw_probability_diagnostic_column": "raw_p_up",
        "future_low_target": "polymarket_l2_future_four_minute_v1",
        "split_summaries": split_summaries,
        "train_rows": len(combined_train),
        "validation_rows": len(validation),
        "no_future_or_label_columns_in_model_features": True,
    }
    _atomic_json(summary, output_dir / "target_build_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-experiment", type=Path, required=True)
    parser.add_argument("--baseline-frame-dir", type=Path, required=True)
    parser.add_argument("--l2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    baseline = pd.concat(
        [pd.read_parquet(args.baseline_frame_dir / "development_frame.parquet"), pd.read_parquet(args.baseline_frame_dir / "validation_frame.parquet")],
        ignore_index=True,
    ).drop_duplicates("market_t0")
    l2 = _read_partitioned(args.l2_root / "first_minute_features")
    references = _read_partitioned(args.l2_root / "first_minute_price_reference")
    lows = _read_partitioned(args.l2_root / "future_four_minute_lows")
    summaries = {}
    for variant_dir in sorted(path for path in args.direction_experiment.iterdir() if (path / "report.json").exists()):
        summaries[variant_dir.name] = build_variant(
            variant_dir, baseline, l2, references, lows, args.output_root / variant_dir.name
        )
    _atomic_json({"variants": summaries}, args.output_root / "target_build_summary.json")


if __name__ == "__main__":
    main()
