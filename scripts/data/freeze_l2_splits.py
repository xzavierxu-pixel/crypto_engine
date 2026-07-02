from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze common L2 train/calibration/validation market IDs")
    parser.add_argument(
        "--eligible-markets", type=Path,
        default=ROOT / "artifacts/data_v2/polymarket_l2/l2_eligible_markets.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/data_v2/polymarket_l2/l2_frozen_splits.parquet",
    )
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--calibration-days", type=int, default=7)
    parser.add_argument("--expected-markets-per-day", type=int, default=288)
    args = parser.parse_args()
    frame = pd.read_parquet(args.eligible_markets).sort_values("market_t0", kind="stable")
    frame["market_t0"] = pd.to_datetime(frame["market_t0"], utc=True)
    frame["utc_day"] = frame["market_t0"].dt.floor("D")
    day_counts = frame.groupby("utc_day").size()
    first_day = frame["utc_day"].min()
    last_day = frame["utc_day"].max()
    first_complete = first_day if day_counts.get(first_day, 0) == args.expected_markets_per_day else first_day + pd.Timedelta(days=1)
    last_complete = last_day if frame["market_t0"].max() == last_day + pd.Timedelta(minutes=1435) else last_day - pd.Timedelta(days=1)
    complete_days = pd.date_range(first_complete, last_complete, freq="D", tz="UTC")
    needed = args.validation_days + args.calibration_days
    if len(complete_days) < needed:
        raise ValueError(f"need {needed} complete UTC days, found {len(complete_days)}")
    validation_days = complete_days[-args.validation_days:]
    calibration_days = complete_days[-needed:-args.validation_days]
    validation_start = validation_days.min()
    calibration_start = calibration_days.min()
    frame["split"] = "train"
    frame.loc[frame["utc_day"].isin(calibration_days), "split"] = "calibration"
    frame.loc[frame["utc_day"].isin(validation_days), "split"] = "validation"
    # Any incomplete boundary day between train and calibration is excluded, never reassigned.
    selected_days = pd.Index(calibration_days).union(pd.Index(validation_days))
    boundary = (frame["utc_day"] >= calibration_start) & ~frame["utc_day"].isin(selected_days)
    frame.loc[boundary, "split"] = "excluded_incomplete_boundary_day"
    _atomic_parquet(frame, args.output)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_days": args.validation_days,
        "calibration_days": args.calibration_days,
        "expected_markets_per_day": args.expected_markets_per_day,
        "split_counts": frame["split"].value_counts().to_dict(),
        "split_windows": {
            split: {
                "start": group["market_t0"].min(), "end": group["market_t0"].max(), "market_count": len(group)
            }
            for split, group in frame.groupby("split")
        },
        "validation_market_ids_hash": pd.util.hash_pandas_object(
            frame.loc[frame["split"].eq("validation"), "market_slug"], index=False
        ).sum(),
        "optimistic_validation": True,
    }
    _atomic_json(summary, args.output.with_name("l2_frozen_splits_manifest.json"))


if __name__ == "__main__":
    main()
