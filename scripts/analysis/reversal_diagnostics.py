from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.constants import DEFAULT_TARGET_COLUMN
from src.model.evaluation import (
    compute_reversal_trend_slice_metrics,
    compute_selective_binary_metrics,
    evaluate_selective_binary_decisions,
)


def _load_probabilities(path: Path) -> pd.Series:
    frame = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    for column in ("p_up", "probability", "probability_up", "predicted_probability"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    if len(frame.columns) == 1:
        return pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    raise ValueError(f"Could not find a p_up probability column in {path}.")


def _first_minute_return(frame: pd.DataFrame) -> pd.Series:
    for column in ("fm_ret", "ret_1", "prev_bar_return"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(float("nan"), index=frame.index)


def _bucket_records(
    frame: pd.DataFrame,
    probabilities: pd.Series,
    *,
    t_up: float,
    t_down: float,
    column: str,
    bucket_name: str,
) -> list[dict[str, Any]]:
    if column not in frame.columns or frame[column].nunique(dropna=True) < 2:
        return []
    if column == "timestamp":
        buckets = pd.to_datetime(frame[column], utc=True).dt.hour.astype(str)
    else:
        try:
            buckets = pd.qcut(frame[column].astype(float).rank(method="first"), q=5, labels=False).astype(str)
        except ValueError:
            return []
    records: list[dict[str, Any]] = []
    for bucket, index in buckets.groupby(buckets).groups.items():
        metrics = compute_selective_binary_metrics(
            frame.loc[index, DEFAULT_TARGET_COLUMN],
            probabilities.loc[index],
            t_up=t_up,
            t_down=t_down,
        )
        records.append({"bucket_type": bucket_name, "bucket": str(bucket), **metrics})
    return records


def build_reversal_diagnostics(
    frame: pd.DataFrame,
    probabilities: pd.Series,
    *,
    t_up: float,
    t_down: float,
) -> dict[str, Any]:
    if len(frame) != len(probabilities):
        raise ValueError("Frame and probabilities must have the same length.")
    probabilities = probabilities.reset_index(drop=True).astype(float)
    frame = frame.reset_index(drop=True)
    target = frame[DEFAULT_TARGET_COLUMN].astype(int)
    fm_ret = _first_minute_return(frame)
    first_minute_up = fm_ret >= 0.0
    trend_following = first_minute_up == (target == 1)
    decisions = evaluate_selective_binary_decisions(probabilities, t_up=t_up, t_down=t_down)
    accepted = decisions != "ABSTAIN"
    false_positive = ((decisions == "UP") & (target == 0)) | ((decisions == "DOWN") & (target == 1))

    slice_metrics = compute_reversal_trend_slice_metrics(
        target,
        probabilities,
        fm_ret,
        t_up=t_up,
        t_down=t_down,
    )
    bucket_metrics: list[dict[str, Any]] = []
    bucket_metrics.extend(_bucket_records(frame, probabilities, t_up=t_up, t_down=t_down, column="timestamp", bucket_name="hour"))
    bucket_metrics.extend(_bucket_records(frame, probabilities, t_up=t_up, t_down=t_down, column="p_up", bucket_name="confidence"))
    if "p_up" not in frame.columns:
        working = frame.copy()
        working["p_up"] = probabilities
        bucket_metrics.extend(_bucket_records(working, probabilities, t_up=t_up, t_down=t_down, column="p_up", bucket_name="confidence"))

    side_records: list[dict[str, Any]] = []
    for side in ("UP", "DOWN", "ABSTAIN"):
        mask = decisions == side
        if bool(mask.any()):
            side_records.append(
                {
                    "side": side,
                    **compute_selective_binary_metrics(target.loc[mask], probabilities.loc[mask], t_up=t_up, t_down=t_down),
                }
            )

    return {
        "overall": compute_selective_binary_metrics(target, probabilities, t_up=t_up, t_down=t_down),
        "reversal_trend_metrics": slice_metrics,
        "accepted_count_by_bucket": {
            "trend_following": int((accepted & trend_following).sum()),
            "post_first_minute_reversal": int((accepted & ~trend_following).sum()),
        },
        "false_positives_by_bucket": {
            "trend_following": int((false_positive & trend_following).sum()),
            "post_first_minute_reversal": int((false_positive & ~trend_following).sum()),
        },
        "confidence_bucket_metrics": [record for record in bucket_metrics if record["bucket_type"] == "confidence"],
        "hour_bucket_metrics": [record for record in bucket_metrics if record["bucket_type"] == "hour"],
        "side_bucket_metrics": side_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report trend-following and post-first-minute reversal diagnostics.")
    parser.add_argument("--frame", required=True, help="Validation frame parquet/csv with target and ret_1 or fm_ret.")
    parser.add_argument("--probabilities", required=True, help="CSV/parquet containing p_up probabilities.")
    parser.add_argument("--t-up", required=True, type=float)
    parser.add_argument("--t-down", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame_path = Path(args.frame)
    frame = pd.read_parquet(frame_path) if frame_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(frame_path)
    payload = build_reversal_diagnostics(
        frame,
        _load_probabilities(Path(args.probabilities)),
        t_up=args.t_up,
        t_down=args.t_down,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
