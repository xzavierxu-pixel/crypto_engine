from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _aligned_trade_audit(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["timestamp", "event_type", "best_bid", "best_ask", "trade_price", "trade_is_mirror"],
    )
    quotes = frame.loc[
        frame["event_type"].eq("price_change") & frame["best_bid"].notna() & frame["best_ask"].notna(),
        ["timestamp", "best_bid", "best_ask"],
    ].sort_values("timestamp")
    trades = frame.loc[
        frame["event_type"].eq("last_trade_price") & frame["trade_price"].notna(),
        ["timestamp", "trade_price", "trade_is_mirror"],
    ].sort_values("timestamp")
    if quotes.empty or trades.empty:
        return pd.DataFrame()
    aligned = pd.merge_asof(
        trades,
        quotes,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=2),
    ).dropna(subset=["best_bid", "best_ask"])
    aligned["primary_mid"] = (aligned["best_bid"] + aligned["best_ask"]) / 2.0
    aligned["raw_to_primary_error"] = (aligned["trade_price"] - aligned["primary_mid"]).abs()
    aligned["raw_to_complement_error"] = (aligned["trade_price"] - (1.0 - aligned["primary_mid"])).abs()
    aligned["raw_closer_to_primary"] = aligned["raw_to_primary_error"] <= aligned["raw_to_complement_error"]
    aligned["month"] = frame["timestamp"].min().strftime("%Y-%m")
    aligned["market_slug"] = path.stem
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PMData mirror and UP/DOWN price semantics")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "artifacts/pmdata/poly_l2")
    parser.add_argument(
        "--price-metadata",
        type=Path,
        default=ROOT / "artifacts/data_v2/polymarket_prices/btc_updown_5m_first_price.parquet",
    )
    parser.add_argument(
        "--resolved-labels",
        type=Path,
        default=ROOT / "artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/data_v2/polymarket_l2/qa/side_mapping_audit.json",
    )
    parser.add_argument("--sample-markets", type=int, default=100)
    args = parser.parse_args()

    files = sorted(args.source_dir.glob("*.parquet"))
    if not files:
        raise SystemExit("no L2 files found")
    sample_indices = np.linspace(0, len(files) - 1, min(args.sample_markets, len(files)), dtype=int)
    sampled_files = [files[index] for index in sample_indices]
    aligned_frames = [_aligned_trade_audit(path) for path in sampled_files]
    aligned = pd.concat([frame for frame in aligned_frames if len(frame)], ignore_index=True)
    if aligned.empty:
        raise RuntimeError("no quote-aligned trades available for side audit")

    metadata = pd.read_parquet(args.price_metadata)
    labels = pd.read_parquet(args.resolved_labels)
    metadata_slugs = set(metadata["polymarket_slug"].astype(str))
    label_slugs = set(labels["polymarket_slug"].astype(str))
    per_group: list[dict] = []
    mirror_key = aligned["trade_is_mirror"].astype("string").fillna("missing")
    for (month, mirror), group in aligned.groupby([aligned["month"], mirror_key], dropna=False):
        per_group.append(
            {
                "month": month,
                "trade_is_mirror": str(mirror),
                "trade_count": len(group),
                "market_count": group["market_slug"].nunique(),
                "raw_closer_to_primary_share": float(group["raw_closer_to_primary"].mean()),
                "median_raw_to_primary_error": float(group["raw_to_primary_error"].median()),
                "median_raw_to_complement_error": float(group["raw_to_complement_error"].median()),
            }
        )
    known = aligned["trade_is_mirror"].notna()
    missing = ~known
    known_primary_share = float(aligned.loc[known, "raw_closer_to_primary"].mean()) if known.any() else 0.0
    missing_primary_share = float(aligned.loc[missing, "raw_closer_to_primary"].mean()) if missing.any() else None
    passed = known_primary_share >= 0.98
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_market_count": len(sampled_files),
        "quote_aligned_trade_count": len(aligned),
        "metadata_overlap_market_count": sum(path.stem in metadata_slugs for path in sampled_files),
        "resolved_label_overlap_market_count": sum(path.stem in label_slugs for path in sampled_files),
        "primary_token_side": "UP",
        "primary_side_evidence": "polymarket_prices.up_token_id plus yes_mid_price metadata",
        "observed_raw_price_semantics": "mirror and non-mirror raw prices both use primary-UP coordinates",
        "canonicalization": {
            "trade_is_mirror_false": "UP at raw trade_price",
            "trade_is_mirror_true": "DOWN at 1 - raw trade_price",
            "trade_is_mirror_missing": "unrecoverable token side; market excluded from eligible set",
            "book": "UP book; DOWN book is binary complement",
        },
        "known_mirror_raw_closer_to_primary_share": known_primary_share,
        "missing_mirror_raw_closer_to_primary_share": missing_primary_share,
        "pass_threshold": 0.98,
        "passed": passed,
        "blocking_schema_drift": bool(missing.any()),
        "groups": per_group,
    }
    _atomic_json(payload, args.output)
    if not passed:
        raise SystemExit("side mapping audit failed")


if __name__ == "__main__":
    main()
