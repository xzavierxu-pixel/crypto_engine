from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.polymarket_l2 import market_t0_from_slug  # noqa: E402


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


def inspect_market(path: Path, coverage_tolerance_ms: int = 1000) -> dict:
    slug = path.stem
    t0 = market_t0_from_slug(slug)
    end = t0 + pd.Timedelta(seconds=300)
    cutoff = t0 + pd.Timedelta(seconds=60)
    frame = pq.read_table(path, columns=["timestamp", "event_type", "trade_is_mirror"]).to_pandas()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    trades = frame.loc[frame["event_type"].eq("last_trade_price")]
    first = trades.loc[trades["timestamp"].between(t0, cutoff, inclusive="both")]
    future = trades.loc[(trades["timestamp"] > cutoff) & (trades["timestamp"] <= end)]
    reasons = []
    tolerance = pd.Timedelta(milliseconds=coverage_tolerance_ms)
    if frame["timestamp"].min() > t0 + tolerance:
        reasons.append("missing_market_start_coverage")
    if frame["timestamp"].max() < end - tolerance:
        reasons.append("missing_market_end_coverage")
    if trades["trade_is_mirror"].isna().any():
        reasons.append("unrecoverable_trade_side")
    for name, window in (("first", first), ("future", future)):
        if not window["trade_is_mirror"].eq(False).any():
            reasons.append(f"missing_{name}_up_trade")
        if not window["trade_is_mirror"].eq(True).any():
            reasons.append(f"missing_{name}_down_trade")
    return {
        "market_slug": slug,
        "market_t0": t0,
        "source_path": str(path),
        "source_row_count": len(frame),
        "first_window_trade_count": len(first),
        "future_window_trade_count": len(future),
        "eligible": not reasons,
        "exclusion_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the leakage-safe L2 eligible market universe")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "artifacts/pmdata/poly_l2")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/data_v2/polymarket_l2/l2_eligible_markets.parquet"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--coverage-tolerance-ms", type=int, default=1000)
    args = parser.parse_args()
    paths = sorted(args.source_dir.glob("*.parquet"))
    if args.limit:
        paths = paths[: args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(lambda path: inspect_market(path, args.coverage_tolerance_ms), paths))
    audit = pd.DataFrame(rows).sort_values("market_t0", kind="stable")
    eligible = audit.loc[audit["eligible"]].drop(columns=["eligible", "exclusion_reasons"])
    _atomic_parquet(eligible, args.output)
    _atomic_parquet(audit, args.output.with_name("l2_market_eligibility_audit.parquet"))
    reason_counts: dict[str, int] = {}
    for reasons in audit.loc[~audit["eligible"], "exclusion_reasons"]:
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    _atomic_json(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_market_count": len(audit),
            "eligible_market_count": len(eligible),
            "excluded_market_count": int((~audit["eligible"]).sum()),
            "exclusion_reason_counts": reason_counts,
            "eligible_start": eligible["market_t0"].min() if len(eligible) else None,
            "eligible_end": eligible["market_t0"].max() if len(eligible) else None,
            "coverage_tolerance_ms": args.coverage_tolerance_ms,
            "eligibility_rule": "complete [t0,t0+60s] and (t0+60s,t0+300s] windows within configured capture tolerance; explicit mirror side and both side trades",
        },
        args.output.with_name("l2_eligible_markets_manifest.json"),
    )


if __name__ == "__main__":
    main()
