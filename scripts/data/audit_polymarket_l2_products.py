from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.polymarket_l2 import (  # noqa: E402
    PolymarketL2Config,
    build_first_minute_features,
    build_trade_products,
)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _load_config(path: Path) -> PolymarketL2Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))["polymarket_l2"]
    names = {field.name for field in fields(PolymarketL2Config)}
    values = {key: value for key, value in payload.items() if key in names}
    for key in ("depth_levels", "rolling_windows_seconds"):
        if key in values:
            values[key] = tuple(values[key])
    return PolymarketL2Config(**values)


def _equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    common = sorted(set(left.columns).intersection(right.columns))
    left = left[common].reset_index(drop=True)
    right = right[common].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=1e-10, atol=1e-12)
        return True
    except AssertionError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute a deterministic 100-market L2 product audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "artifacts/pmdata/poly_l2")
    parser.add_argument("--l2-root", type=Path, default=ROOT / "artifacts/data_v2/polymarket_l2")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()
    config = _load_config(args.config)
    eligible = pd.read_parquet(args.l2_root / "l2_eligible_markets.parquet").sort_values("market_t0")
    indices = np.linspace(0, len(eligible) - 1, min(args.sample_size, len(eligible)), dtype=int)
    sample = eligible.iloc[indices]
    built_features = pd.concat([pd.read_parquet(path) for path in (args.l2_root / "first_minute_features").rglob("*.parquet")])
    built_refs = pd.concat([pd.read_parquet(path) for path in (args.l2_root / "first_minute_price_reference").rglob("*.parquet")])
    built_lows = pd.concat([pd.read_parquet(path) for path in (args.l2_root / "future_four_minute_lows").rglob("*.parquet")])
    failures = []
    future_append_failures = []
    for row in sample.itertuples(index=False):
        raw = pd.read_parquet(args.source_dir / f"{row.market_slug}.parquet")
        feature = build_first_minute_features(raw, config)
        reference, low = build_trade_products(raw, config)
        checks = {
            "feature": _equal(feature, built_features.loc[built_features["market_slug"].eq(row.market_slug)]),
            "reference": _equal(reference, built_refs.loc[built_refs["market_slug"].eq(row.market_slug)]),
            "future_low": _equal(low, built_lows.loc[built_lows["market_slug"].eq(row.market_slug)]),
        }
        if not all(checks.values()):
            failures.append({"market_slug": row.market_slug, "checks": checks})
        cutoff = feature["feature_cutoff_time"].iloc[0]
        pre_cutoff = raw.loc[pd.to_datetime(raw["timestamp"], utc=True) <= cutoff]
        pre_feature = build_first_minute_features(pre_cutoff, config)
        if not _equal(feature, pre_feature):
            future_append_failures.append(row.market_slug)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(sample),
        "sampling": "deterministic evenly spaced over frozen eligible markets",
        "recomputation_failure_count": len(failures),
        "future_append_invariance_failure_count": len(future_append_failures),
        "failures": failures,
        "future_append_failures": future_append_failures,
        "passed": not failures and not future_append_failures,
    }
    _atomic_json(payload, args.l2_root / "qa/random_100_market_recomputation.json")
    if not payload["passed"]:
        raise SystemExit("L2 product audit failed")


if __name__ == "__main__":
    main()
