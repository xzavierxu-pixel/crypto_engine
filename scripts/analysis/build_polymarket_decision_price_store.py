from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow.parquet as pq


USER_AGENT = "crypto-engine-polymarket-price-eval/1.0"
GAMMA_MARKET_URL = "https://gamma-api.polymarket.com/markets/{market_id}"
GAMMA_MARKET_BY_SLUG_URL = "https://gamma-api.polymarket.com/markets/slug/{slug}"
CLOB_PRICE_HISTORY_URL = "https://clob.polymarket.com/prices-history"


def _read_json_url(url: str, *, retries: int = 3, timeout: int = 20) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last_error


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _collect_slugs_from_artifact(artifact_dir: Path) -> set[str]:
    slugs: set[str] = set()
    for name in ("validation_predictions.parquet", "validation_frame.parquet"):
        path = artifact_dir / name
        if not path.exists():
            continue
        columns = set(pq.ParquetFile(path).schema_arrow.names)
        wanted = [column for column in ("polymarket_slug", "market_t0", "timestamp") if column in columns]
        if not wanted:
            continue
        frame = pd.read_parquet(path, columns=wanted)
        if "polymarket_slug" in frame:
            slugs.update(frame["polymarket_slug"].dropna().astype(str))
    return slugs


def _collect_slugs_from_replay(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    slugs: set[str] = set()
    for combo in payload.get("combos", []):
        for row in combo.get("rows", []):
            slug = row.get("slug") or row.get("polymarket_slug")
            if slug:
                slugs.add(str(slug))
    return slugs


def _select_up_token(market: dict[str, Any]) -> tuple[str | None, list[str], list[str]]:
    outcomes = [str(value) for value in _json_list(market.get("outcomes"))]
    token_ids = [str(value) for value in _json_list(market.get("clobTokenIds"))]
    if not token_ids:
        return None, outcomes, token_ids
    for idx, outcome in enumerate(outcomes):
        if outcome.strip().lower() in {"up", "yes"} and idx < len(token_ids):
            return token_ids[idx], outcomes, token_ids
    return token_ids[0], outcomes, token_ids


def _fetch_one(row: dict[str, Any]) -> dict[str, Any]:
    market_id = None if pd.isna(row.get("market_id")) else str(row.get("market_id"))
    slug = str(row["polymarket_slug"])
    market_t0 = pd.Timestamp(row["market_t0"]).tz_convert("UTC")
    start_ts = int(market_t0.timestamp())
    end_ts = start_ts + 300
    result: dict[str, Any] = {
        "polymarket_slug": str(row["polymarket_slug"]),
        "market_t0": market_t0.isoformat(),
        "market_id": market_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "source": "clob_prices_history_first_point_in_market_window",
    }
    try:
        market = (
            _read_json_url(GAMMA_MARKET_URL.format(market_id=market_id))
            if market_id
            else _read_json_url(GAMMA_MARKET_BY_SLUG_URL.format(slug=slug))
        )
        market_id = str(market.get("id") or market_id)
        up_token_id, outcomes, token_ids = _select_up_token(market)
        result.update(
            {
                "market_id": market_id,
                "up_token_id": up_token_id,
                "outcomes": json.dumps(outcomes),
                "clob_token_ids": json.dumps(token_ids),
            }
        )
        if up_token_id is None:
            result.update({"price_status": "missing_token"})
            return result
        query = urlencode({"market": up_token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 1})
        history_payload = _read_json_url(f"{CLOB_PRICE_HISTORY_URL}?{query}")
        history = history_payload.get("history") or []
        if not history:
            result.update({"price_status": "missing_history"})
            return result
        first = min(history, key=lambda item: int(item["t"]))
        yes_mid_price = float(first["p"])
        result.update(
            {
                "price_status": "ok",
                "price_ts": int(first["t"]),
                "price_time": pd.Timestamp(int(first["t"]), unit="s", tz="UTC").isoformat(),
                "seconds_from_t0": int(first["t"]) - start_ts,
                "yes_mid_price": yes_mid_price,
                "down_mid_price": 1.0 - yes_mid_price,
                "history_point_count": int(len(history)),
            }
        )
        return result
    except Exception as exc:
        result.update({"price_status": "error", "error": str(exc)})
        return result


def build_price_store(
    *,
    label_store_path: Path,
    artifact_dirs: list[Path],
    replay_jsons: list[Path],
    output_path: Path,
    workers: int,
    limit: int | None,
) -> pd.DataFrame:
    label_store = pd.read_parquet(label_store_path)
    needed_slugs: set[str] = set()
    for artifact_dir in artifact_dirs:
        needed_slugs.update(_collect_slugs_from_artifact(artifact_dir))
    for replay_json in replay_jsons:
        needed_slugs.update(_collect_slugs_from_replay(replay_json))
    if not needed_slugs:
        needed_slugs.update(label_store["polymarket_slug"].dropna().astype(str))

    target = label_store[label_store["polymarket_slug"].astype(str).isin(needed_slugs)].copy()
    target = target.drop_duplicates(subset=["polymarket_slug"])
    missing_slugs = sorted(needed_slugs - set(target["polymarket_slug"].dropna().astype(str)))
    if missing_slugs:
        extra = pd.DataFrame(
            {
                "polymarket_slug": missing_slugs,
                "market_t0": [pd.Timestamp(int(slug.rsplit("-", 1)[-1]), unit="s", tz="UTC") for slug in missing_slugs],
                "market_id": [None] * len(missing_slugs),
            }
        )
        target = pd.concat([target[["polymarket_slug", "market_t0", "market_id"]], extra], ignore_index=True)
    target = target.sort_values("market_t0")
    if limit is not None:
        target = target.head(limit)

    existing = pd.DataFrame()
    if output_path.exists():
        existing = pd.read_parquet(output_path)
    done_slugs = set(existing.get("polymarket_slug", pd.Series(dtype=str)).dropna().astype(str))
    pending = target[~target["polymarket_slug"].astype(str).isin(done_slugs)]

    records: list[dict[str, Any]] = []
    rows = pending[["polymarket_slug", "market_t0", "market_id"]].to_dict("records")
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = {executor.submit(_fetch_one, row): row for row in rows}
        for idx, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if idx % 100 == 0:
                print(json.dumps({"fetched": idx, "pending": len(rows)}, sort_keys=True))

    new_frame = pd.DataFrame.from_records(records)
    combined = pd.concat([existing, new_frame], ignore_index=True) if not existing.empty else new_frame
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["polymarket_slug"], keep="last").sort_values("market_t0")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    summary = {
        "output": str(output_path),
        "requested_slug_count": int(len(target)),
        "existing_count": int(len(existing)),
        "fetched_count": int(len(new_frame)),
        "total_count": int(len(combined)),
        "status_counts": combined["price_status"].value_counts(dropna=False).to_dict() if not combined.empty else {},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Polymarket first-price store from CLOB prices-history.")
    parser.add_argument(
        "--label-store",
        default="artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet",
    )
    parser.add_argument("--artifact-dir", action="append", default=[])
    parser.add_argument("--replay-json", action="append", default=[])
    parser.add_argument(
        "--output",
        default="artifacts/data_v2/polymarket_prices/btc_updown_5m_first_price.parquet",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build_price_store(
        label_store_path=Path(args.label_store),
        artifact_dirs=[Path(path) for path in args.artifact_dir],
        replay_jsons=[Path(path) for path in args.replay_json],
        output_path=Path(args.output),
        workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
