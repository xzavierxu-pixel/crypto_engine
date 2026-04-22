from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import requests

from src.core.config import Settings
from src.core.schemas import Signal
from src.execution.mappers.base import MarketMapper


class BTC5mPolymarketMapper(MarketMapper):
    name = "btc_5m_polymarket"

    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.config = settings.execution.polymarket
        self.base_url = self.config["gamma_base_url"].rstrip("/")
        self.timeout = float(self.config.get("gamma_timeout_seconds", 5))
        self.session = session or requests.Session()

    def get_window_bounds(self, signal: Signal) -> tuple[datetime, datetime]:
        t0 = signal.t0.astimezone(UTC).replace(second=0, microsecond=0)
        minute = t0.minute - (t0.minute % 5)
        window_start = t0.replace(minute=minute)
        window_end = window_start + timedelta(minutes=5)
        return window_start, window_end

    def build_btc_5m_slug(self, window_start: datetime) -> str:
        return f"btc-updown-5m-{int(window_start.timestamp())}"

    def get_market_by_slug(self, slug: str) -> dict | None:
        response = self.session.get(
            f"{self.base_url}/markets",
            params={"slug": slug},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return None
        return payload[0]

    def fallback_find_market(self, window_start: datetime, window_end: datetime) -> dict | None:
        response = self.session.get(
            f"{self.base_url}/events",
            params={
                "active": "true",
                "closed": "false",
                "limit": 100,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        events = response.json()

        for event in events:
            for market in event.get("markets", []):
                try:
                    start = datetime.fromisoformat(market["startDate"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
                except Exception:
                    continue

                slug = str(market.get("slug", ""))
                if start == window_start and end == window_end and "btc" in slug and "5m" in slug:
                    return market

        return None

    def _parse_json_list(self, value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return json.loads(value)
        raise TypeError(f"Unsupported list payload type: {type(value)!r}")

    def _extract_tokens(self, market: dict) -> list[dict]:
        tokens = market.get("tokens")
        if isinstance(tokens, list) and tokens:
            return tokens

        outcomes = self._parse_json_list(market.get("outcomes"))
        prices = self._parse_json_list(market.get("outcomePrices"))
        token_ids = self._parse_json_list(market.get("clobTokenIds"))

        normalized: list[dict] = []
        for index, outcome in enumerate(outcomes):
            normalized.append(
                {
                    "token_id": token_ids[index] if index < len(token_ids) else None,
                    "outcome": outcome,
                    "price": float(prices[index]) if index < len(prices) else None,
                }
            )
        return normalized

    def _normalize_market(
        self,
        market: dict,
        window_start: datetime,
        window_end: datetime,
        slug: str,
    ) -> dict:
        tokens = self._extract_tokens(market)
        if len(tokens) < 2:
            raise RuntimeError(f"Gamma market '{slug}' did not include enough token information.")

        up_token = next(
            (token for token in tokens if str(token.get("outcome", "")).lower() in {"yes", "up"}),
            tokens[0],
        )
        down_token = next(
            (token for token in tokens if str(token.get("outcome", "")).lower() in {"no", "down"}),
            tokens[1],
        )

        return {
            "market_id": str(market["id"]),
            "condition_id": market.get("conditionId"),
            "slug": slug,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "asset": "BTC/USDT",
            "horizon": "5m",
            "yes_token_id": up_token.get("token_id"),
            "no_token_id": down_token.get("token_id"),
            "yes_outcome": up_token.get("outcome"),
            "no_outcome": down_token.get("outcome"),
            "yes_price": float(up_token["price"]) if up_token.get("price") is not None else None,
            "no_price": float(down_token["price"]) if down_token.get("price") is not None else None,
            "active": market.get("active", False),
            "closed": market.get("closed", False),
            "accepting_orders": market.get("acceptingOrders", market.get("accepting_orders", False)),
            "best_bid": float(market["bestBid"]) if market.get("bestBid") is not None else None,
            "best_ask": float(market["bestAsk"]) if market.get("bestAsk") is not None else None,
            "liquidity_clob": float(market["liquidityClob"]) if market.get("liquidityClob") is not None else None,
            "order_min_size": market.get("orderMinSize"),
            "tokens": tokens,
        }

    def map_signal(self, signal: Signal) -> dict:
        if signal.horizon != "5m":
            raise ValueError(
                f"{self.name} only supports 5m signals, received horizon='{signal.horizon}'."
            )
        window_start, window_end = self.get_window_bounds(signal)
        slug = self.build_btc_5m_slug(window_start)

        market = self.get_market_by_slug(slug)
        if market is None:
            market = self.fallback_find_market(window_start, window_end)
            if market is None:
                raise LookupError(
                    "No current BTC 5-minute market found "
                    f"for window_start={window_start.isoformat()} window_end={window_end.isoformat()}."
                )
            slug = str(market["slug"])

        return self._normalize_market(
            market,
            window_start=window_start,
            window_end=window_end,
            slug=slug,
        )
