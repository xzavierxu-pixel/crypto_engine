from __future__ import annotations

from datetime import UTC, datetime

from src.core.config import load_settings
from src.core.schemas import Signal
from src.execution.mappers.btc_5m_polymarket import BTC5mPolymarketMapper


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, slug_payload, events_payload) -> None:
        self.slug_payload = slug_payload
        self.events_payload = events_payload
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params or {}))
        if url.endswith("/markets"):
            return FakeResponse(self.slug_payload)
        if url.endswith("/events"):
            return FakeResponse(self.events_payload)
        raise AssertionError(f"Unexpected URL: {url}")


def make_signal() -> Signal:
    return Signal(
        asset="BTC/USDT",
        horizon="5m",
        t0=datetime(2026, 4, 8, 13, 45, tzinfo=UTC),
        p_up=0.57,
        model_version="m1",
        feature_version="v1",
    )


def test_mapper_uses_slug_lookup_and_normalizes_gamma_market() -> None:
    settings = load_settings()
    gamma_market = {
        "id": "1907386",
        "conditionId": "0xabc",
        "slug": "btc-updown-5m-1775655900",
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "outcomes": "[\"Up\", \"Down\"]",
        "outcomePrices": "[\"0.515\", \"0.485\"]",
        "clobTokenIds": "[\"up-token\", \"down-token\"]",
    }
    mapper = BTC5mPolymarketMapper(settings, session=FakeSession([gamma_market], []))

    result = mapper.map_signal(make_signal())

    assert result["market_id"] == "1907386"
    assert result["condition_id"] == "0xabc"
    assert result["slug"] == "btc-updown-5m-1775655900"
    assert result["yes_token_id"] == "up-token"
    assert result["no_token_id"] == "down-token"
    assert result["yes_outcome"] == "Up"
    assert result["accepting_orders"] is True


def test_mapper_falls_back_to_events_scan_when_slug_lookup_misses() -> None:
    settings = load_settings()
    events_payload = [
        {
            "markets": [
                {
                    "id": "1907386",
                    "conditionId": "0xabc",
                    "slug": "btc-updown-5m-1775655900",
                    "startDate": "2026-04-08T13:45:00Z",
                    "endDate": "2026-04-08T13:50:00Z",
                    "active": True,
                    "closed": False,
                    "acceptingOrders": True,
                    "tokens": [
                        {"token_id": "up-token", "outcome": "Up", "price": 0.515},
                        {"token_id": "down-token", "outcome": "Down", "price": 0.485},
                    ],
                }
            ]
        }
    ]
    session = FakeSession([], events_payload)
    mapper = BTC5mPolymarketMapper(settings, session=session)

    result = mapper.map_signal(make_signal())

    assert result["market_id"] == "1907386"
    assert result["yes_token_id"] == "up-token"
    assert result["no_token_id"] == "down-token"
    assert any(url.endswith("/events") for url, _ in session.calls)
