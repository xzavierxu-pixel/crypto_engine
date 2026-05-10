from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

from execution_engine.config import PolymarketConfig
from src.core.schemas import MarketQuote, OrderRequest


@dataclass(frozen=True)
class PolymarketMarket:
    slug: str
    market_id: str
    yes_token_id: str
    no_token_id: str
    active: bool
    closed: bool
    accepting_orders: bool
    metadata: dict[str, Any]


class PolymarketV2Adapter:
    def __init__(
        self,
        config: PolymarketConfig,
        client: Any | None = None,
        session: requests.Session | None = None,
        force_derive_api_creds: bool = False,
    ) -> None:
        self.config = config
        self.force_derive_api_creds = force_derive_api_creds
        self.api_creds_source = "injected_client" if client is not None else None
        self.client = client or self._build_client()
        self.session = session or requests.Session()

    def _build_client(self) -> Any:
        ApiCreds, ClobClient = _import_clob_client()

        private_key = os.getenv(self.config.private_key_env)
        if self.force_derive_api_creds:
            creds = self._derive_api_creds(ClobClient, private_key)
            self.api_creds_source = "derived_from_private_key"
        else:
            creds = self._load_api_creds(ApiCreds)
        signature_type = self._signature_type()
        funder = self._funder()
        return ClobClient(
            self.config.host,
            chain_id=int(self.config.chain_id),
            key=private_key,
            creds=creds,
            signature_type=signature_type,
            funder=funder,
        )

    def _derive_api_creds(self, clob_client_type: Any, private_key: str | None) -> Any:
        temp_client = clob_client_type(
            self.config.host,
            chain_id=int(self.config.chain_id),
            key=private_key,
        )
        if hasattr(temp_client, "create_or_derive_api_key"):
            return temp_client.create_or_derive_api_key()
        if hasattr(temp_client, "create_or_derive_api_creds"):
            return temp_client.create_or_derive_api_creds()
        raise RuntimeError("Polymarket CLOB client cannot derive API credentials.")

    def _load_api_creds(self, api_creds_type: Any) -> Any | None:
        if self.force_derive_api_creds:
            return None
        api_key = os.getenv(self.config.api_key_env)
        api_secret = os.getenv(self.config.api_secret_env)
        api_passphrase = os.getenv(self.config.api_passphrase_env)
        if api_key and api_secret and api_passphrase:
            self.api_creds_source = "env"
            return api_creds_type(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
        self.api_creds_source = None
        return None

    def _signature_type(self) -> int | None:
        signature_type = self.config.signature_type
        env_signature_type = os.getenv("POLYMARKET_SIGNATURE_TYPE")
        if signature_type is None and env_signature_type:
            signature_type = int(env_signature_type)
        return signature_type

    def _funder(self) -> str | None:
        if self.config.funder:
            return self.config.funder
        env_signature_type = os.getenv("POLYMARKET_SIGNATURE_TYPE")
        signature_type = self.config.signature_type
        if signature_type is None and env_signature_type:
            signature_type = int(env_signature_type)
        if signature_type == 3:
            return os.getenv("DEPOSIT_WALLET_ADDRESS") or _get_expected_deposit_wallet_from_relayer(
                os.getenv(self.config.private_key_env),
                int(self.config.chain_id),
            )
        return os.getenv("POLYMARKET_FUNDER") or os.getenv("DEPOSIT_WALLET_ADDRESS")

    def get_market_by_slug(self, slug: str) -> PolymarketMarket | None:
        response = self.session.get(
            f"{self.config.gamma_base_url.rstrip('/')}/markets",
            params={"slug": slug},
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return None
        return normalize_gamma_market(payload[0], slug=slug)

    def get_orderbook(self, token_id: str, metadata: dict[str, Any] | None = None) -> MarketQuote:
        try:
            orderbook = self.client.get_order_book(token_id)
        except Exception as exc:
            if _is_missing_orderbook_error(exc):
                return MarketQuote(
                    market_id=token_id,
                    yes_price=0.0,
                    no_price=None,
                    metadata={
                        **(metadata or {}),
                        "asset_id": token_id,
                        "best_bid": None,
                        "best_ask": None,
                        "tick_size": None,
                        "orderbook_error": str(exc),
                    },
                )
            raise
        bids = _book_value(orderbook, "bids") or []
        asks = _book_value(orderbook, "asks") or []
        bid_prices = [_level_price(level) for level in bids]
        ask_prices = [_level_price(level) for level in asks]
        best_bid = max((price for price in bid_prices if price is not None), default=None)
        best_ask = min((price for price in ask_prices if price is not None), default=None)
        last_trade = _book_value(orderbook, "last_trade_price")
        last_trade_price = float(last_trade) if last_trade else None
        yes_price = best_ask if best_ask is not None else last_trade_price if last_trade_price is not None else best_bid
        return MarketQuote(
            market_id=token_id,
            yes_price=0.0 if yes_price is None else yes_price,
            no_price=None if yes_price is None else 1.0 - yes_price,
            metadata={
                **(metadata or {}),
                "asset_id": _book_value(orderbook, "asset_id") or token_id,
                "market": _book_value(orderbook, "market"),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "tick_size": _book_value(orderbook, "tick_size"),
                "last_trade_price": last_trade_price,
                "hash": _book_value(orderbook, "hash"),
            },
        )

    def place_limit_order(self, order: OrderRequest) -> dict:
        self._ensure_authenticated()
        try:
            from py_clob_client_v2 import OrderArgs, OrderType, PartialCreateOrderOptions, Side
        except ImportError as exc:  # pragma: no cover - depends on optional deployment package.
            raise ImportError("py-clob-client-v2 is required to place live orders.") from exc

        args = OrderArgs(
            token_id=order.market_id,
            price=order.price,
            size=order.size,
            side=Side.BUY,
        )
        options = PartialCreateOrderOptions(tick_size=str(order.metadata.get("tick_size", "0.01")))
        if hasattr(self.client, "create_and_post_order"):
            response = self.client.create_and_post_order(
                order_args=args,
                options=options,
                order_type=OrderType.GTC,
            )
        else:
            signed = self.client.create_order(args)
            response = self.client.post_order(signed, OrderType.GTC)
        return {
            "request": {
                "token_id": order.market_id,
                "side": "BUY",
                "price": order.price,
                "size": order.size,
            },
            "response": response,
        }

    def _ensure_authenticated(self) -> None:
        if getattr(self.client, "creds", None) is not None:
            return
        if getattr(self.client, "signer", None) is None:
            raise RuntimeError(
                f"Polymarket private key is not configured in {self.config.private_key_env}."
            )
        if self.force_derive_api_creds:
            _, ClobClient = _import_clob_client()
            creds = self._derive_api_creds(ClobClient, os.getenv(self.config.private_key_env))
            self.api_creds_source = "derived_from_private_key"
            self._replace_client_with_creds(creds)
            return
        if hasattr(self.client, "create_or_derive_api_key"):
            creds = self.client.create_or_derive_api_key()
            if hasattr(self.client, "set_api_creds"):
                self.client.set_api_creds(creds)
            else:
                self._replace_client_with_creds(creds)
            return
        if hasattr(self.client, "create_or_derive_api_creds"):
            creds = self.client.create_or_derive_api_creds()
            if hasattr(self.client, "set_api_creds"):
                self.client.set_api_creds(creds)
            else:
                self._replace_client_with_creds(creds)
            return
        raise RuntimeError("Polymarket CLOB client cannot derive API credentials.")

    def _replace_client_with_creds(self, creds: Any) -> None:
        _, ClobClient = _import_clob_client()
        self.client = ClobClient(
            self.config.host,
            chain_id=int(self.config.chain_id),
            key=os.getenv(self.config.private_key_env),
            creds=creds,
            signature_type=self._signature_type(),
            funder=self._funder(),
        )


def _import_clob_client() -> tuple[Any, Any]:
    try:
        from py_clob_client_v2.client import ClobClient
        from py_clob_client_v2.clob_types import ApiCreds

        return ApiCreds, ClobClient
    except ImportError:
        pass
    try:
        from py_clob_client_v2 import ApiCreds, ClobClient

        return ApiCreds, ClobClient
    except ImportError as exc:  # pragma: no cover - depends on optional deployment package.
        raise ImportError(
            "py-clob-client-v2 is required for live Polymarket execution. "
            "Install it in the deployment virtualenv before using live mode."
        ) from exc


def normalize_gamma_market(market: dict[str, Any], slug: str) -> PolymarketMarket:
    tokens = _extract_tokens(market)
    if len(tokens) < 2:
        raise RuntimeError(f"Gamma market '{slug}' did not include enough token information.")
    yes = next((token for token in tokens if str(token.get("outcome", "")).lower() in {"yes", "up"}), tokens[0])
    no = next((token for token in tokens if str(token.get("outcome", "")).lower() in {"no", "down"}), tokens[1])
    return PolymarketMarket(
        slug=slug,
        market_id=str(market["id"]),
        yes_token_id=str(yes.get("token_id")),
        no_token_id=str(no.get("token_id")),
        active=bool(market.get("active", False)),
        closed=bool(market.get("closed", False)),
        accepting_orders=bool(market.get("acceptingOrders", market.get("accepting_orders", False))),
        metadata={
            "condition_id": market.get("conditionId"),
            "yes_outcome": yes.get("outcome"),
            "no_outcome": no.get("outcome"),
            "best_bid": _optional_float(market.get("bestBid")),
            "best_ask": _optional_float(market.get("bestAsk")),
            "liquidity_clob": _optional_float(market.get("liquidityClob")),
            "order_min_size": market.get("orderMinSize"),
            "tokens": tokens,
        },
    )


def _extract_tokens(market: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = market.get("tokens")
    if isinstance(tokens, list) and tokens:
        return tokens
    import json

    outcomes = _json_list(market.get("outcomes"), json)
    prices = _json_list(market.get("outcomePrices"), json)
    token_ids = _json_list(market.get("clobTokenIds"), json)
    return [
        {
            "token_id": token_ids[index] if index < len(token_ids) else None,
            "outcome": outcome,
            "price": float(prices[index]) if index < len(prices) else None,
        }
        for index, outcome in enumerate(outcomes)
    ]


def _json_list(value: Any, json_module) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json_module.loads(value)
    raise TypeError(f"Unsupported JSON list payload type: {type(value)!r}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _book_value(orderbook: Any, key: str) -> Any:
    if isinstance(orderbook, dict):
        return orderbook.get(key)
    return getattr(orderbook, key, None)


def _level_price(level: Any) -> float | None:
    value = level.get("price") if isinstance(level, dict) else getattr(level, "price", None)
    return None if value is None else float(value)


def _is_missing_orderbook_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status_code == 404 or "no orderbook exists" in text


def _get_expected_deposit_wallet_from_relayer(private_key: str | None, chain_id: int) -> str | None:
    if not private_key:
        return None
    try:
        from py_builder_relayer_client.client import RelayClient
    except ImportError as exc:  # pragma: no cover - depends on optional deployment package.
        raise ImportError(
            "py-builder-relayer-client is required to derive DEPOSIT_WALLET_ADDRESS. "
            "Set DEPOSIT_WALLET_ADDRESS explicitly or install the relayer client."
        ) from exc

    relayer_url = (
        os.getenv("POLYMARKET_RELAYER_URL")
        or os.getenv("RELAYER_URL")
        or "https://relayer-v2.polymarket.com/"
    )

    builder_config = _load_builder_config()
    relayer = RelayClient(relayer_url, chain_id, private_key, builder_config)
    if not hasattr(relayer, "get_expected_deposit_wallet"):
        raise RuntimeError(
            "Installed py_builder_relayer_client does not expose get_expected_deposit_wallet(). "
            "Upgrade the server dependency or set DEPOSIT_WALLET_ADDRESS explicitly."
        )
    deposit_wallet = relayer.get_expected_deposit_wallet()
    print("DEPOSIT_WALLET_ADDRESS =", deposit_wallet)
    return deposit_wallet


def _load_builder_config() -> Any | None:
    key = os.getenv("BUILDER_API_KEY")
    secret = os.getenv("BUILDER_SECRET")
    passphrase = os.getenv("BUILDER_PASS_PHRASE")
    if not (key and secret and passphrase):
        return None
    try:
        from py_builder_signing_sdk.config import BuilderApiKeyCreds, BuilderConfig
    except ImportError as exc:  # pragma: no cover - depends on optional deployment package.
        raise ImportError(
            "py-builder-signing-sdk is required when BUILDER_API_KEY credentials are configured."
        ) from exc
    return BuilderConfig(
        local_builder_creds=BuilderApiKeyCreds(
            key=key,
            secret=secret,
            passphrase=passphrase,
        )
    )
