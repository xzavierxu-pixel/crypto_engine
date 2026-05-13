from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution_engine.config import load_execution_config
from execution_engine.polymarket_v2 import PolymarketV2Adapter
from execution_engine.run_once import build_btc_5m_slug
from src.core.schemas import OrderRequest


@dataclass(frozen=True)
class EnvStatus:
    name: str
    present: bool


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Secrets file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ[key] = value


def current_window_start(now: datetime | None = None) -> datetime:
    ts = (now or datetime.now(UTC)).astimezone(UTC).replace(second=0, microsecond=0)
    minute = ts.minute - (ts.minute % 5)
    return ts.replace(minute=minute)


def env_status(config_path: str, secrets_path: str, private_key_env: str, api_key_env: str) -> dict[str, Any]:
    names = [
        private_key_env,
        api_key_env,
        "CLOB_SECRET",
        "CLOB_PASS_PHRASE",
        "POLYMARKET_SIGNATURE_TYPE",
        "POLYMARKET_FUNDER",
        "DEPOSIT_WALLET_ADDRESS",
    ]
    return {
        "config_path": config_path,
        "secrets_path": secrets_path,
        "loaded": [EnvStatus(name, bool(os.getenv(name))).__dict__ for name in names],
    }


def short_address(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 12:
        return value
    return f"{value[:6]}...{value[-4:]}"


def client_diagnostics(adapter: PolymarketV2Adapter) -> dict[str, Any]:
    signer = getattr(getattr(adapter.client, "signer", None), "address", lambda: None)()
    builder = getattr(adapter.client, "builder", None)
    signature_type = getattr(builder, "signature_type", getattr(adapter.client, "signature_type", None))
    funder = getattr(builder, "funder", getattr(adapter.client, "funder", None))
    creds = getattr(adapter.client, "creds", None)
    return {
        "signer": short_address(signer),
        "signature_type": None if signature_type is None else int(signature_type),
        "funder": short_address(funder),
        "funder_present": bool(funder),
        "api_creds_present": bool(creds),
        "api_key": short_address(getattr(creds, "api_key", None)),
        "api_creds_source": getattr(adapter, "api_creds_source", None),
    }


def build_diagnostics(
    config_path: str,
    secrets_path: str,
    env_overrides: dict[str, str | None] | None = None,
    force_derive_api_creds: bool = True,
) -> dict[str, Any]:
    config = load_execution_config(config_path)
    load_env_file(Path(secrets_path))
    apply_env_overrides(env_overrides)
    payload: dict[str, Any] = {
        "env": env_status(
            config_path,
            secrets_path,
            config.polymarket.private_key_env,
            config.polymarket.api_key_env,
        ),
    }
    try:
        adapter = PolymarketV2Adapter(config.polymarket, force_derive_api_creds=force_derive_api_creds)
        adapter._ensure_authenticated()
        payload["client"] = client_diagnostics(adapter)
        payload["relayer"] = relayer_diagnostics(config.polymarket.chain_id, config.polymarket.private_key_env)
    except Exception as exc:  # pragma: no cover - diagnostics path for deployment.
        payload["client_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return payload


def relayer_diagnostics(chain_id: int, private_key_env: str) -> dict[str, Any]:
    try:
        from py_builder_relayer_client.client import RelayClient
        from py_builder_relayer_client.models import TransactionType
    except Exception as exc:  # pragma: no cover - optional deployment dependency.
        return {"error": {"type": type(exc).__name__, "message": str(exc)}}

    private_key = os.getenv(private_key_env)
    relayer_url = os.getenv("RELAYER_URL") or os.getenv("POLYMARKET_RELAYER_URL") or "https://relayer-v2.polymarket.com/"
    relayer = RelayClient(relayer_url, int(chain_id), private_key)
    deposit_wallet = relayer.get_expected_deposit_wallet()
    payload: dict[str, Any] = {
        "url": relayer_url,
        "deposit_wallet": short_address(deposit_wallet),
    }
    try:
        payload["deposit_wallet_deployed"] = relayer.get_deployed(
            deposit_wallet,
            TransactionType.WALLET.value,
        )
    except Exception as exc:
        payload["deployed_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return payload


def submit_min_price_order(
    config_path: str,
    secrets_path: str,
    token_side: str,
    target_window_start: datetime | None,
    env_overrides: dict[str, str | None] | None = None,
    force_derive_api_creds: bool = True,
) -> dict[str, Any]:
    config = load_execution_config(config_path)
    load_env_file(Path(secrets_path))
    apply_env_overrides(env_overrides)
    adapter = PolymarketV2Adapter(config.polymarket, force_derive_api_creds=force_derive_api_creds)

    window_start = target_window_start or current_window_start()
    slug, window_start, window_end = build_btc_5m_slug(window_start, offset_windows=0)
    market = adapter.get_market_by_slug(slug)
    if market is None:
        raise RuntimeError(f"Market not found for slug: {slug}")
    if not market.active or market.closed or not market.accepting_orders:
        raise RuntimeError(
            "Market is not accepting orders: "
            f"slug={slug} active={market.active} closed={market.closed} "
            f"accepting_orders={market.accepting_orders}"
        )

    side = token_side.upper()
    if side == "YES":
        token_id = market.yes_token_id
    elif side == "NO":
        token_id = market.no_token_id
    else:
        raise ValueError("--token-side must be YES or NO")

    quote = adapter.get_orderbook(
        token_id,
        metadata={
            "slug": market.slug,
            "market_id": market.market_id,
            "yes_token_id": market.yes_token_id,
            "no_token_id": market.no_token_id,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
        },
    )
    tick_size = float(quote.metadata.get("tick_size") or config.orders.tick_size_default)
    order = OrderRequest(
        market_id=token_id,
        side=side,
        price=float(config.orders.min_price),
        size=float(config.orders.first.size),
        signal_t0=window_start - timedelta(minutes=5),
        metadata={
            "leg": "auth_test_min_price",
            "tick_size": tick_size,
            "best_bid": quote.metadata.get("best_bid"),
            "best_ask": quote.metadata.get("best_ask"),
        },
    )
    response = adapter.place_limit_order(order)

    return {
        "submitted": True,
        "env": env_status(
            config_path,
            secrets_path,
            config.polymarket.private_key_env,
            config.polymarket.api_key_env,
        ),
        "client": client_diagnostics(adapter),
        "market": {
            "slug": market.slug,
            "market_id": market.market_id,
            "token_side": side,
            "target_token_id": token_id,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "best_bid": quote.metadata.get("best_bid"),
            "best_ask": quote.metadata.get("best_ask"),
            "tick_size": tick_size,
        },
        "order": {
            "price": order.price,
            "size": order.size,
        },
        "response": response,
    }


def apply_env_overrides(env_overrides: dict[str, str | None] | None) -> None:
    if not env_overrides:
        return
    for key, value in env_overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one minimum-price Polymarket order to test live auth.")
    parser.add_argument("--config", default="execution_engine/config.yaml")
    parser.add_argument("--secrets", default="execution_engine/secrets.env")
    parser.add_argument("--token-side", choices=["YES", "NO"], default="YES")
    parser.add_argument("--target-window-start", default=None)
    parser.add_argument("--signature-type", type=int, default=None)
    parser.add_argument("--relayer-url", default=None)
    parser.add_argument("--relayer-api-key", default=None)
    parser.add_argument("--relayer-api-key-address", default=None)
    parser.add_argument("--funder", default=None)
    parser.add_argument("--deposit-wallet", default=None)
    parser.add_argument("--clear-deposit-wallet", action="store_true")
    parser.add_argument("--use-env-api-creds", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    target_window_start = (
        None
        if args.target_window_start is None
        else datetime.fromisoformat(args.target_window_start.replace("Z", "+00:00")).astimezone(UTC)
    )
    env_overrides = {
        "POLYMARKET_SIGNATURE_TYPE": None if args.signature_type is None else str(args.signature_type),
        "RELAYER_URL": args.relayer_url,
        "RELAYER_API_KEY": args.relayer_api_key,
        "RELAYER_API_KEY_ADDRESS": args.relayer_api_key_address,
        "POLYMARKET_FUNDER": args.funder,
        "DEPOSIT_WALLET_ADDRESS": args.deposit_wallet,
    }
    if args.clear_deposit_wallet:
        env_overrides["DEPOSIT_WALLET_ADDRESS"] = None
    env_overrides = {
        key: value
        for key, value in env_overrides.items()
        if value is not None or (key == "DEPOSIT_WALLET_ADDRESS" and args.clear_deposit_wallet)
    }
    try:
        result = submit_min_price_order(
            args.config,
            args.secrets,
            args.token_side,
            target_window_start,
            env_overrides,
            force_derive_api_creds=not args.use_env_api_creds,
        )
    except Exception as exc:
        failure = {
            "submitted": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            **build_diagnostics(
                args.config,
                args.secrets,
                env_overrides,
                force_derive_api_creds=not args.use_env_api_creds,
            ),
        }
        print(json.dumps(failure, indent=2, ensure_ascii=False, default=str))
        raise SystemExit(1) from exc
    if args.print_json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"submitted={result['submitted']}")
        print(f"slug={result['market']['slug']}")
        print(f"token_side={result['market']['token_side']}")
        print(f"price={result['order']['price']}")
        print(f"size={result['order']['size']}")


if __name__ == "__main__":
    main()
