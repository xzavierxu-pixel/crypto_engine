# Execution Engine Next Steps

## Binance aggTrade WebSocket Cache

The current runtime now treats REST aggTrade completeness as a hard online/offline alignment requirement: if `sl_agg_*` features are used, inference should only run when agg trades are available through the last pre-signal second.

Next step: add a lightweight Binance `aggTrade` websocket cache writer.

Target design:

- Keep REST as the backfill and recovery path.
- Run a persistent websocket process that writes `agg_trades.parquet` or an append-only local store using `agg_trade_id` as the dedupe key.
- On each `run_once`, read the websocket cache first, then use REST `fromId` only to fill gaps after the latest cached `agg_trade_id`.
- If websocket disconnects or a gap is detected, mark the cache unhealthy and require REST to recover through `t0-1s` before inference.
- Continue to fail closed for the current window if agg trades cannot be proven complete by `binance.agg_trade_wait_seconds`.

Reason:

REST `/api/v3/aggTrades` is adequate for backfill, but it is not the best source for second-boundary completeness at order time. A websocket cache should reduce trigger-time wait while preserving the same feature semantics used offline.
