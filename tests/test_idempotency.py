from __future__ import annotations

from pathlib import Path

from src.execution.idempotency import IdempotencyStore


def test_idempotency_store_records_and_reads_keys() -> None:
    path = Path("artifacts/test_state/idempotency.json")
    if path.exists():
        path.unlink()
    store = IdempotencyStore(path)

    assert store.has("k1") is False
    store.record("k1", {"market_id": "yes-1"})
    assert store.has("k1") is True
