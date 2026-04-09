from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


class IdempotencyStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, payload: dict[str, dict]) -> None:
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def has(self, key: str) -> bool:
        return key in self._load()

    def record(self, key: str, payload: dict | None = None) -> None:
        state = self._load()
        state[key] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": payload or {},
        }
        self._save(state)
