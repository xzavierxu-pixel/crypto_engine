from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from src.core.schemas import AuditEvent


class AuditService:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AuditEvent) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), default=str) + "\n")
