from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def hash_config(config: Any) -> str:
    if is_dataclass(config):
        payload = asdict(config)
    else:
        payload = config
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
