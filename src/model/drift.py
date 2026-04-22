from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd


def compute_ks_distance(reference: pd.Series, observed: pd.Series) -> float:
    if reference.empty or observed.empty:
        return 0.0
    ref = reference.astype(float).sort_values().reset_index(drop=True)
    obs = observed.astype(float).sort_values().reset_index(drop=True)
    values = sorted(set(ref.tolist()) | set(obs.tolist()))
    if not values:
        return 0.0
    max_distance = 0.0
    for value in values:
        ref_cdf = float((ref <= value).mean())
        obs_cdf = float((obs <= value).mean())
        max_distance = max(max_distance, abs(ref_cdf - obs_cdf))
    return max_distance


@dataclass
class Stage1DriftMonitor:
    reference: pd.Series
    threshold: float = 0.1
    window_size: int = 500
    min_history: int = 50
    alert_consecutive: int = 1

    def __post_init__(self) -> None:
        self._history: deque[float] = deque(maxlen=self.window_size)
        self._consecutive_alerts = 0

    def update(self, probability: float) -> dict[str, float | bool | int]:
        self._history.append(float(probability))
        observed = pd.Series(list(self._history), dtype="float64")
        ks_distance = compute_ks_distance(self.reference, observed)
        threshold_breached = bool(len(self._history) >= min(self.min_history, self.window_size) and ks_distance > self.threshold)
        if threshold_breached:
            self._consecutive_alerts += 1
        else:
            self._consecutive_alerts = 0
        return {
            "enabled": True,
            "window_size": len(self._history),
            "ks_distance": ks_distance,
            "threshold_breached": threshold_breached,
            "consecutive_alerts": self._consecutive_alerts,
            "alert": bool(self._consecutive_alerts >= self.alert_consecutive),
        }
