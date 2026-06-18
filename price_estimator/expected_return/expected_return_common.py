from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def session_label(ts: pd.Series) -> pd.Series:
    hour = pd.to_datetime(ts, utc=True).dt.hour
    return pd.cut(
        hour,
        bins=[-1, 7, 15, 23],
        labels=["asia", "europe", "us"],
        include_lowest=True,
    ).astype("string")


def threshold_frame(decision_time: pd.Series, manifest: dict[str, Any]) -> pd.DataFrame:
    ts = pd.to_datetime(decision_time, utc=True)
    policy = manifest.get("threshold_policy") or {}
    if str(policy.get("type", "fallback")) != "utc_day_session_coordinate":
        return pd.DataFrame(
            {
                "selected_t_up": float(manifest["t_up"]),
                "selected_t_down": float(manifest["t_down"]),
            },
            index=decision_time.index,
        )
    thresholds = policy.get("thresholds", {})
    fallback_up = float(policy.get("fallback_t_up", manifest["t_up"]))
    fallback_down = float(policy.get("fallback_t_down", manifest["t_down"]))
    sessions = session_label(ts)
    keys = ["d" + str(int(day)) + "_" + str(sess) for day, sess in zip(ts.dt.weekday, sessions)]
    t_up = np.asarray([float(thresholds.get(key, {}).get("t_up", fallback_up)) for key in keys], dtype=float)
    t_down = np.asarray([float(thresholds.get(key, {}).get("t_down", fallback_down)) for key in keys], dtype=float)
    return pd.DataFrame({"selected_t_up": t_up, "selected_t_down": t_down}, index=decision_time.index)


def choose_side(p_up: pd.Series, decision_time: pd.Series, manifest: dict[str, Any]) -> pd.DataFrame:
    thresholds = threshold_frame(decision_time, manifest)
    p = pd.to_numeric(p_up, errors="coerce")
    up = p >= thresholds["selected_t_up"]
    down = p <= thresholds["selected_t_down"]
    side = np.full(len(p), "DROP", dtype=object)
    side[up.to_numpy()] = "UP"
    side[down.to_numpy()] = "DOWN"
    out = thresholds.copy()
    out["selected_side"] = side
    out["accepted"] = out["selected_side"] != "DROP"
    out["selected_outcome"] = np.where(out["selected_side"].eq("UP"), "up", "down")
    out.loc[~out["accepted"], "selected_outcome"] = pd.NA
    return out


def p_side_bucket(p_side: pd.Series) -> pd.Series:
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 1.01]
    labels = ["0.50_0.55", "0.55_0.60", "0.60_0.65", "0.65_0.70", "0.70_1.00"]
    return pd.cut(p_side, bins=bins, labels=labels, include_lowest=True, right=False).astype("string").fillna("missing")


def price_grid(p_side: float, tick_size: float, min_bid: float) -> np.ndarray:
    hi = math.floor((float(p_side) + 1e-12) / tick_size) * tick_size
    lo = max(float(min_bid), float(tick_size))
    if hi < lo:
        return np.asarray([max(min(float(p_side), lo), 0.0)], dtype=float)
    count = int(round((hi - lo) / tick_size)) + 1
    return np.round(lo + np.arange(count, dtype=float) * tick_size, 10)


def empirical_cdf(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    if len(sorted_values) == 0:
        return np.zeros_like(points, dtype=float)
    return np.searchsorted(sorted_values, np.asarray(points, dtype=float), side="right") / float(len(sorted_values))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
