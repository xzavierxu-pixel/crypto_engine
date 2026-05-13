from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution_engine.config import load_execution_config
from execution_engine.run_once import current_5m_window_start, run_once
from src.core.schemas import Decision


def next_trigger_time(now: datetime, *, delay_seconds: int) -> datetime:
    window_start = current_5m_window_start(now)
    trigger = window_start + timedelta(seconds=delay_seconds)
    if now.astimezone(UTC) < trigger:
        return trigger
    return window_start + timedelta(minutes=5, seconds=delay_seconds)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _opposite_side(side: str | None) -> str | None:
    if side == "YES":
        return "NO"
    if side == "NO":
        return "YES"
    return None


def _load_cached_minute_frame(cache_path: str | None) -> pd.DataFrame:
    if cache_path is None:
        raise RuntimeError("last-window-momentum policy requires binance.cache_path to be configured.")
    cache_dir = Path(cache_path)
    if cache_dir.suffix:
        cache_dir = cache_dir.with_suffix("")
    minute_path = cache_dir / "minute.parquet"
    if not minute_path.exists():
        raise RuntimeError(f"Minute cache not found for last-window-momentum policy: {minute_path}")
    frame = pd.read_parquet(minute_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.drop_duplicates("timestamp").set_index("timestamp").sort_index()


def _momentum_side(
    minute_frame: pd.DataFrame,
    signal_t0: str,
    *,
    start_offset_minutes: int,
    end_offset_minutes: int,
) -> str:
    t0 = pd.Timestamp(signal_t0).tz_convert("UTC")
    start_time = t0 - pd.Timedelta(minutes=start_offset_minutes)
    end_time = t0 - pd.Timedelta(minutes=end_offset_minutes)
    if start_time not in minute_frame.index or end_time not in minute_frame.index:
        raise RuntimeError(
            "Minute cache missing rows required for momentum paper policy: "
            f"{start_time.isoformat()} / {end_time.isoformat()}"
        )
    start_close = float(minute_frame.loc[start_time, "close"])
    end_close = float(minute_frame.loc[end_time, "close"])
    return "YES" if end_close >= start_close else "NO"


def _last_window_momentum_side(minute_frame: pd.DataFrame, signal_t0: str) -> str:
    t0 = pd.Timestamp(signal_t0).tz_convert("UTC")
    previous_window_open_time = t0 - pd.Timedelta(minutes=5)
    previous_window_close_time = t0 - pd.Timedelta(minutes=1)
    if previous_window_open_time not in minute_frame.index or previous_window_close_time not in minute_frame.index:
        raise RuntimeError(
            "Minute cache missing rows required for last-window-momentum policy: "
            f"{previous_window_open_time.isoformat()} / {previous_window_close_time.isoformat()}"
        )
    previous_open = float(minute_frame.loc[previous_window_open_time, "open"])
    previous_close = float(minute_frame.loc[previous_window_close_time, "close"])
    return "YES" if previous_close >= previous_open else "NO"


def apply_experiment_policy(summary: dict[str, Any], *, policy: str, config_path: str) -> dict[str, Any]:
    if policy == "model":
        summary["experiment_policy"] = {"name": "model", "overrode_decision": False}
        return summary

    original_decision = summary.get("decision") or {}
    if policy == "inverse-model":
        side = _opposite_side(original_decision.get("side"))
        if side is None:
            decision = Decision(False, None, None, "inverse_model_abstain", 0.0)
        else:
            decision = Decision(True, side, original_decision.get("edge"), "inverse_model_signal_passed", 5.0)
    elif policy == "last-window-momentum":
        config = load_execution_config(config_path)
        minute_frame = _load_cached_minute_frame(config.binance.cache_path)
        side = _last_window_momentum_side(minute_frame, summary["signal"]["t0"])
        decision = Decision(True, side, None, "last_window_momentum_signal_passed", 5.0)
    elif policy == "prev3-momentum":
        config = load_execution_config(config_path)
        minute_frame = _load_cached_minute_frame(config.binance.cache_path)
        side = _momentum_side(
            minute_frame,
            summary["signal"]["t0"],
            start_offset_minutes=3,
            end_offset_minutes=1,
        )
        decision = Decision(True, side, None, "prev3_momentum_signal_passed", 5.0)
    else:
        raise ValueError(f"Unsupported paper experiment policy: {policy}")

    summary["experiment_policy"] = {
        "name": policy,
        "overrode_decision": True,
        "original_decision": original_decision,
    }
    summary["decision"] = asdict(decision)
    if summary.get("summary_path"):
        Path(summary["summary_path"]).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    return summary


def run_paper_experiment(
    *,
    config_path: str,
    cycles: int,
    delay_seconds: int | None = None,
    output_jsonl: str,
    policy: str = "model",
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    config = load_execution_config(config_path)
    resolved_delay = config.schedule.trigger_delay_seconds if delay_seconds is None else delay_seconds
    output_path = Path(output_jsonl)
    records: list[dict[str, Any]] = []

    for index in range(cycles):
        trigger = next_trigger_time(datetime.now(UTC), delay_seconds=resolved_delay)
        wait_seconds = max(0.0, (trigger - datetime.now(UTC)).total_seconds())
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        target_window_start = current_5m_window_start(datetime.now(UTC))
        started_at = datetime.now(UTC)
        record: dict[str, Any] = {
            "cycle_index": index,
            "started_at": started_at.isoformat(),
            "target_window_start": target_window_start.isoformat(),
            "mode_override": "paper",
            "dry_run": dry_run,
        }
        if not dry_run:
            try:
                summary = run_once(
                    config_path,
                    mode_override="paper",
                    target_window_start=target_window_start,
                )
                summary = apply_experiment_policy(summary, policy=policy, config_path=config_path)
                record.update(
                    {
                        "ok": True,
                        "experiment_policy": summary.get("experiment_policy"),
                        "summary_path": summary.get("summary_path"),
                        "decision": summary.get("decision"),
                        "signal": summary.get("signal"),
                        "market": summary.get("market"),
                        "skipped": summary.get("skipped"),
                        "submitted": summary.get("submitted"),
                    }
                )
            except Exception as exc:
                record.update({"ok": False, "error": repr(exc)})
        append_jsonl(output_path, record)
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled paper execution_engine experiment.")
    parser.add_argument("--config", default="execution_engine/config.yaml")
    parser.add_argument("--cycles", type=int, default=13)
    parser.add_argument("--delay-seconds", type=int, default=None)
    parser.add_argument(
        "--output-jsonl",
        default="artifacts/logs/execution_engine/paper_experiment.jsonl",
    )
    parser.add_argument(
        "--policy",
        choices=["model", "inverse-model", "last-window-momentum", "prev3-momentum"],
        default="model",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = run_paper_experiment(
        config_path=args.config,
        cycles=args.cycles,
        delay_seconds=args.delay_seconds,
        output_jsonl=args.output_jsonl,
        policy=args.policy,
        dry_run=args.dry_run,
    )
    print(json.dumps({"records": records}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
