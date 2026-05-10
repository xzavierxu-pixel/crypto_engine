from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution_engine.artifacts import load_baseline_artifact
from execution_engine.config import load_execution_config
from execution_engine.feature_runtime import RuntimeInferenceEngine
from execution_engine.realtime_data import BinanceRealtimeClient
from src.core.config import load_settings


def prewarm(config_path: str, cache_output: str | None = None) -> dict:
    config = load_execution_config(config_path)
    baseline = load_baseline_artifact(config.baseline)
    settings = load_settings(config.baseline.settings_path)
    client = BinanceRealtimeClient(config.binance)
    minute_frame, second_frame, agg_trades_frame = client.fetch_runtime_frames()
    inference = RuntimeInferenceEngine(settings, baseline)
    feature_frame, second_level_frame = inference.build_feature_frame(minute_frame, second_frame, agg_trades_frame)
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "minute_rows": len(minute_frame),
        "second_rows": len(second_frame),
        "agg_trade_rows": len(agg_trades_frame),
        "feature_rows": len(feature_frame),
        "second_level_rows": len(second_level_frame),
        "minute_latest": None if minute_frame.empty else minute_frame["timestamp"].iloc[-1].isoformat(),
        "second_latest": None if second_frame.empty else second_frame["timestamp"].iloc[-1].isoformat(),
        "agg_trade_latest": None if agg_trades_frame.empty else agg_trades_frame["timestamp"].iloc[-1].isoformat(),
        "feature_latest": None if feature_frame.empty else feature_frame["timestamp"].iloc[-1].isoformat(),
        "baseline_artifact_dir": str(baseline.artifact_dir),
        "feature_count": len(baseline.feature_columns),
        "cache_output": cache_output,
    }
    if cache_output:
        output_dir = Path(cache_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        minute_frame.to_parquet(output_dir / "minute.parquet", index=False)
        second_frame.to_parquet(output_dir / "second.parquet", index=False)
        agg_trades_frame.to_parquet(output_dir / "agg_trades.parquet", index=False)
        second_level_frame.to_parquet(output_dir / "second_level_sampled.parquet", index=False)
        feature_frame.to_parquet(output_dir / "features.parquet", index=False)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prewarm Binance runtime data cache.")
    parser.add_argument("--config", default="execution_engine/config.yaml")
    parser.add_argument("--cache-output", default=None)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    summary = prewarm(args.config, cache_output=args.cache_output)
    if args.print_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
