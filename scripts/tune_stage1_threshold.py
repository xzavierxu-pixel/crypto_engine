from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.data.derivatives.feature_store import load_derivatives_frame_from_settings
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.model.train import train_two_stage_model


def _load_input(path: Path):
    if path.suffix.lower() == ".csv":
        return load_ohlcv_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return load_ohlcv_parquet(path)
    if path.suffix.lower() in {".feather", ".ft"}:
        return load_ohlcv_feather(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the two-stage model and export the Stage 1 threshold scan.")
    parser.add_argument("--input", required=True, help="Path to OHLCV CSV or parquet input.")
    parser.add_argument("--output", required=True, help="Path to JSON output.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--horizon", default="5m", help="Horizon name to train.")
    parser.add_argument("--validation-window-days", type=int, default=None, help="Validation window in days.")
    parser.add_argument("--purge-rows", type=int, default=1, help="Rows purged between train and validation splits.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    source = _load_input(Path(args.input))
    derivatives_frame = load_derivatives_frame_from_settings(settings)
    training = build_training_frame(
        source,
        settings,
        horizon_name=args.horizon,
        derivatives_frame=derivatives_frame,
    )
    artifacts = train_two_stage_model(
        training,
        settings,
        validation_window_days=args.validation_window_days or settings.dataset.validation_window_days,
        purge_rows=args.purge_rows,
    )
    payload = {
        "horizon": args.horizon,
        "stage1_threshold": artifacts.stage1_threshold,
        "buy_threshold": artifacts.buy_threshold,
        "base_rate": artifacts.base_rate,
        "threshold_scan": artifacts.stage1_threshold_scan,
        "stage1_probability_summary": artifacts.stage1_probability_summary,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
