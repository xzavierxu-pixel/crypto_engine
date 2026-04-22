from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.data.dataset_builder import build_training_frame
from src.data.derivatives.feature_store import load_derivatives_frame_from_settings
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet


def _load_input(path: Path):
    if path.suffix.lower() == ".csv":
        return load_ohlcv_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return load_ohlcv_parquet(path)
    if path.suffix.lower() in {".feather", ".ft"}:
        return load_ohlcv_feather(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BTC Polymarket training frame.")
    parser.add_argument("--input", required=True, help="Path to OHLCV CSV or parquet input.")
    parser.add_argument("--output", required=True, help="Path to parquet output.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--horizon", default="5m", help="Horizon name to build.")
    parser.add_argument("--funding-input", help="Optional funding raw input override.")
    parser.add_argument("--basis-input", help="Optional basis raw input override.")
    parser.add_argument("--oi-input", help="Optional OI raw input override.")
    parser.add_argument("--options-input", help="Optional options raw input override.")
    parser.add_argument(
        "--derivatives-path-mode",
        choices=["latest", "archive"],
        default=None,
        help="Override derivatives path mode. Defaults to settings.derivatives.path_mode.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    source = _load_input(Path(args.input))
    derivatives_frame = load_derivatives_frame_from_settings(
        settings,
        funding_path=args.funding_input,
        basis_path=args.basis_input,
        oi_path=args.oi_input,
        options_path=args.options_input,
        path_mode=args.derivatives_path_mode,
    )
    training = build_training_frame(
        source,
        settings,
        horizon_name=args.horizon,
        derivatives_frame=derivatives_frame,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training.frame.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
