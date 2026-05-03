from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'src').is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.data.dataset_builder import build_feature_schema_qa, build_training_frame
from src.data.derivatives.feature_store import load_derivatives_frame_from_settings
from src.data.loaders import load_ohlcv_csv, load_ohlcv_feather, load_ohlcv_parquet
from src.data.second_level_features import load_sampled_second_level_features


def _default_dataset_output(data_root: str, market: str, horizon: str) -> Path:
    market_label = market.replace("/", "")
    return Path(data_root) / "datasets" / f"market={market_label}" / f"horizon={horizon}" / "training_frame.parquet"


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
    parser.add_argument("--output", help="Path to parquet output. Defaults to data-root/datasets/market=.../horizon=.../training_frame.parquet.")
    parser.add_argument("--data-root", help="Unified data root for new outputs. Defaults to settings.second_level.data_root.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--horizon", default="5m", help="Horizon name to build.")
    parser.add_argument("--funding-input", help="Optional funding raw input override.")
    parser.add_argument("--basis-input", help="Optional basis raw input override.")
    parser.add_argument("--oi-input", help="Optional OI raw input override.")
    parser.add_argument("--options-input", help="Optional options raw input override.")
    parser.add_argument(
        "--second-level-feature-store",
        help="Materialized second-level feature store parquet or partitioned directory. Defaults to settings.second_level.feature_store_path.",
    )
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
    second_level_features_frame = None
    second_level_store_path = args.second_level_feature_store or settings.second_level.feature_store_path
    if settings.second_level.enabled:
        if second_level_store_path is None:
            raise ValueError("settings.second_level.enabled requires a second-level feature store path.")
        if Path(second_level_store_path).exists():
            second_level_features_frame = load_sampled_second_level_features(source, second_level_store_path)
        elif args.second_level_feature_store:
            raise FileNotFoundError(f"Second-level feature store does not exist: {second_level_store_path}")
    training = build_training_frame(
        source,
        settings,
        horizon_name=args.horizon,
        derivatives_frame=derivatives_frame,
        second_level_features_frame=second_level_features_frame,
    )

    data_root = args.data_root or settings.second_level.data_root
    output_path = Path(args.output) if args.output else _default_dataset_output(data_root, settings.second_level.market, args.horizon)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training.frame.to_parquet(output_path, index=False)
    qa_payload = build_feature_schema_qa(training.frame, training.feature_columns)
    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(args.input).resolve()),
        "output": str(output_path.resolve()),
        "config": str(Path(args.config).resolve()),
        "horizon": args.horizon,
        "row_count": int(len(training.frame)),
        "column_count": int(len(training.frame.columns)),
        "feature_count": int(len(training.feature_columns)),
        "target_column": training.target_column,
        "sample_weight_column": training.sample_weight_column,
        "timestamp_start": str(training.frame["timestamp"].min()) if not training.frame.empty else None,
        "timestamp_end": str(training.frame["timestamp"].max()) if not training.frame.empty else None,
        "feature_schema_qa": qa_payload,
    }
    output_path.with_name("training_frame_qa.json").write_text(json.dumps(qa_payload, indent=2), encoding="utf-8")
    output_path.with_name("training_frame_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
