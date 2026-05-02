from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.data.second_level_features import (
    build_second_level_source_tables,
    build_second_level_feature_store,
    load_second_level_frame,
    resolve_second_level_feature_profile,
    write_partitioned_second_level_feature_store,
    write_second_level_source_tables,
    write_second_level_feature_store,
)


def _default_second_level_output(data_root: str, version: str, market: str) -> Path:
    return Path(data_root) / "second_level" / f"version={version}" / f"market={market}" / "second_features.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the materialized second-level feature store.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--kline-1s-input", required=True, help="Canonical Binance Spot 1s kline input.")
    parser.add_argument("--agg-trades-input", help="Optional Binance Spot aggTrades enrichment input.")
    parser.add_argument("--book-ticker-input", help="Optional Binance Spot bookTicker liquidity input.")
    parser.add_argument("--depth-input", help="Optional multi-level depth snapshot input.")
    parser.add_argument("--perp-kline-1s-input", help="Optional perp 1s kline input for cross-market features.")
    parser.add_argument("--perp-book-ticker-input", help="Optional perp bookTicker input for cross-market quote-state features.")
    parser.add_argument("--eth-kline-1s-input", help="Optional ETH 1s kline input for crypto beta residual features.")
    parser.add_argument("--data-root", help="Unified data root for new outputs. Defaults to settings.second_level.data_root.")
    parser.add_argument("--output", help="Output parquet path. Defaults to data-root/second_level/version=.../market=.../second_features.parquet.")
    parser.add_argument("--write-source-tables", action="store_true", help="Write source-normalized 1s tables next to the feature store.")
    parser.add_argument(
        "--partition-frequency",
        choices=["none", "daily", "monthly"],
        default="none",
        help="Write partitioned feature-store chunks instead of one wide parquet.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=int,
        default=300,
        help="Lookback seconds prepended to each partition before trimming the output chunk.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    data_root = args.data_root or settings.second_level.data_root
    output_path = args.output or _default_second_level_output(
        data_root,
        settings.second_level.feature_store_version,
        settings.second_level.market,
    )
    if output_path is None:
        raise ValueError("Second-level feature store output path is required.")
    feature_profile_payload = settings.second_level.get_profile_payload()
    feature_profile = resolve_second_level_feature_profile(feature_profile_payload)

    kline_frame = load_second_level_frame(args.kline_1s_input)
    use_partitioned_inputs = args.partition_frequency != "none"
    agg_trades_frame = args.agg_trades_input if use_partitioned_inputs and args.agg_trades_input else load_second_level_frame(args.agg_trades_input) if args.agg_trades_input else None
    book_frame = args.book_ticker_input if use_partitioned_inputs and args.book_ticker_input else load_second_level_frame(args.book_ticker_input) if args.book_ticker_input else None
    depth_frame = args.depth_input if use_partitioned_inputs and args.depth_input else load_second_level_frame(args.depth_input) if args.depth_input else None
    cross_market_frame = args.perp_kline_1s_input if use_partitioned_inputs and args.perp_kline_1s_input else load_second_level_frame(args.perp_kline_1s_input) if args.perp_kline_1s_input else None
    cross_market_book_frame = args.perp_book_ticker_input if use_partitioned_inputs and args.perp_book_ticker_input else load_second_level_frame(args.perp_book_ticker_input) if args.perp_book_ticker_input else None
    eth_kline_frame = args.eth_kline_1s_input if use_partitioned_inputs and args.eth_kline_1s_input else load_second_level_frame(args.eth_kline_1s_input) if args.eth_kline_1s_input else None
    source_manifest = {
        "source_paths": {
            "kline_1s": str(Path(args.kline_1s_input).resolve()),
            "agg_trades": str(Path(args.agg_trades_input).resolve()) if args.agg_trades_input else None,
            "book_ticker": str(Path(args.book_ticker_input).resolve()) if args.book_ticker_input else None,
            "depth": str(Path(args.depth_input).resolve()) if args.depth_input else None,
            "perp_kline_1s": str(Path(args.perp_kline_1s_input).resolve()) if args.perp_kline_1s_input else None,
            "perp_book_ticker": str(Path(args.perp_book_ticker_input).resolve()) if args.perp_book_ticker_input else None,
            "eth_kline_1s": str(Path(args.eth_kline_1s_input).resolve()) if args.eth_kline_1s_input else None,
        },
        "source_precedence": {
            "canonical_price_volume_flow": "1s_klines",
            "event_structure": "aggTrades",
            "liquidity_state": "bookTicker",
            "depth_state": "depth snapshots",
            "cross_market": "perp 1s klines",
            "crypto_beta": "ETH 1s klines",
        },
        "data_root": str(Path(data_root).resolve()),
        "feature_profile": settings.second_level.feature_profile,
        "feature_packs": list(feature_profile.packs),
    }
    source_table_outputs = {}
    if args.write_source_tables:
        if use_partitioned_inputs:
            source_manifest["source_tables_note"] = (
                "Source tables were not materialized because partitioned raw inputs are streamed by partition."
            )
        else:
            source_tables = build_second_level_source_tables(
                kline_frame=kline_frame,
                agg_trades_frame=agg_trades_frame,
                book_frame=book_frame,
            )
            source_table_outputs = write_second_level_source_tables(source_tables, Path(output_path).with_name("source_tables"))
    if args.partition_frequency == "none":
        store = build_second_level_feature_store(
            kline_frame=kline_frame,
            agg_trades_frame=agg_trades_frame,
            book_frame=book_frame,
            depth_frame=depth_frame,
            cross_market_frame=cross_market_frame,
            cross_market_book_frame=cross_market_book_frame,
            eth_kline_frame=eth_kline_frame,
            market=settings.second_level.market,
            exchange=settings.second_level.exchange,
            source_manifest_id="local_cli",
            large_trade_quantile=settings.second_level.large_trade_quantile,
            large_trade_window_seconds=settings.second_level.large_trade_window_seconds,
            feature_profile=feature_profile,
        )
        manifest = write_second_level_feature_store(store, output_path, manifest=source_manifest)
    else:
        partition_output = Path(output_path)
        if partition_output.suffix:
            partition_output = partition_output.with_suffix("")
        manifest = write_partitioned_second_level_feature_store(
            kline_frame=kline_frame,
            output_dir=partition_output,
            partition_frequency=args.partition_frequency,
            warmup_seconds=args.warmup_seconds,
            agg_trades_frame=agg_trades_frame,
            book_frame=book_frame,
            depth_frame=depth_frame,
            cross_market_frame=cross_market_frame,
            cross_market_book_frame=cross_market_book_frame,
            eth_kline_frame=eth_kline_frame,
            market=settings.second_level.market,
            exchange=settings.second_level.exchange,
            source_manifest_id="local_cli",
            large_trade_quantile=settings.second_level.large_trade_quantile,
            large_trade_window_seconds=settings.second_level.large_trade_window_seconds,
            feature_profile=feature_profile,
            manifest=source_manifest,
        )
        output_path = str(partition_output)
    print(json.dumps({"output": str(Path(output_path).resolve()), "source_tables": source_table_outputs, **manifest}, indent=2))


if __name__ == "__main__":
    main()
