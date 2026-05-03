from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.core.constants import DEFAULT_TIMESTAMP_COLUMN


DEFAULT_MRC_FEATURE_CANDIDATES = [
    "sec_close",
    "sec_volume",
    "sec_quote_volume",
    "sec_trade_count",
    "sec_taker_buy_base_volume",
    "sec_taker_sell_base_volume",
    "sec_vwap",
    "sl_return_1s",
    "sl_return_3s",
    "sl_return_5s",
    "sl_return_10s",
    "sl_return_30s",
    "sl_return_60s",
    "sl_return_300s",
    "sl_log_return_1s",
    "sl_log_return_5s",
    "sl_log_return_30s",
    "sl_log_return_60s",
    "sl_rv_5s",
    "sl_rv_30s",
    "sl_rv_60s",
    "sl_rv_300s",
    "sl_range_30s",
    "sl_range_60s",
    "sl_price_slope_5s",
    "sl_price_slope_30s",
    "sl_direction_flips_30s",
    "sl_momentum_persistence_30s",
    "sl_directional_efficiency_30s",
    "sl_choppiness_30s",
    "sl_taker_buy_volume_30s",
    "sl_taker_sell_volume_30s",
    "sl_taker_volume_imbalance_30s",
    "sl_trade_count_30s",
    "sl_total_volume_30s",
    "sl_relative_volume_60s",
    "sl_volume_zscore_300s",
    "sl_vwap_distance_bps",
    "sl_range_position_60s",
    "sl_drawdown_60s",
    "sl_runup_60s",
    "sl_agg_large_trade_count_1s",
    "sl_agg_large_trade_count_5s",
    "sl_agg_large_trade_count_30s",
    "sl_agg_large_trade_count_60s",
    "sl_agg_large_trade_count_300s",
    "sl_agg_large_trade_volume_share_30s",
    "sl_agg_large_buy_volume_share_30s",
    "sl_agg_large_sell_volume_share_30s",
    "sl_agg_large_trade_imbalance_30s",
    "sl_agg_signed_large_notional_30s",
    "sl_agg_mean_interarrival_ms_30s",
    "sl_agg_min_interarrival_ms_30s",
    "sl_agg_interarrival_cv_30s",
    "sl_agg_trade_cluster_score_30s",
    "sl_agg_buy_trade_cluster_score_30s",
    "sl_agg_sell_trade_cluster_score_30s",
    "sl_agg_side_switch_count_30s",
    "sl_agg_buy_run_length_30s",
    "sl_agg_sell_run_length_30s",
    "sl_agg_intrasecond_flow_concentration_30s",
    "sl_agg_end_second_buy_pressure_30s",
    "sl_agg_end_second_sell_pressure_30s",
    "sl_agg_top_trade_notional_share_60s",
    "sl_agg_trade_notional_hhi_60s",
]


@dataclass(frozen=True)
class SequenceFrame:
    timestamps: pd.Series
    values: np.ndarray
    feature_columns: list[str]


class MRCSequenceDataset(Dataset):
    def __init__(
        self,
        *,
        values: np.ndarray,
        positions: np.ndarray,
        labels: np.ndarray,
        sequence_length: int,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.values = values
        self.positions = positions.astype("int64")
        self.labels = labels.astype("float32")
        self.sequence_length = int(sequence_length)
        self.sample_weights = (
            sample_weights.astype("float32")
            if sample_weights is not None
            else np.ones(len(self.labels), dtype="float32")
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = int(self.positions[index]) + 1
        start = end - self.sequence_length
        return (
            torch.from_numpy(self.values[start:end]),
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(self.sample_weights[index], dtype=torch.float32),
        )


def list_split_store_feature_columns(split_store_root: str | Path) -> list[str]:
    root = Path(split_store_root)
    columns: list[str] = []
    for store_name in ("second_features_kline", "second_features_agg"):
        store = root / store_name
        parquet_files = sorted(store.glob("date=*/second_features.parquet"))
        if not parquet_files:
            continue
        schema = pq.read_schema(parquet_files[0])
        for column in schema.names:
            if column.startswith("sl_") or column.startswith("sec_"):
                columns.append(column)
    return columns


def resolve_mrc_feature_columns(
    split_store_root: str | Path,
    *,
    max_features: int = 64,
    requested_columns: Sequence[str] | None = None,
) -> list[str]:
    available = list_split_store_feature_columns(split_store_root)
    available_set = set(available)
    if requested_columns:
        selected = [column for column in requested_columns if column in available_set]
    else:
        selected = [column for column in DEFAULT_MRC_FEATURE_CANDIDATES if column in available_set]
        for column in available:
            if len(selected) >= max_features:
                break
            if column not in selected:
                selected.append(column)
    return selected[:max_features]


def _read_store_columns(store_dir: Path, columns: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for parquet_path in sorted(store_dir.glob("date=*/second_features.parquet")):
        label = parquet_path.parent.name.removeprefix("date=")
        day = pd.Timestamp(label, tz="UTC")
        if day > end.floor("D") or day < start.floor("D"):
            continue
        schema_columns = set(pq.read_schema(parquet_path).names)
        read_columns = [DEFAULT_TIMESTAMP_COLUMN, *[column for column in columns if column in schema_columns]]
        frame = pd.read_parquet(parquet_path, columns=read_columns)
        timestamps = pd.to_datetime(frame[DEFAULT_TIMESTAMP_COLUMN], utc=True)
        frame = frame.loc[(timestamps >= start) & (timestamps <= end)].reset_index(drop=True)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No split-store partitions found in {store_dir} for {start} to {end}.")
    return pd.concat(frames, ignore_index=True)


def load_split_sequence_frame(
    split_store_root: str | Path,
    *,
    feature_columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> SequenceFrame:
    root = Path(split_store_root)
    kline_available = set()
    agg_available = set()
    kline_files = sorted((root / "second_features_kline").glob("date=*/second_features.parquet"))
    agg_files = sorted((root / "second_features_agg").glob("date=*/second_features.parquet"))
    if kline_files:
        kline_available = set(pq.read_schema(kline_files[0]).names)
    if agg_files:
        agg_available = set(pq.read_schema(agg_files[0]).names)
    kline_columns = [column for column in feature_columns if column in kline_available]
    agg_columns = [column for column in feature_columns if column in agg_available]

    kline = _read_store_columns(root / "second_features_kline", kline_columns, start, end)
    pieces = [kline[[DEFAULT_TIMESTAMP_COLUMN, *kline_columns]]]
    if agg_columns:
        agg = _read_store_columns(root / "second_features_agg", agg_columns, start, end)
        if not kline[DEFAULT_TIMESTAMP_COLUMN].equals(agg[DEFAULT_TIMESTAMP_COLUMN]):
            agg = agg.set_index(DEFAULT_TIMESTAMP_COLUMN).reindex(kline[DEFAULT_TIMESTAMP_COLUMN]).reset_index()
        pieces.append(agg[agg_columns])
    frame = pd.concat(pieces, axis=1)
    frame[DEFAULT_TIMESTAMP_COLUMN] = pd.to_datetime(frame[DEFAULT_TIMESTAMP_COLUMN], utc=True)
    frame = frame.sort_values(DEFAULT_TIMESTAMP_COLUMN).reset_index(drop=True)
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return SequenceFrame(
        timestamps=frame[DEFAULT_TIMESTAMP_COLUMN],
        values=frame[feature_columns].astype("float32").to_numpy(copy=True),
        feature_columns=feature_columns,
    )


def build_sequence_sample_positions(
    *,
    feature_timestamps: pd.Series,
    sample_timestamps: pd.Series,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_ns = pd.to_datetime(feature_timestamps, utc=True).astype("int64").to_numpy()
    sample_ns = pd.to_datetime(sample_timestamps, utc=True).astype("int64").to_numpy()
    positions = np.searchsorted(feature_ns, sample_ns)
    valid = (positions < len(feature_ns)) & (feature_ns[np.clip(positions, 0, len(feature_ns) - 1)] == sample_ns)
    valid &= positions >= (sequence_length - 1)
    return positions[valid], valid
