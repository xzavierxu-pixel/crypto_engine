#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from price_estimator_common import (
    load_config,
    load_deploy_manifest,
    load_feature_columns,
    logit_price,
    market_time_bucket,
    p_side_bucket,
    resolve_path,
    time_to_lowest_bucket,
    write_json,
)
from src.data.second_level_features import load_sampled_second_level_features, sample_second_level_feature_store


def read_trades(trades_dir: Path) -> pd.DataFrame:
    files = sorted(trades_dir.glob("date=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No trade parquet files found in {trades_dir}")
    frames = [pd.read_parquet(path) for path in files]
    trades = pd.concat(frames, ignore_index=True)
    trades["trade_time"] = pd.to_datetime(trades["trade_time"], utc=True)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    return trades.dropna(subset=["price", "trade_time", "condition_id", "outcome"])


def load_p_up(config: dict) -> pd.DataFrame:
    deploy_dir = resolve_path(config["paths"]["deploy_artifact_dir"])
    frames = []
    for name in ["train_predictions.parquet", "validation_predictions.parquet"]:
        path = deploy_dir / name
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["timestamp", "decision_time", "p_up"]))
    if not frames:
        raise FileNotFoundError(f"No deploy prediction parquet files found in {deploy_dir}")
    pred = pd.concat(frames, ignore_index=True)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    pred["decision_time"] = pd.to_datetime(pred["decision_time"], utc=True)
    return pred.drop_duplicates(subset=["timestamp", "decision_time"])


def second_level_summary(frame: pd.DataFrame, requested: list[str]) -> dict:
    feature_columns = [c for c in frame.columns if c.startswith(("sl_", "fm_"))]
    requested_present = [c for c in requested if c in frame.columns]
    requested_missing = sorted(set(requested) - set(requested_present))
    missing_ratio = {
        c: float(frame[c].isna().mean())
        for c in requested_present
    }
    return {
        "feature_count": len(feature_columns),
        "requested_features": requested,
        "requested_present": requested_present,
        "requested_missing": requested_missing,
        "requested_missing_ratio": missing_ratio,
    }


def add_second_level_features(config: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    second_level = config.get("second_level", {})
    if not bool(second_level.get("enabled", False)):
        return df, {"enabled": False, "feature_count": 0}
    store_path = second_level.get("feature_store_path")
    if not store_path:
        raise ValueError("second_level.enabled=true requires second_level.feature_store_path")
    sampled = load_price_estimator_second_level_features(df[["timestamp"]], resolve_path(store_path))
    feature_columns = [c for c in sampled.columns if c.startswith(("sl_", "fm_"))]
    duplicate_columns = [c for c in feature_columns if c in df.columns]
    if duplicate_columns:
        sampled = sampled.drop(columns=duplicate_columns)
        feature_columns = [c for c in feature_columns if c not in duplicate_columns]
    out = pd.concat([df.reset_index(drop=True), sampled[feature_columns].reset_index(drop=True)], axis=1)
    requested = list(second_level.get("required_features", []))
    summary = second_level_summary(out, requested)
    summary.update(
        {
            "enabled": True,
            "feature_store_path": str(resolve_path(store_path)),
            "duplicate_columns_dropped": duplicate_columns,
        }
    )
    return out, summary


def load_price_estimator_second_level_features(decision_frame: pd.DataFrame, store_path: Path) -> pd.DataFrame:
    if store_path.is_dir():
        split_store_dirs = [
            child
            for child in (store_path / "second_features_kline", store_path / "second_features_agg")
            if child.exists() and child.is_dir()
        ]
        if split_store_dirs:
            sampled_stores = [load_daily_partition_sample(decision_frame, child) for child in split_store_dirs]
            combined = sampled_stores[0]
            for sampled in sampled_stores[1:]:
                join_columns = [c for c in sampled.columns if c != "timestamp"]
                combined = combined.merge(
                    sampled[["timestamp", *join_columns]],
                    on="timestamp",
                    how="left",
                    suffixes=("", "_duplicate"),
                )
                duplicate_columns = [c for c in combined.columns if c.endswith("_duplicate")]
                if duplicate_columns:
                    combined = combined.drop(columns=duplicate_columns)
            feature_columns = [c for c in combined.columns if c.startswith(("sl_", "fm_"))]
            if feature_columns:
                combined[feature_columns] = combined[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return combined.reset_index(drop=True)
    return load_sampled_second_level_features(decision_frame, store_path)


def load_daily_partition_sample(decision_frame: pd.DataFrame, store_dir: Path) -> pd.DataFrame:
    decisions = decision_frame[["timestamp"]].copy()
    decisions["timestamp"] = pd.to_datetime(decisions["timestamp"], utc=True)
    sampled_parts: list[pd.DataFrame] = []
    decision_days = decisions["timestamp"].dt.floor("D")
    for day, decision_slice in decisions.groupby(decision_days, sort=True):
        partition_paths = []
        for candidate_day in [day - pd.Timedelta(days=1), day]:
            path = store_dir / f"date={candidate_day.strftime('%Y-%m-%d')}" / "second_features.parquet"
            if path.exists():
                partition_paths.append(path)
        if not partition_paths:
            sampled = pd.DataFrame({"timestamp": decision_slice["timestamp"]}, index=decision_slice.index)
        else:
            store = pd.concat((pd.read_parquet(path) for path in partition_paths), ignore_index=True)
            sampled = sample_second_level_feature_store(decision_slice, store)
            sampled.index = decision_slice.index
        sampled_parts.append(sampled)
    if not sampled_parts:
        return pd.DataFrame({"timestamp": decisions["timestamp"]}, index=decision_frame.index).reset_index(drop=True)
    return pd.concat(sampled_parts, axis=0).sort_index().reset_index(drop=True)


def build_split(config: dict, split: str, frame_path: Path, trades: pd.DataFrame, p_up: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    feature_columns = load_feature_columns(config)
    base_columns = [
        "timestamp",
        "decision_time",
        "market_t0",
        "polymarket_slug",
        "condition_id",
        "target",
        "endDate",
    ]
    df = pd.read_parquet(frame_path, columns=base_columns + feature_columns)
    for col in ["timestamp", "decision_time", "market_t0", "endDate"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df, sl_summary = add_second_level_features(config, df)
    before_rows = len(df)
    df = df.merge(p_up, on=["timestamp", "decision_time"], how="left", validate="one_to_one")
    missing_p_up = int(df["p_up"].isna().sum())
    df["selected_side"] = np.where(df["target"].astype(int) == 1, "UP", "DOWN")
    df["selected_outcome"] = np.where(df["selected_side"].eq("UP"), "up", "down")
    df["p_side"] = np.where(df["selected_side"].eq("UP"), df["p_up"], 1.0 - df["p_up"])
    df["direction_confidence"] = (df["p_up"] - 0.5).abs()
    df["p_bin"] = p_side_bucket(df["p_side"])

    selected = df[["condition_id", "selected_outcome", "decision_time", "endDate"]].copy()
    joined = trades.merge(
        selected,
        left_on=["condition_id", "outcome"],
        right_on=["condition_id", "selected_outcome"],
        how="inner",
    )
    window_minutes = int(config["target"]["window_minutes"])
    start_mask = joined["trade_time"] > joined["decision_time"]
    end_time = joined["decision_time"] + pd.to_timedelta(window_minutes, unit="m")
    if bool(config["target"].get("market_close_cap", True)):
        end_time = pd.concat([end_time, joined["endDate"]], axis=1).min(axis=1)
    end_mask = joined["trade_time"] <= end_time
    joined = joined[start_mask & end_mask].copy()
    if joined.empty:
        raise ValueError(f"No target trades matched for split {split}")

    joined = joined.sort_values(["condition_id", "selected_outcome", "decision_time", "price", "trade_time"])
    lowest = joined.groupby(["condition_id", "selected_outcome", "decision_time"], as_index=False).first()
    lowest = lowest.rename(columns={"price": "lowest_trade_price_next4", "trade_time": "lowest_trade_time_next4"})
    lowest["time_to_lowest_trade_sec"] = (
        lowest["lowest_trade_time_next4"] - lowest["decision_time"]
    ).dt.total_seconds()
    target_cols = [
        "condition_id",
        "selected_outcome",
        "decision_time",
        "lowest_trade_price_next4",
        "lowest_trade_time_next4",
        "time_to_lowest_trade_sec",
    ]
    out = df.merge(lowest[target_cols], on=["condition_id", "selected_outcome", "decision_time"], how="left")
    missing_target = int(out["lowest_trade_price_next4"].isna().sum())
    out = out.dropna(subset=["lowest_trade_price_next4", "p_up"]).copy()
    out["target_raw"] = out["lowest_trade_price_next4"].astype(float)
    out["target_logit"] = logit_price(out["target_raw"], float(config["target"]["eps"]))
    out["p_side_bucket"] = p_side_bucket(out["p_side"])
    out["time_to_lowest_trade_sec_bucket"] = time_to_lowest_bucket(out["time_to_lowest_trade_sec"])
    out["market_time_bucket"] = market_time_bucket(out["decision_time"])

    summary = {
        "split": split,
        "input_rows": before_rows,
        "missing_p_up_rows": missing_p_up,
        "missing_target_rows": missing_target,
        "output_rows": len(out),
        "start": str(out["timestamp"].min()) if len(out) else None,
        "end": str(out["timestamp"].max()) if len(out) else None,
        "selected_up_rows": int((out["selected_side"] == "UP").sum()),
        "selected_down_rows": int((out["selected_side"] == "DOWN").sum()),
        "second_level": second_level_summary(out, list(config.get("second_level", {}).get("required_features", [])))
        if bool(config.get("second_level", {}).get("enabled", False))
        else sl_summary,
    }
    if bool(config.get("second_level", {}).get("enabled", False)):
        summary["second_level"].update(
            {
                "enabled": True,
                "feature_store_path": sl_summary.get("feature_store_path"),
                "pre_filter_feature_count": sl_summary.get("feature_count"),
                "duplicate_columns_dropped": sl_summary.get("duplicate_columns_dropped", []),
            }
        )
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/configs/catboost_quantile_baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_deploy_manifest(config)
    trades = read_trades(resolve_path(config["paths"]["trades_dir"]))
    p_up = load_p_up(config)
    outputs = {}
    summaries = {
        "experiment_id": config["experiment_id"],
        "deploy_experiment_id": manifest.get("experiment_id"),
        "selected_side_source": config["target"]["selected_side_source"],
        "target_window_minutes": config["target"]["window_minutes"],
    }
    for split, key, out_key in [
        ("train", "train_frame", "train_dataset"),
        ("validation", "validation_frame", "validation_dataset"),
    ]:
        data, summary = build_split(config, split, resolve_path(config["paths"][key]), trades, p_up)
        out_path = resolve_path(config["paths"][out_key])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out_path, index=False)
        outputs[split] = str(out_path)
        summaries[f"{split}_summary"] = summary
    summaries["outputs"] = outputs
    write_json(resolve_path(config["paths"]["reports_dir"]) / "target_build_summary.json", summaries)
    print(summaries)


if __name__ == "__main__":
    main()
