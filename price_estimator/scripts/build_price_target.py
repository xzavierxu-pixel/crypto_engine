#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    }
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
