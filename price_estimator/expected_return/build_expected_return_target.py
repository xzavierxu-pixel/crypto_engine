#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
ROOT = PRICE_ESTIMATOR_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))

from price_estimator_common import load_config, load_deploy_manifest, resolve_path  # noqa: E402

from expected_return_common import (  # noqa: E402
    choose_side,
    git_commit,
    p_side_bucket,
    session_label,
    write_json,
)


def load_deploy_model(config: dict[str, Any], manifest: dict[str, Any]) -> Any:
    artifact_dir = resolve_path(config["paths"]["deploy_artifact_dir"])
    model_plugin = str(manifest["model_plugin"])
    model_path = artifact_dir / f"{model_plugin}.binary.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Deploy model not found: {model_path}")
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if hasattr(payload, "predict_proba"):
        return payload
    raise TypeError(f"Unsupported deploy model payload in {model_path}: {type(payload)!r}")


def refresh_deploy_p_up(df: pd.DataFrame, model: Any, manifest: dict[str, Any], source_path: Path) -> pd.DataFrame:
    feature_columns = [str(column) for column in manifest["feature_columns"]]
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        preview = ", ".join(missing[:20])
        raise ValueError(
            f"{source_path} is missing {len(missing)} deploy feature columns; first missing: {preview}"
        )
    refreshed = df.copy()
    proba = model.predict_proba(refreshed[feature_columns])
    if isinstance(proba, pd.Series):
        p_up = proba.to_numpy(dtype=float)
    else:
        p_up = np.asarray(proba, dtype=float)
        if p_up.ndim == 2:
            if p_up.shape[1] < 2:
                raise ValueError(f"Deploy model predict_proba returned shape {p_up.shape}, expected class probabilities")
            p_up = p_up[:, 1]
    refreshed["p_up"] = np.clip(np.asarray(p_up, dtype=float).reshape(-1), 0.0, 1.0)
    return refreshed


def read_trades(trades_dir: Path) -> pd.DataFrame:
    files = sorted(trades_dir.glob("date=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No trade parquet files found in {trades_dir}")
    frames = [pd.read_parquet(path) for path in files]
    trades = pd.concat(frames, ignore_index=True)
    trades["trade_time"] = pd.to_datetime(trades["trade_time"], utc=True, errors="coerce")
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["outcome"] = trades["outcome"].astype("string").str.lower()
    return trades.dropna(subset=["price", "trade_time", "condition_id", "outcome"])


def build_split(
    config: dict[str, Any],
    manifest: dict[str, Any],
    model: Any,
    split: str,
    source_path: Path,
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(source_path)
    for col in ["timestamp", "decision_time", "market_t0", "endDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    before_rows = len(df)
    if "target" not in df.columns:
        raise ValueError(f"{source_path} is missing target")
    source_p_up = pd.to_numeric(df["p_up"], errors="coerce") if "p_up" in df.columns else None
    df = refresh_deploy_p_up(df, model, manifest, source_path)
    p_up_refresh_report = {
        "source_had_p_up": bool(source_p_up is not None),
        "max_abs_p_up_delta": float((df["p_up"] - source_p_up).abs().max()) if source_p_up is not None else None,
        "mean_abs_p_up_delta": float((df["p_up"] - source_p_up).abs().mean()) if source_p_up is not None else None,
    }

    side = choose_side(df["p_up"], df["decision_time"], manifest)
    for col in ["selected_side", "selected_outcome", "accepted", "selected_t_up", "selected_t_down"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = pd.concat([df.reset_index(drop=True), side.reset_index(drop=True)], axis=1)
    df = df.loc[df["accepted"]].copy()
    df["selected_side"] = df["selected_side"].astype("string")
    df["selected_outcome"] = df["selected_outcome"].astype("string")
    df["correct"] = (
        ((df["target"].astype(int) == 1) & df["selected_side"].eq("UP"))
        | ((df["target"].astype(int) == 0) & df["selected_side"].eq("DOWN"))
    )
    df["p_side"] = np.where(df["selected_side"].eq("UP"), df["p_up"], 1.0 - df["p_up"])
    df["direction_confidence"] = (pd.to_numeric(df["p_up"], errors="coerce") - 0.5).abs()
    df["p_bin"] = p_side_bucket(pd.Series(df["p_side"], index=df.index))
    df["p_side_bucket"] = p_side_bucket(pd.Series(df["p_side"], index=df.index))
    df["market_time_bucket"] = session_label(df["decision_time"]).reset_index(drop=True).to_numpy()

    selected = df[["condition_id", "selected_outcome", "decision_time", "endDate"]].copy()
    joined = trades.merge(
        selected,
        left_on=["condition_id", "outcome"],
        right_on=["condition_id", "selected_outcome"],
        how="inner",
    )
    start_mask = joined["trade_time"] > joined["decision_time"]
    end_mask = joined["trade_time"] <= joined["endDate"]
    joined = joined.loc[start_mask & end_mask].copy()
    if joined.empty:
        raise ValueError(f"No selected-side trades matched for split {split}")

    joined = joined.sort_values(["condition_id", "selected_outcome", "decision_time", "price", "trade_time"])
    lowest = joined.groupby(["condition_id", "selected_outcome", "decision_time"], as_index=False).first()
    lowest = lowest.rename(columns={"price": "chosen_low", "trade_time": "chosen_low_trade_time"})
    lowest["time_to_chosen_low_sec"] = (lowest["chosen_low_trade_time"] - lowest["decision_time"]).dt.total_seconds()
    target_cols = [
        "condition_id",
        "selected_outcome",
        "decision_time",
        "chosen_low",
        "chosen_low_trade_time",
        "time_to_chosen_low_sec",
    ]
    out = df.merge(lowest[target_cols], on=["condition_id", "selected_outcome", "decision_time"], how="left")
    missing_low = int(out["chosen_low"].isna().sum())
    out = out.dropna(subset=["chosen_low", "p_up", "p_side"]).copy()
    out["target_raw"] = out["chosen_low"].astype(float)

    correct = out["correct"].astype(bool)
    wrong = ~correct
    wrong_low = pd.to_numeric(out.loc[wrong, "chosen_low"], errors="coerce")
    summary = {
        "split": split,
        "input_rows": int(before_rows),
        "accepted_rows_before_low_join": int(len(df)),
        "missing_low_rows": missing_low,
        "output_rows": int(len(out)),
        "accepted_coverage_vs_source": float(len(df) / before_rows) if before_rows else float("nan"),
        "correct_count": int(correct.sum()),
        "wrong_count": int(wrong.sum()),
        "accepted_accuracy": float(correct.mean()) if len(correct) else float("nan"),
        "selected_up_rows": int(out["selected_side"].eq("UP").sum()),
        "selected_down_rows": int(out["selected_side"].eq("DOWN").sum()),
        "start": str(out["timestamp"].min()) if len(out) and "timestamp" in out.columns else None,
        "end": str(out["timestamp"].max()) if len(out) and "timestamp" in out.columns else None,
        "wrong_low_mean": float(wrong_low.mean()) if len(wrong_low) else float("nan"),
        "wrong_low_p95": float(wrong_low.quantile(0.95)) if len(wrong_low) else float("nan"),
        "wrong_low_lte_001_share": float((wrong_low <= 0.01).mean()) if len(wrong_low) else float("nan"),
        "threshold_policy_type": str((manifest.get("threshold_policy") or {}).get("type", "fallback")),
        "p_up_refresh": p_up_refresh_report,
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/expected_return/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_deploy_manifest(config)
    model = load_deploy_model(config, manifest)
    trades = read_trades(resolve_path(config["paths"]["trades_dir"]))

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")

    outputs: dict[str, str] = {}
    summaries: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_build": git_commit(),
        "config_path": args.config,
        "deploy_experiment_id": manifest.get("experiment_id"),
        "threshold_source": manifest.get("threshold_source"),
        "threshold_policy": manifest.get("threshold_policy"),
        "order_window": config["target"]["order_window"],
        "fee": config["target"]["fee"],
    }
    for split, source_key, out_key in [
        ("train", "train_dataset_source", "train_dataset"),
        ("validation", "validation_dataset_source", "validation_dataset"),
    ]:
        data, summary = build_split(config, manifest, model, split, resolve_path(config["paths"][source_key]), trades)
        out_path = resolve_path(config["paths"][out_key])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out_path, index=False)
        outputs[split] = str(out_path)
        summaries[f"{split}_summary"] = summary
    summaries["outputs"] = outputs
    write_json(reports_dir / "target_build_summary.json", summaries)
    print(summaries)


if __name__ == "__main__":
    main()
