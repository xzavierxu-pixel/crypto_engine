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


def apply_trades_coverage_start(
    df: pd.DataFrame,
    coverage_start: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoff = pd.Timestamp(coverage_start)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    decision_time = pd.to_datetime(df["decision_time"], utc=True, errors="coerce")
    eligible = decision_time >= cutoff
    filtered = df.loc[eligible].copy()
    return filtered, {
        "trades_coverage_start": cutoff.isoformat(),
        "source_rows_before_coverage_filter": int(len(df)),
        "rows_excluded_before_trades_coverage": int((~eligible).sum()),
        "rows_after_trades_coverage_filter": int(eligible.sum()),
    }


def classify_low_join(
    rows: pd.DataFrame,
    trades: pd.DataFrame,
    matched_low: pd.Series,
) -> pd.Series:
    """Explain missing selected-side lows without using threshold acceptance."""
    condition_keys = pd.Index(trades["condition_id"].dropna().unique())
    outcome_keys = pd.MultiIndex.from_frame(trades[["condition_id", "outcome"]].drop_duplicates())
    row_outcome_keys = pd.MultiIndex.from_frame(
        rows[["condition_id", "predicted_outcome"]].rename(columns={"predicted_outcome": "outcome"})
    )
    condition_present = rows["condition_id"].isin(condition_keys).to_numpy()
    outcome_present = row_outcome_keys.isin(outcome_keys)

    last_trade = (
        trades.groupby(["condition_id", "outcome"], as_index=False)["trade_time"]
        .max()
        .rename(columns={"outcome": "predicted_outcome", "trade_time": "last_selected_trade_time"})
    )
    timing = rows[["condition_id", "predicted_outcome", "decision_time", "endDate"]].merge(
        last_trade,
        on=["condition_id", "predicted_outcome"],
        how="left",
    )
    has_after_decision = timing["last_selected_trade_time"] > timing["decision_time"]

    reason = np.full(len(rows), "matched", dtype=object)
    missing = matched_low.isna().to_numpy()
    reason[missing & ~condition_present] = "no_condition_trades"
    reason[missing & condition_present & ~outcome_present] = "no_predicted_outcome_trades"
    reason[missing & outcome_present & ~has_after_decision.to_numpy()] = "no_trades_after_decision"
    reason[missing & outcome_present & has_after_decision.to_numpy()] = "no_trades_before_settlement"
    return pd.Series(reason, index=rows.index, dtype="string")


def low_join_diagnostics(df: pd.DataFrame, split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    diagnostic = df.copy()
    diagnostic["date"] = diagnostic["decision_time"].dt.strftime("%Y-%m-%d")
    diagnostic["week"] = diagnostic["decision_time"].dt.tz_localize(None).dt.to_period("W").astype(str)
    dimensions = ["date", "week", "predicted_side", "threshold_accepted", "chosen_low_reason"]
    tables: dict[str, Any] = {}
    csv_frames: list[pd.DataFrame] = []
    for dimension in dimensions:
        grouped = (
            diagnostic.groupby(dimension, dropna=False)
            .agg(row_count=("condition_id", "size"), missing_low_rows=("chosen_low", lambda value: int(value.isna().sum())))
            .reset_index()
        )
        grouped["missing_low_rate"] = grouped["missing_low_rows"] / grouped["row_count"]
        grouped.insert(0, "dimension", dimension)
        grouped = grouped.rename(columns={dimension: "value"})
        tables[dimension] = grouped.to_dict(orient="records")
        csv_frames.append(grouped)
    reason_counts = diagnostic["chosen_low_reason"].value_counts(dropna=False)
    summary = {
        "split": split,
        "row_count": int(len(diagnostic)),
        "missing_low_rows": int(diagnostic["chosen_low"].isna().sum()),
        "missing_low_rate": float(diagnostic["chosen_low"].isna().mean()) if len(diagnostic) else float("nan"),
        "reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
        "by_dimension": tables,
        "threshold_not_causal_note": (
            "Rows are assigned predicted_side before thresholding and diagnostics include both "
            "threshold_accepted states; threshold only controls final ordering."
        ),
    }
    return summary, pd.concat(csv_frames, ignore_index=True)


def build_split(
    config: dict[str, Any],
    manifest: dict[str, Any],
    model: Any,
    split: str,
    source_path: Path,
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    df = pd.read_parquet(source_path)
    for col in ["timestamp", "decision_time", "market_t0", "endDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df, coverage_filter = apply_trades_coverage_start(df, config["target"]["trades_coverage_start"])
    before_rows = len(df)
    if not before_rows:
        raise ValueError(f"No {split} rows remain at or after trades coverage start")
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
    df = df.rename(columns={"accepted": "threshold_accepted"})
    df["predicted_side"] = np.where(df["p_up"] >= 0.5, "UP", "DOWN")
    df["predicted_outcome"] = np.where(df["predicted_side"] == "UP", "up", "down")
    df["selected_side"] = df["predicted_side"].astype("string")
    df["selected_outcome"] = df["predicted_outcome"].astype("string")
    df["correct"] = (
        ((df["target"].astype(int) == 1) & df["selected_side"].eq("UP"))
        | ((df["target"].astype(int) == 0) & df["selected_side"].eq("DOWN"))
    )
    df["p_side"] = np.where(df["selected_side"].eq("UP"), df["p_up"], 1.0 - df["p_up"])
    df["direction_confidence"] = (pd.to_numeric(df["p_up"], errors="coerce") - 0.5).abs()
    df["p_bin"] = p_side_bucket(pd.Series(df["p_side"], index=df.index))
    df["p_side_bucket"] = p_side_bucket(pd.Series(df["p_side"], index=df.index))
    df["market_time_bucket"] = session_label(df["decision_time"]).reset_index(drop=True).to_numpy()

    selected = df[["condition_id", "predicted_outcome", "decision_time", "endDate"]].copy()
    joined = trades.merge(
        selected,
        left_on=["condition_id", "outcome"],
        right_on=["condition_id", "predicted_outcome"],
        how="inner",
    )
    start_mask = joined["trade_time"] > joined["decision_time"]
    end_mask = joined["trade_time"] <= joined["endDate"]
    joined = joined.loc[start_mask & end_mask].copy()
    if joined.empty:
        raise ValueError(f"No selected-side trades matched for split {split}")

    joined = joined.sort_values(["condition_id", "predicted_outcome", "decision_time", "price", "trade_time"])
    lowest = joined.groupby(["condition_id", "predicted_outcome", "decision_time"], as_index=False).first()
    lowest = lowest.rename(columns={"price": "chosen_low", "trade_time": "chosen_low_trade_time"})
    lowest["time_to_chosen_low_sec"] = (lowest["chosen_low_trade_time"] - lowest["decision_time"]).dt.total_seconds()
    target_cols = [
        "condition_id",
        "predicted_outcome",
        "decision_time",
        "chosen_low",
        "chosen_low_trade_time",
        "time_to_chosen_low_sec",
    ]
    out = df.merge(lowest[target_cols], on=["condition_id", "predicted_outcome", "decision_time"], how="left")
    missing_low = int(out["chosen_low"].isna().sum())
    out["chosen_low_reason"] = classify_low_join(out, trades, out["chosen_low"])
    out = out.dropna(subset=["p_up", "p_side"]).copy()
    out["target_raw"] = pd.to_numeric(out["chosen_low"], errors="coerce")
    diagnostics, diagnostic_table = low_join_diagnostics(out, split)

    correct = out["correct"].astype(bool)
    wrong = ~correct
    threshold_accepted = out["threshold_accepted"].astype(bool)
    accepted_correct = correct[threshold_accepted]
    wrong_low = pd.to_numeric(out.loc[wrong, "chosen_low"], errors="coerce")
    summary = {
        "split": split,
        "input_rows": int(before_rows),
        **coverage_filter,
        "all_side_rows_before_low_join": int(len(df)),
        "threshold_accepted_rows": int(out["threshold_accepted"].sum()),
        "missing_low_rows": missing_low,
        "output_rows": int(len(out)),
        "accepted_coverage_vs_source": float(out["threshold_accepted"].mean()) if len(out) else float("nan"),
        "all_side_correct_count": int(correct.sum()),
        "all_side_wrong_count": int(wrong.sum()),
        "threshold_accepted_correct_count": int(accepted_correct.sum()),
        "threshold_accepted_wrong_count": int((~accepted_correct).sum()),
        "accepted_accuracy": float(accepted_correct.mean()) if len(accepted_correct) else float("nan"),
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
    return out, summary, diagnostics, diagnostic_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/expected_return/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_deploy_manifest(config)
    model = load_deploy_model(config, manifest)
    trades_dir = resolve_path(config["paths"]["trades_dir"])
    trade_files = sorted(trades_dir.glob("date=*.parquet"))
    if not trade_files:
        raise FileNotFoundError(f"No trade parquet files found in {trades_dir}")
    first_partition_date = trade_files[0].stem.removeprefix("date=")
    configured_coverage_start = pd.Timestamp(config["target"]["trades_coverage_start"])
    configured_date = configured_coverage_start.strftime("%Y-%m-%d")
    if configured_date < first_partition_date:
        raise ValueError(
            "Configured trades_coverage_start precedes the first local trade partition: "
            f"configured={configured_date}, first_partition={first_partition_date}"
        )
    trades = read_trades(trades_dir)

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")

    outputs: dict[str, str] = {}
    summaries: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_build": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "deploy_experiment_id": manifest.get("experiment_id"),
        "threshold_source": manifest.get("threshold_source"),
        "threshold_policy": manifest.get("threshold_policy"),
        "order_window": config["target"]["order_window"],
        "trades_coverage": {
            "configured_start": str(config["target"]["trades_coverage_start"]),
            "first_local_partition": first_partition_date,
            "last_local_partition": trade_files[-1].stem.removeprefix("date="),
            "partition_count": len(trade_files),
        },
    }
    for split, source_key, out_key in [
        ("train", "train_dataset_source", "train_dataset"),
        ("validation", "validation_dataset_source", "validation_dataset"),
    ]:
        data, summary, diagnostics, diagnostic_table = build_split(
            config, manifest, model, split, resolve_path(config["paths"][source_key]), trades
        )
        out_path = resolve_path(config["paths"][out_key])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out_path, index=False)
        outputs[split] = str(out_path)
        summaries[f"{split}_summary"] = summary
        diagnostic_json_path = reports_dir / f"chosen_low_missing_{split}.json"
        diagnostic_csv_path = reports_dir / f"chosen_low_missing_{split}.csv"
        write_json(diagnostic_json_path, diagnostics)
        diagnostic_table.to_csv(diagnostic_csv_path, index=False)
        summaries[f"{split}_summary"]["chosen_low_diagnostic_json"] = str(diagnostic_json_path)
        summaries[f"{split}_summary"]["chosen_low_diagnostic_csv"] = str(diagnostic_csv_path)
    summaries["outputs"] = outputs
    write_json(reports_dir / "target_build_summary.json", summaries)
    print(summaries)


if __name__ == "__main__":
    main()
