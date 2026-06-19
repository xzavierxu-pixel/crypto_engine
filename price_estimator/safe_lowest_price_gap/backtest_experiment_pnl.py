#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "price_estimator" / "upper_bound_mlp"))

from price_estimator.expected_return.expected_return_common import choose_side  # noqa: E402
from train_upper_bound_mlp import Preprocessor, UpperBoundMLP  # noqa: E402


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _refresh_deploy_probability(frame: pd.DataFrame, manifest_path: Path, manifest: dict[str, Any]) -> float:
    model_path = manifest_path.parent / f"{manifest['model_plugin']}.binary.pkl"
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    model = payload.get("model") if isinstance(payload, dict) else payload
    columns = [str(column) for column in manifest["feature_columns"]]
    probabilities = np.asarray(model.predict_proba(frame[columns]), dtype=float)
    refreshed = probabilities[:, 1] if probabilities.ndim == 2 else probabilities.reshape(-1)
    stored = pd.to_numeric(frame["p_up"], errors="raise").to_numpy(dtype=float)
    max_delta = float(np.max(np.abs(refreshed - stored)))
    if max_delta > 1e-12:
        raise ValueError(f"Validation p_up is stale relative to deploy model: max_abs_delta={max_delta}")
    frame["p_up"] = refreshed
    return max_delta


def realized_order_pnl(
    correct: np.ndarray,
    chosen_low: np.ndarray,
    bid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conservative limit-order replay used by the expected-return backtester."""
    correct = np.asarray(correct, dtype=bool)
    chosen_low = np.asarray(chosen_low, dtype=float)
    bid = np.asarray(bid, dtype=float)
    submitted = np.isfinite(bid) & (bid > 0.0)
    printed_filled = submitted & np.isfinite(chosen_low) & (chosen_low <= bid + 1e-12)
    filled = (correct & printed_filled) | ((~correct) & submitted)
    pnl = np.zeros(len(bid), dtype=float)
    pnl[filled & correct] = 1.0 - bid[filled & correct]
    pnl[filled & ~correct] = -bid[filled & ~correct]
    return pnl, filled, printed_filled


class _CheckpointSafeGapModel:
    """Reader supporting both additive-delta and normalized-delta checkpoints."""

    def __init__(self, checkpoint_path: Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.payload = payload
        self.preprocessor = Preprocessor(**payload["preprocessor"])
        model_cfg = payload["model"]
        self.model = UpperBoundMLP(payload["input_dim"], model_cfg["hidden_dims"], model_cfg["dropout"])
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = self.preprocessor.transform(frame)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x)).numpy().reshape(-1)
        f_model = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        calibration = self.payload["calibration"]
        target = self.payload["target"]
        p_side = pd.to_numeric(frame["p_side"], errors="coerce").to_numpy(dtype=float)
        if "delta_norm" in calibration:
            fallback_delta = float(calibration["delta_norm"])
            delta_model = calibration.get("delta_model", {})
            if str(delta_model.get("mode", "global")) == "pside_bin":
                edges = np.asarray(delta_model["edges"], dtype=float)
                indexes = np.clip(np.digitize(p_side, edges, right=False), 1, len(edges) - 1)
                by_index = {int(row["bucket_index"]): float(row["delta_norm"]) for row in delta_model["table"]}
                delta = np.asarray([by_index.get(int(index), fallback_delta) for index in indexes])
            else:
                delta = np.full(len(frame), float(delta_model.get("fallback_delta_norm", fallback_delta)))
            s_floor = float(self.payload["loss"]["s_floor"])
            raw = f_model + delta * np.clip(p_side - f_model, s_floor, None)
        else:
            raw = f_model + float(calibration["delta"])
        tick = float(target["tick_size"])
        tol = float(target["tick_rounding_tolerance"])
        p_pred = np.minimum(np.ceil(raw / tick - tol) * tick, p_side)
        bucket_model = calibration["bucket_model"]
        keys = [str(key) for key in bucket_model.get("keys", [])]
        labels = frame[keys].astype("string").fillna("missing").astype(str).agg("|".join, axis=1)
        table = bucket_model.get("table", {})
        global_miss = float(bucket_model.get("global_miss_rate", 1.0))
        miss = labels.map(lambda value: float(table.get(str(value), {}).get("miss_rate", global_miss)))
        conf_ok = miss.to_numpy(dtype=float) <= float(calibration["bucket_miss_threshold"])
        action = np.full(len(frame), "active", dtype=object)
        clamp = conf_ok & (raw >= p_side)
        action[clamp] = "clamp_over_pside"
        action[~conf_ok] = "abstain_low_conf"
        p_pred[clamp | ~conf_ok] = p_side[clamp | ~conf_ok]
        return pd.DataFrame({"p_pred": p_pred, "safe_gap_action": action, "safe_gap_conf_ok": conf_ok})


def _load_price_model(experiment_dir: Path) -> _CheckpointSafeGapModel:
    checkpoint = experiment_dir / "models" / "safe_lowest_price_gap.pt"
    if checkpoint.exists():
        return _CheckpointSafeGapModel(checkpoint)
    raise FileNotFoundError(f"No safe-lowest-price-gap artifact under {experiment_dir}")


def _direction_metrics(frame: pd.DataFrame, available_count: int) -> dict[str, float]:
    correct = frame["correct"].astype(bool).to_numpy()
    up = frame["selected_side"].astype("string").eq("UP").to_numpy()
    coverage = len(frame) / available_count
    accuracy = float(correct.mean())
    utility = coverage * (2.0 * accuracy - 1.0)
    downside = math.sqrt(coverage * (1.0 - accuracy))
    target = pd.to_numeric(frame["target"], errors="raise").to_numpy(dtype=int)
    p_up = pd.to_numeric(frame["p_up"], errors="raise").to_numpy(dtype=float)
    up_count = int(up.sum())
    down_count = int((~up).sum())
    return {
        "sample_count": float(available_count), "coverage": float(coverage),
        "precision_up": float(correct[up].mean()), "precision_down": float(correct[~up].mean()),
        "balanced_precision": float((correct[up].mean() + correct[~up].mean()) / 2.0),
        "all_sample_accuracy": float(correct.sum() / available_count),
        "accepted_sample_accuracy": accuracy,
        "share_up_predictions": float(up.mean()), "share_down_predictions": float((~up).mean()),
        "selected_t_up": float(pd.to_numeric(frame["selected_t_up"]).mean()),
        "selected_t_down": float(pd.to_numeric(frame["selected_t_down"]).mean()),
        "accepted_count": float(len(frame)), "up_prediction_count": float(up_count),
        "down_prediction_count": float(down_count),
        "roc_auc": float(roc_auc_score(target, p_up)),
        "brier_score": float(brier_score_loss(target, p_up)),
        "log_loss": float(log_loss(target, p_up, labels=[0, 1])),
        "utility": float(utility), "downside_risk": float(downside),
        "selection_score": float(utility / downside),
        "up_signal_count": float(up_count), "down_signal_count": float(down_count),
        "total_signal_count": float(len(frame)), "signal_coverage": float(coverage),
        "overall_signal_accuracy": accuracy,
    }


def _pnl_metrics(frame: pd.DataFrame, predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    bid = pd.to_numeric(predictions["p_pred"], errors="coerce").to_numpy(dtype=float)
    correct = frame["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(frame["chosen_low"], errors="coerce").to_numpy(dtype=float)
    pnl, filled, printed_filled = realized_order_pnl(correct, low, bid)
    submitted = np.isfinite(bid) & (bid > 0.0)
    out = frame[["timestamp", "decision_time", "condition_id", "polymarket_slug", "selected_side", "p_up", "p_side", "target", "correct", "chosen_low"]].copy()
    out["bid"] = bid
    out["safe_gap_action"] = predictions["safe_gap_action"].to_numpy()
    out["submitted"] = submitted
    out["printed_filled"] = printed_filled
    out["filled"] = filled
    out["realized_pnl_per_share"] = pnl
    win = filled & correct
    loss = filled & ~correct
    metrics = {
        "order_count": int(submitted.sum()), "filled_count": int(filled.sum()),
        "unfilled_count": int(submitted.sum() - filled.sum()),
        "fill_rate": float(filled.sum() / submitted.sum()),
        "winning_fill_count": int(win.sum()), "losing_fill_count": int(loss.sum()),
        "filled_accuracy": float(correct[filled].mean()),
        "mean_bid": float(bid[submitted].mean()),
        "win_pnl": float(pnl[win].sum()), "loss_pnl": float(pnl[loss].sum()),
        "total_pnl_per_one_share": float(pnl.sum()),
        "mean_pnl_per_order": float(pnl[submitted].mean()),
        "mean_pnl_per_fill": float(pnl[filled].mean()),
        "correct_signal_fill_rate": float(filled[correct].mean()),
        "wrong_signal_forced_fill_rate": float(filled[~correct].mean()),
        "missing_chosen_low_count": int(np.isnan(low).sum()),
        "action_counts": {str(k): int(v) for k, v in predictions["safe_gap_action"].value_counts().items()},
    }
    return metrics, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay safe-lowest-price-gap experiments with deploy-baseline directions.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validation_path = _resolve(config["paths"]["validation_dataset"])
    deploy_manifest_path = _resolve(config["paths"]["deploy_manifest"])
    experiments_dir = _resolve(config["paths"]["experiments_dir"])
    output_dir = _resolve(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(validation_path)
    manifest = json.loads(deploy_manifest_path.read_text(encoding="utf-8"))
    max_p_up_delta = _refresh_deploy_probability(frame, deploy_manifest_path, manifest)
    recalculated = choose_side(frame["p_up"], frame["decision_time"], manifest)
    stored = frame["threshold_accepted"].astype(bool).reset_index(drop=True)
    if not np.array_equal(stored.to_numpy(), recalculated["accepted"].to_numpy()):
        raise ValueError("Validation direction data is stale relative to the deploy manifest threshold policy; rebuild it first.")
    accepted = frame.loc[stored.to_numpy()].copy().reset_index(drop=True)
    expected_side = recalculated.loc[recalculated["accepted"], "selected_side"].reset_index(drop=True)
    if not accepted["selected_side"].astype("string").reset_index(drop=True).equals(expected_side.astype("string")):
        raise ValueError("Stored selected_side does not match the deploy manifest threshold policy.")

    direction_metrics = _direction_metrics(accepted, len(frame))
    results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    if bool(config["backtest"].get("include_p_side_baseline", False)):
        p_side_predictions = pd.DataFrame(
            {
                "p_pred": pd.to_numeric(accepted["p_side"], errors="raise").to_numpy(dtype=float),
                "safe_gap_action": np.full(len(accepted), "p_side", dtype=object),
            }
        )
        metrics, replay = _pnl_metrics(accepted, p_side_predictions)
        replay.insert(0, "experiment_id", "p_side_full_price")
        prediction_frames.append(replay)
        results.append({"experiment_id": "p_side_full_price", **metrics})
    for experiment_dir in sorted(path for path in experiments_dir.iterdir() if path.is_dir()):
        if not (experiment_dir / "models").exists():
            continue
        model = _load_price_model(experiment_dir)
        predictions = model.predict(accepted)
        metrics, replay = _pnl_metrics(accepted, predictions)
        replay.insert(0, "experiment_id", experiment_dir.name)
        prediction_frames.append(replay)
        results.append({"experiment_id": experiment_dir.name, **metrics})

    results.sort(key=lambda row: row["total_pnl_per_one_share"], reverse=True)
    validation_window = {
        "row_count": int(len(frame)), "start": str(frame["decision_time"].min()), "end": str(frame["decision_time"].max())
    }
    report = {
        "experiment_id": config["experiment_id"],
        "git_commit": _git_commit(),
        "git_commit_before_experiment": _git_commit(),
        "config_path": str(config_path.relative_to(ROOT)),
        "report_path": str((output_dir / "report.json").relative_to(ROOT)),
        "primary_metric": "total_pnl_per_one_share",
        "pnl_unit": "USDC per one token share ordered per accepted baseline signal",
        "fill_assumption": config["backtest"]["fill_assumption"],
        "baseline_manifest": str(deploy_manifest_path.relative_to(ROOT)),
        "baseline_threshold_source": manifest.get("threshold_source"),
        "baseline_p_up_max_abs_delta_vs_stored": max_p_up_delta,
        "train_metrics": manifest.get("train_metrics", {}),
        "train_window": manifest.get("train_window", {}),
        "validation_metrics": direction_metrics,
        "validation_window": validation_window,
        "coverage_constraint_satisfied": bool(direction_metrics["coverage"] >= float(config["objective"]["min_coverage"])),
        "price_experiment_results": results,
    }
    (output_dir / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_filename = str(config["backtest"].get("summary_filename", "pnl_summary.csv"))
    predictions_filename = str(config["backtest"].get("predictions_filename", "predictions_validation.parquet"))
    pd.DataFrame(results).to_csv(output_dir / summary_filename, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output_dir / predictions_filename, index=False)
    print(json.dumps({"output_dir": str(output_dir), "best": results[0], "experiment_count": len(results)}, indent=2))


if __name__ == "__main__":
    main()
