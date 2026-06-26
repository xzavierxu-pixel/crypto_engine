#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "upper_bound_mlp"))
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "expected_return"))

from price_estimator_common import load_config, load_deploy_manifest, resolve_path  # noqa: E402
from train_upper_bound_mlp import Preprocessor, UpperBoundMLP, feature_set  # noqa: E402
from train_low_cdf_and_backtest import BacktestResult, backtest_metrics, backtest_with_bid  # noqa: E402
from expected_return_common import git_commit, write_json  # noqa: E402


class BidDataset(Dataset):
    def __init__(self, x: np.ndarray, p_side: np.ndarray, chosen_low: np.ndarray, correct: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.p_side = torch.from_numpy(p_side.astype(np.float32)).view(-1, 1)
        self.chosen_low = torch.from_numpy(chosen_low.astype(np.float32)).view(-1, 1)
        self.correct = torch.from_numpy(correct.astype(np.float32)).view(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.p_side[idx], self.chosen_low[idx], self.correct[idx]


@dataclass(frozen=True)
class CandidateResult:
    tau: float
    lambda_wrong: float
    theta: float
    epoch: int
    calibration_metrics: dict[str, float]
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float] | None
    state_dict: dict[str, torch.Tensor]
    train_result: BacktestResult
    calibration_result: BacktestResult
    validation_result: BacktestResult | None
    raw_predictions: dict[str, np.ndarray]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_fit_calibration(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_col = str(config["split"].get("timestamp_column", "timestamp"))
    ordered = df.sort_values(ts_col).reset_index(drop=True)
    ts = pd.to_datetime(ordered[ts_col], utc=True)
    cutoff = ts.max() - pd.Timedelta(days=int(config["split"]["calibration_tail_days"]))
    fit = ordered.loc[ts < cutoff].copy().reset_index(drop=True)
    calibration = ordered.loc[ts >= cutoff].copy().reset_index(drop=True)
    if fit.empty or calibration.empty:
        raise ValueError("Calibration tail split produced an empty fit or calibration set")
    return fit, calibration


def accepted_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["threshold_accepted", "p_side", "correct", "target", "p_up"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    out = df.loc[df["threshold_accepted"].astype(bool)].copy()
    out = out.dropna(subset=["p_side", "correct", "target", "p_up"]).reset_index(drop=True)
    return out


def hard_window(df: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return {"row_count": int(len(df)), "start": str(ts.min()), "end": str(ts.max())}


def soft_pnl_loss(
    logits: torch.Tensor,
    p_side: torch.Tensor,
    chosen_low: torch.Tensor,
    correct: torch.Tensor,
    tau: float,
    lambda_wrong: float,
    margin_multiplier: float,
) -> torch.Tensor:
    bid = torch.clamp(p_side, 0.0, 1.0) * torch.sigmoid(logits)
    margin = float(margin_multiplier) * float(tau)
    soft_fill = torch.sigmoid((bid - chosen_low - margin) / float(tau))
    pnl = correct * soft_fill * (1.0 - bid) - float(lambda_wrong) * (1.0 - correct) * bid
    return -pnl.mean()


@torch.no_grad()
def predict_bid_raw(model: nn.Module, x: np.ndarray, p_side: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
        z = model(xb).detach().cpu().numpy().reshape(-1)
        ps = np.asarray(p_side[start : start + batch_size], dtype=float)
        rows.append(np.clip(ps * (1.0 / (1.0 + np.exp(-z))), 0.0, ps))
    return np.concatenate(rows) if rows else np.asarray([], dtype=float)


def floor_to_tick(values: np.ndarray, tick_size: float) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=float) / tick_size + 1e-12) * tick_size


def postprocess_bid(raw_bid: np.ndarray, p_side: np.ndarray, theta: float, tick_size: float, min_bid: float) -> np.ndarray:
    raw = np.asarray(raw_bid, dtype=float)
    side = np.asarray(p_side, dtype=float)
    cap = floor_to_tick(side, tick_size)
    bid = floor_to_tick(raw, tick_size)
    bid = np.where(raw >= float(theta), np.maximum(bid, min_bid), 0.0)
    bid = np.minimum(bid, cap)
    bid = np.where(cap >= min_bid, bid, 0.0)
    return np.round(np.clip(bid, 0.0, side), 10)


def theta_candidates(raw_bid: np.ndarray, config: dict[str, Any]) -> list[float]:
    cfg = config["target"]["theta_candidates"]
    values = [float(v) for v in cfg.get("absolute", [])]
    positive = np.asarray(raw_bid, dtype=float)
    positive = positive[np.isfinite(positive)]
    for coverage in [float(v) for v in cfg.get("coverage_quantiles", [])]:
        if len(positive) and 0.0 < coverage <= 1.0:
            values.append(float(np.quantile(positive, max(0.0, 1.0 - coverage))))
    return sorted(set(round(v, 10) for v in values if math.isfinite(v) and v >= 0.0))


def select_theta(
    frame: pd.DataFrame,
    raw_bid: np.ndarray,
    available_count: int,
    config: dict[str, Any],
    min_order_coverage: float,
) -> tuple[float, dict[str, float], BacktestResult, list[dict[str, float]]]:
    tick = float(config["target"]["tick_size"])
    min_bid = float(config["target"]["min_bid"])
    p_side = pd.to_numeric(frame["p_side"], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, float]] = []
    best: tuple[float, float, float, float, float, dict[str, float], BacktestResult] | None = None
    for theta in theta_candidates(raw_bid, config):
        bid = postprocess_bid(raw_bid, p_side, theta, tick, min_bid)
        result = backtest_with_bid(frame, bid)
        metrics = backtest_metrics(frame, result, available_count)
        row = {"theta": float(theta), **metrics}
        rows.append(row)
        if metrics["order_coverage"] + 1e-12 < min_order_coverage:
            continue
        win_loss = metrics["win_pnl_sum"] / abs(metrics["loss_pnl_sum"]) if metrics["loss_pnl_sum"] < 0 else float("inf")
        key = (
            metrics["mean_accepted_pnl"],
            metrics["sum_pnl"],
            metrics["order_coverage"],
            -metrics["wrong_mean_bid"] if "wrong_mean_bid" in metrics else 0.0,
            -metrics["mean_bid"],
        )
        candidate = (*key, metrics, result)
        if best is None or candidate[:5] > best[:5]:
            best = candidate
            best_metrics = dict(metrics)
            best_metrics["win_loss_ratio"] = float(win_loss)
            selected_theta = float(theta)
    if best is None:
        raise ValueError(f"No theta satisfies order_coverage >= {min_order_coverage}")
    return selected_theta, best_metrics, best[-1], rows


def enrich_bid_metrics(df: pd.DataFrame, result: BacktestResult, metrics: dict[str, float]) -> dict[str, float]:
    correct = df["correct"].astype(bool).to_numpy()
    submitted = result.bid > 0.0
    wrong_submitted = (~correct) & submitted
    correct_submitted = correct & submitted
    out = dict(metrics)
    out["wrong_order_rate"] = float(wrong_submitted.sum() / max((~correct).sum(), 1))
    out["wrong_mean_bid"] = float(result.bid[wrong_submitted].mean()) if wrong_submitted.any() else float("nan")
    out["submitted_correct_count"] = float(correct_submitted.sum())
    out["submitted_wrong_count"] = float(wrong_submitted.sum())
    out["filled_accuracy"] = float(correct[result.filled].mean()) if result.filled.any() else float("nan")
    out["win_loss_ratio"] = (
        float(out["win_pnl_sum"] / abs(out["loss_pnl_sum"])) if out.get("loss_pnl_sum", 0.0) < 0 else float("inf")
    )
    return out


def grouped_tables(df: pd.DataFrame, result: BacktestResult) -> dict[str, list[dict[str, Any]]]:
    work = df.copy()
    work["bid"] = result.bid
    work["pnl"] = result.pnl
    work["ordered"] = result.bid > 0.0
    tables: dict[str, list[dict[str, Any]]] = {}
    for col in ["p_side_bucket", "selected_side", "market_time_bucket"]:
        if col not in work.columns:
            continue
        rows: list[dict[str, Any]] = []
        for value, part in work.groupby(col, dropna=False):
            idx = part.index.to_numpy()
            rows.append(
                {
                    "value": str(value),
                    "sample_count": int(len(part)),
                    "order_coverage": float(part["ordered"].mean()) if len(part) else float("nan"),
                    "mean_bid": float(part.loc[part["ordered"], "bid"].mean()) if part["ordered"].any() else float("nan"),
                    "sum_pnl": float(part["pnl"].sum()),
                    "mean_accepted_pnl": float(part["pnl"].mean()) if len(part) else float("nan"),
                    "accepted_sample_accuracy": float(part["correct"].astype(bool).mean()) if len(part) else float("nan"),
                    "correct_fill_rate": float(result.filled[idx][part["correct"].astype(bool).to_numpy()].mean())
                    if part["correct"].astype(bool).any()
                    else float("nan"),
                }
            )
        tables[col] = rows
    return tables


def write_predictions(path: Path, df: pd.DataFrame, raw_bid: np.ndarray, result: BacktestResult) -> None:
    cols = [
        "timestamp",
        "decision_time",
        "condition_id",
        "polymarket_slug",
        "selected_side",
        "p_up",
        "p_side",
        "selected_t_up",
        "selected_t_down",
        "target",
        "correct",
        "chosen_low",
        "chosen_low_trade_time",
        "time_to_chosen_low_sec",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    out["bid_raw"] = raw_bid
    out["bid"] = result.bid
    out["filled"] = result.filled
    out["printed_filled"] = result.printed_filled
    out["realized_pnl"] = result.pnl
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def train_candidate(
    tau: float,
    lambda_wrong: float,
    config: dict[str, Any],
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    x_fit: np.ndarray,
    x_calibration: np.ndarray,
    x_validation: np.ndarray,
    device: torch.device,
) -> tuple[CandidateResult, list[dict[str, Any]]]:
    seed = int(config["training"]["random_seed"]) + int(round(tau * 100000)) + int(round(lambda_wrong * 100))
    set_seed(seed)
    model = UpperBoundMLP(
        input_dim=x_fit.shape[1],
        hidden_dims=[int(v) for v in config["model"]["hidden_dims"]],
        dropout=[float(v) for v in config["model"]["dropout"]],
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    correct_fit = fit["correct"].astype(bool)
    finite_low = pd.to_numeric(fit["chosen_low"], errors="coerce").notna()
    train_mask = ((~correct_fit) | finite_low).to_numpy()
    chosen_low_train = pd.to_numeric(fit.loc[train_mask, "chosen_low"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    dataset = BidDataset(
        x_fit[train_mask],
        pd.to_numeric(fit.loc[train_mask, "p_side"], errors="coerce").to_numpy(dtype=float),
        chosen_low_train,
        fit.loc[train_mask, "correct"].astype(bool).to_numpy(dtype=float),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
        generator=torch.Generator().manual_seed(seed),
    )
    batch_size = int(config["training"]["batch_size"])
    min_cal_coverage = float(config["objective"]["calibration_min_order_coverage"])
    patience = int(config["training"].get("early_stop_patience", 6))
    margin_multiplier = float(config["model"].get("soft_fill_margin_multiplier", 0.0))
    best: CandidateResult | None = None
    stale = 0
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for xb, p_side_b, low_b, correct_b in loader:
            xb = xb.to(device)
            p_side_b = p_side_b.to(device)
            low_b = low_b.to(device)
            correct_b = correct_b.to(device)
            opt.zero_grad(set_to_none=True)
            loss = soft_pnl_loss(model(xb), p_side_b, low_b, correct_b, tau, lambda_wrong, margin_multiplier)
            loss.backward()
            clip = float(config["training"].get("gradient_clip_norm", 0.0))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        raw_fit = predict_bid_raw(model, x_fit, fit["p_side"].to_numpy(dtype=float), device, batch_size)
        raw_cal = predict_bid_raw(model, x_calibration, calibration["p_side"].to_numpy(dtype=float), device, batch_size)
        try:
            theta, cal_metrics, cal_result, theta_rows = select_theta(
                calibration,
                raw_cal,
                len(calibration),
                config,
                min_cal_coverage,
            )
        except ValueError:
            stale += 1
            epoch_rows.append({"tau": tau, "lambda_wrong": lambda_wrong, "epoch": epoch, "loss": float(np.mean(losses)), "eligible": False})
            if stale >= patience:
                break
            continue
        train_bid = postprocess_bid(
            raw_fit,
            fit["p_side"].to_numpy(dtype=float),
            theta,
            float(config["target"]["tick_size"]),
            float(config["target"]["min_bid"]),
        )
        train_result = backtest_with_bid(fit, train_bid)
        train_metrics = enrich_bid_metrics(fit, train_result, backtest_metrics(fit, train_result, len(fit)))
        cal_metrics = enrich_bid_metrics(calibration, cal_result, cal_metrics)
        epoch_rows.append(
            {
                "tau": tau,
                "lambda_wrong": lambda_wrong,
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "theta": theta,
                "calibration_mean_accepted_pnl": cal_metrics["mean_accepted_pnl"],
                "calibration_sum_pnl": cal_metrics["sum_pnl"],
                "calibration_order_coverage": cal_metrics["order_coverage"],
                "calibration_wrong_mean_bid": cal_metrics["wrong_mean_bid"],
                "eligible": True,
            }
        )
        candidate = CandidateResult(
            tau=tau,
            lambda_wrong=lambda_wrong,
            theta=theta,
            epoch=epoch,
            calibration_metrics=cal_metrics,
            train_metrics=train_metrics,
            validation_metrics=None,
            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            train_result=train_result,
            calibration_result=cal_result,
            validation_result=None,
            raw_predictions={"train": raw_fit, "calibration": raw_cal},
        )
        if best is None or (
            cal_metrics["mean_accepted_pnl"],
            cal_metrics["sum_pnl"],
            cal_metrics["order_coverage"],
            -cal_metrics["wrong_mean_bid"],
        ) > (
            best.calibration_metrics["mean_accepted_pnl"],
            best.calibration_metrics["sum_pnl"],
            best.calibration_metrics["order_coverage"],
            -best.calibration_metrics["wrong_mean_bid"],
        ):
            best = candidate
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best is None:
        raise ValueError(f"No eligible epoch for tau={tau}, lambda_wrong={lambda_wrong}")
    return best, epoch_rows


def evaluate_validation_once(
    selected: CandidateResult,
    config: dict[str, Any],
    validation: pd.DataFrame,
    x_validation: np.ndarray,
    device: torch.device,
    input_dim: int,
) -> CandidateResult:
    model = UpperBoundMLP(
        input_dim=input_dim,
        hidden_dims=[int(v) for v in config["model"]["hidden_dims"]],
        dropout=[float(v) for v in config["model"]["dropout"]],
    ).to(device)
    model.load_state_dict(selected.state_dict)
    raw_val = predict_bid_raw(
        model,
        x_validation,
        validation["p_side"].to_numpy(dtype=float),
        device,
        int(config["training"]["batch_size"]),
    )
    val_bid = postprocess_bid(
        raw_val,
        validation["p_side"].to_numpy(dtype=float),
        selected.theta,
        float(config["target"]["tick_size"]),
        float(config["target"]["min_bid"]),
    )
    val_result = backtest_with_bid(validation, val_bid)
    val_metrics = enrich_bid_metrics(validation, val_result, backtest_metrics(validation, val_result, len(validation)))
    return CandidateResult(
        tau=selected.tau,
        lambda_wrong=selected.lambda_wrong,
        theta=selected.theta,
        epoch=selected.epoch,
        calibration_metrics=selected.calibration_metrics,
        train_metrics=selected.train_metrics,
        validation_metrics=val_metrics,
        state_dict=selected.state_dict,
        train_result=selected.train_result,
        calibration_result=selected.calibration_result,
        validation_result=val_result,
        raw_predictions={**selected.raw_predictions, "validation": raw_val},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/bid_policy/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_deploy_manifest(config)
    set_seed(int(config["training"]["random_seed"]))
    device_name = str(config["training"].get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if device.type == "auto":
        device = torch.device("cpu")

    train_all_raw = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    validation_all_raw = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    train_all = accepted_frame(train_all_raw)
    validation = accepted_frame(validation_all_raw)
    fit, calibration = split_fit_calibration(train_all, config)

    columns, cat_cols = feature_set(config, train_all)
    forbidden_present = sorted(set(config["features"]["forbidden_columns"]).intersection(columns))
    if forbidden_present:
        raise ValueError(f"Forbidden columns selected as features: {forbidden_present}")
    preprocessor = Preprocessor.fit(fit, columns, cat_cols)
    x_fit = preprocessor.transform(fit)
    x_calibration = preprocessor.transform(calibration)
    x_validation = preprocessor.transform(validation)

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")

    candidates: list[CandidateResult] = []
    epoch_rows: list[dict[str, Any]] = []
    for tau in [float(v) for v in config["model"]["tau_grid"]]:
        for lambda_wrong in [float(v) for v in config["model"]["lambda_wrong_grid"]]:
            candidate, rows = train_candidate(
                tau,
                lambda_wrong,
                config,
                fit,
                calibration,
                validation,
                x_fit,
                x_calibration,
                x_validation,
                device,
            )
            candidates.append(candidate)
            epoch_rows.extend(rows)

    selected = max(
        candidates,
        key=lambda item: (
            item.calibration_metrics["mean_accepted_pnl"],
            item.calibration_metrics["sum_pnl"],
            item.calibration_metrics["order_coverage"],
            -item.calibration_metrics["wrong_mean_bid"],
        ),
    )
    selected = evaluate_validation_once(selected, config, validation, x_validation, device, x_fit.shape[1])
    if selected.validation_result is None or selected.validation_metrics is None:
        raise RuntimeError("Selected candidate is missing validation evaluation")

    checkpoint_path = models_dir / "predicted_side_soft_pnl_bid_policy.pt"
    torch.save(
        {
            "state_dict": selected.state_dict,
            "model": config["model"],
            "preprocessor": preprocessor.to_dict(),
            "feature_columns": columns,
            "categorical_columns": cat_cols,
            "selected_tau": selected.tau,
            "selected_lambda_wrong": selected.lambda_wrong,
            "selected_theta": selected.theta,
            "target": config["target"],
        },
        checkpoint_path,
    )

    write_predictions(resolve_path(config["paths"]["predictions_train"]), fit, selected.raw_predictions["train"], selected.train_result)
    write_predictions(
        resolve_path(config["paths"]["predictions_calibration"]),
        calibration,
        selected.raw_predictions["calibration"],
        selected.calibration_result,
    )
    write_predictions(
        resolve_path(config["paths"]["predictions_validation"]),
        validation,
        selected.raw_predictions["validation"],
        selected.validation_result,
    )
    pd.DataFrame(epoch_rows).to_csv(reports_dir / "epoch_policy_search.csv", index=False)
    candidate_rows = [
        {
            "tau": item.tau,
            "lambda_wrong": item.lambda_wrong,
            "theta": item.theta,
            "epoch": item.epoch,
            **{f"calibration_{k}": v for k, v in item.calibration_metrics.items()},
            **{f"validation_{k}": v for k, v in (item.validation_metrics or {}).items()},
        }
        for item in candidates
    ]
    pd.DataFrame(candidate_rows).to_csv(reports_dir / "candidate_summary.csv", index=False)
    tables = grouped_tables(validation, selected.validation_result)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(reports_dir / f"validation_by_{name}.csv", index=False)

    val = selected.validation_metrics
    cal = selected.calibration_metrics
    baseline = config["baseline"]
    report = {
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit(),
        "git_commit_at_training": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "summary_metrics.json"),
        "primary_metric": "validation mean_accepted_pnl with accepted-universe order_coverage >= 0.70",
        "objective": config["objective"],
        "model_family": config["model"]["family"],
        "selected_policy": {
            "tau": selected.tau,
            "lambda_wrong": selected.lambda_wrong,
            "theta": selected.theta,
            "epoch": selected.epoch,
            "selection_source": "calibration_hard_pnl",
            "soft_fill_margin_multiplier": float(config["model"].get("soft_fill_margin_multiplier", 0.0)),
        },
        "leakage_note": (
            "Fit/calibration use accepted development rows only. Validation is evaluated once after selecting "
            "tau/lambda/theta from calibration. Features exclude target/correct/chosen_low/future/trade fields."
        ),
        "deploy_experiment_id": manifest.get("experiment_id"),
        "deploy_training_mode": config["metadata"]["deploy_training_mode"],
        "offline_validation_metric_source": config["metadata"]["offline_validation_metric_source"],
        "order_window": config["target"]["order_window"],
        "trades_coverage_start": config["target"]["trades_coverage_start"],
        "train_metrics": selected.train_metrics,
        "calibration_metrics": cal,
        "validation_metrics": val,
        "train_window": hard_window(fit),
        "calibration_window": hard_window(calibration),
        "validation_window": hard_window(validation),
        "validation_group_diagnostics": tables,
        "candidate_count": len(candidates),
        "coverage_constraint_satisfied": bool(val["order_coverage"] >= float(config["objective"]["min_order_coverage"])),
        "direction_coverage_constraint_satisfied": bool(
            len(validation) / len(validation_all_raw) >= float(config["objective"]["min_direction_coverage"])
        ),
        "target_mean_accepted_pnl_satisfied": bool(
            val["mean_accepted_pnl"] > float(config["objective"]["target_validation_mean_accepted_pnl"])
        ),
        "target_sum_pnl_satisfied": bool(val["sum_pnl"] > float(config["objective"]["target_validation_sum_pnl"])),
        "target_win_loss_ratio_satisfied": bool(val["win_loss_ratio"] > float(config["objective"]["target_win_loss_ratio"])),
        "baseline_comparison": {
            "h2_validation_mean_accepted_pnl": float(baseline["h2_validation_mean_accepted_pnl"]),
            "h2_validation_sum_pnl": float(baseline["h2_validation_sum_pnl"]),
            "h2_win_loss_ratio": float(baseline["h2_win_loss_ratio"]),
            "delta_mean_accepted_pnl": float(val["mean_accepted_pnl"] - float(baseline["h2_validation_mean_accepted_pnl"])),
            "delta_sum_pnl": float(val["sum_pnl"] - float(baseline["h2_validation_sum_pnl"])),
            "delta_order_coverage_vs_h2": float(val["order_coverage"] - 0.5403596021423106),
        },
        "signal_coverage": float(len(validation) / len(validation_all_raw)),
        "accepted_universe_order_coverage": val["order_coverage"],
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "config_snapshot": str(reports_dir / "config_used.yaml"),
            "epoch_policy_search": str(reports_dir / "epoch_policy_search.csv"),
            "candidate_summary": str(reports_dir / "candidate_summary.csv"),
            "train_predictions": str(resolve_path(config["paths"]["predictions_train"])),
            "calibration_predictions": str(resolve_path(config["paths"]["predictions_calibration"])),
            "validation_predictions": str(resolve_path(config["paths"]["predictions_validation"])),
        },
        "feature_count_raw": len(columns),
        "feature_count_encoded": int(x_fit.shape[1]),
        "feature_columns": columns,
        "categorical_columns": cat_cols,
    }
    write_json(reports_dir / "summary_metrics.json", report)
    print(json.dumps({"selected_policy": report["selected_policy"], "validation_metrics": val}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
