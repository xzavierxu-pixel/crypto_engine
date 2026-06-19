#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
ROOT = PRICE_ESTIMATOR_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "upper_bound_mlp"))

from price_estimator_common import load_config, load_deploy_manifest, resolve_path  # noqa: E402
from train_upper_bound_mlp import Preprocessor, UpperBoundMLP, feature_set  # noqa: E402

from expected_return_common import empirical_cdf, git_commit, price_grid, write_json  # noqa: E402


class LowPointDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class HazardDataset(Dataset):
    def __init__(self, x: np.ndarray, event_index: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.event_index = torch.from_numpy(event_index.astype(np.int64))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.event_index[idx]


class HazardMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: list[float],
        output_dim: int,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != len(dropout):
            raise ValueError("hidden_dims and dropout must have the same length")
        layers: list[nn.Module] = []
        width = input_dim
        for next_width, drop in zip(hidden_dims, dropout):
            layers.extend([nn.Linear(width, next_width), nn.ReLU(), nn.Dropout(drop)])
            width = next_width
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.trunk(x))


@dataclass(frozen=True)
class BacktestResult:
    bid: np.ndarray
    expected_ev: np.ndarray
    fill_prob: np.ndarray
    pnl: np.ndarray
    filled: np.ndarray
    printed_filled: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_fit_calibration(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_col = str(config["split"].get("timestamp_column", "timestamp"))
    days = int(config["split"].get("calibration_tail_days", 31))
    ordered = df.sort_values(ts_col).reset_index(drop=True)
    ts = pd.to_datetime(ordered[ts_col], utc=True)
    cutoff = ts.max() - pd.Timedelta(days=days)
    fit = ordered.loc[ts < cutoff].copy()
    calibration = ordered.loc[ts >= cutoff].copy()
    if fit.empty or calibration.empty:
        raise ValueError("Calibration tail split produced an empty fit or calibration set")
    return fit, calibration


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
        z = model(xb).detach().cpu().numpy().reshape(-1)
        out.append(1.0 / (1.0 + np.exp(-z)))
    return np.concatenate(out) if out else np.asarray([], dtype=float)


def build_tick_grid(tick_size: float, max_price: float) -> np.ndarray:
    if tick_size <= 0.0 or max_price < tick_size:
        raise ValueError("tick_size must be positive and max_price must cover at least one tick")
    count = int(math.floor(max_price / tick_size + 1e-12))
    return np.round(np.arange(1, count + 1, dtype=float) * tick_size, 10)


def event_indices(chosen_low: np.ndarray, tick_grid: np.ndarray) -> np.ndarray:
    """Return the first grid index at or above the low; len(grid) means censored above it."""
    values = np.asarray(chosen_low, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Hazard training rows must have finite chosen_low values")
    return np.searchsorted(tick_grid, values, side="left").astype(np.int64)


def hazard_nll(
    logits: torch.Tensor,
    index: torch.Tensor,
    smoothness_penalty: float = 0.0,
) -> torch.Tensor:
    k = logits.shape[1]
    positions = torch.arange(k, device=logits.device).view(1, -1)
    mask = positions <= index.view(-1, 1)
    event = (positions == index.view(-1, 1)) & (index.view(-1, 1) < k)
    target = event.to(logits.dtype)
    cell_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = (cell_loss * mask.to(logits.dtype)).sum(dim=1).mean()
    if smoothness_penalty > 0.0 and k > 1:
        hazard = torch.sigmoid(logits)
        loss = loss + smoothness_penalty * torch.mean((hazard[:, 1:] - hazard[:, :-1]) ** 2)
    return loss


def survival_cdf(logits: torch.Tensor) -> torch.Tensor:
    """Stable 1-prod(1-hazard), monotone along the price grid by construction."""
    log_survival = torch.cumsum(F.logsigmoid(-logits), dim=1)
    return -torch.expm1(log_survival)


@torch.no_grad()
def predict_hazard(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    hazard_rows: list[np.ndarray] = []
    cdf_rows: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
        logits = model(xb)
        hazard_rows.append(torch.sigmoid(logits).cpu().numpy())
        cdf_rows.append(survival_cdf(logits).cpu().numpy())
    if not cdf_rows:
        width = int(getattr(model, "output").out_features)
        empty = np.empty((0, width), dtype=float)
        return empty, empty.copy()
    return np.concatenate(hazard_rows), np.concatenate(cdf_rows)


def evaluate_hazard_nll(
    model: nn.Module,
    x: np.ndarray,
    index: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses: list[float] = []
    weights: list[int] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
            ib = torch.from_numpy(index[start : start + batch_size].astype(np.int64)).to(device)
            losses.append(float(hazard_nll(model(xb), ib).cpu()))
            weights.append(len(xb))
    return float(np.average(losses, weights=weights)) if weights else float("inf")


def train_hazard_model(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal_correct: np.ndarray,
    y_cal_correct: np.ndarray,
    tick_grid: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = HazardMLP(
        input_dim=x_fit.shape[1],
        hidden_dims=[int(v) for v in config["model"]["hidden_dims"]],
        dropout=[float(v) for v in config["model"]["dropout"]],
        output_dim=len(tick_grid),
    ).to(device)
    fit_index = event_indices(y_fit, tick_grid)
    cal_index = event_indices(y_cal_correct, tick_grid)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loader = DataLoader(
        HazardDataset(x_fit, fit_index),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    penalty = float(config["model"].get("hazard", {}).get("smoothness_penalty", 0.0))
    batch_size = int(config["training"]["batch_size"])
    patience = int(config["training"].get("early_stop_patience", 10))
    best_cal = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    rows: list[dict[str, float]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for xb, ib in loader:
            xb, ib = xb.to(device), ib.to(device)
            opt.zero_grad(set_to_none=True)
            loss = hazard_nll(model(xb), ib, penalty)
            loss.backward()
            clip = float(config["training"].get("gradient_clip_norm", 0.0))
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        cal_nll = evaluate_hazard_nll(model, x_cal_correct, cal_index, device, batch_size)
        rows.append({"epoch": float(epoch), "fit_loss": float(np.mean(losses)), "calibration_nll": cal_nll})
        if cal_nll < best_cal:
            best_cal = cal_nll
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rows


def train_point_model(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal_correct: np.ndarray,
    y_cal_correct: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
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
    loader = DataLoader(
        LowPointDataset(x_fit, y_fit),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    loss_fn = nn.SmoothL1Loss()
    rows: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_cal = float("inf")
    stale = 0
    patience = int(config["training"].get("early_stop_patience", 10))
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = torch.sigmoid(model(xb))
            loss = loss_fn(pred, yb)
            loss.backward()
            clip = float(config["training"].get("gradient_clip_norm", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        cal_pred = predict(model, x_cal_correct, device, int(config["training"]["batch_size"]))
        cal_mae = float(np.mean(np.abs(cal_pred - y_cal_correct))) if len(y_cal_correct) else float("inf")
        rows.append({"epoch": float(epoch), "fit_loss": float(np.mean(losses)), "calibration_mae": cal_mae})
        if cal_mae < best_cal:
            best_cal = cal_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rows


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    return {
        "sample_count": float(len(y)),
        "mae": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "rmse": float(np.sqrt(np.mean(err * err))) if len(err) else float("nan"),
        "bias": float(np.mean(err)) if len(err) else float("nan"),
        "pred_mean": float(np.mean(pred)) if len(pred) else float("nan"),
        "target_mean": float(np.mean(y)) if len(y) else float("nan"),
    }


def choose_expected_return_bids(
    p_side: np.ndarray,
    f_pred: np.ndarray,
    residuals: np.ndarray,
    tick_size: float,
    min_bid: float,
    min_ev: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bids = np.zeros(len(p_side), dtype=float)
    evs = np.zeros(len(p_side), dtype=float)
    fill_probs = np.zeros(len(p_side), dtype=float)
    for i, (q, f) in enumerate(zip(p_side, f_pred)):
        grid = price_grid(float(q), tick_size, min_bid)
        gc = empirical_cdf(residuals, grid - float(f))
        ev = float(q) * gc * (1.0 - grid) - (1.0 - float(q)) * grid
        j = int(np.argmax(ev))
        bids[i] = grid[j] if ev[j] > min_ev else 0.0
        evs[i] = ev[j]
        fill_probs[i] = gc[j]
    return bids, evs, fill_probs


def choose_survival_expected_return_bids(
    q_side: np.ndarray,
    gc_matrix: np.ndarray,
    tick_grid: np.ndarray,
    min_bid: float,
    min_ev: float,
    min_fill_probability: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_values = np.asarray(q_side, dtype=float)
    gc_values = np.asarray(gc_matrix, dtype=float)
    if gc_values.shape != (len(q_values), len(tick_grid)):
        raise ValueError("gc_matrix shape must be (sample_count, tick_grid_count)")
    if np.any(np.diff(gc_values, axis=1) < -1e-7):
        raise AssertionError("Survival CDF must be non-decreasing along the price grid")
    bids = np.zeros(len(q_values), dtype=float)
    evs = np.zeros(len(q_values), dtype=float)
    fill_probs = np.zeros(len(q_values), dtype=float)
    for i, q in enumerate(q_values):
        valid = (
            (tick_grid >= min_bid - 1e-12)
            & (tick_grid <= q + 1e-12)
            & (gc_values[i] > min_fill_probability)
        )
        if not valid.any():
            continue
        grid = tick_grid[valid]
        gc = gc_values[i, valid]
        ev = q * gc * (1.0 - grid) - (1.0 - q) * grid
        j = int(np.argmax(ev))
        bids[i] = grid[j] if ev[j] > min_ev else 0.0
        evs[i] = ev[j]
        fill_probs[i] = gc[j]
    return bids, evs, fill_probs


def fit_q_calibrator(calibration: pd.DataFrame) -> IsotonicRegression:
    p_side = pd.to_numeric(calibration["p_side"], errors="coerce").to_numpy(dtype=float)
    correct = calibration["correct"].astype(bool).to_numpy(dtype=float)
    finite = np.isfinite(p_side)
    if finite.sum() < 2 or len(np.unique(p_side[finite])) < 2:
        raise ValueError("Calibration split has insufficient distinct p_side values for isotonic calibration")
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_side[finite], correct[finite])


def calibrated_q(df: pd.DataFrame, calibrator: IsotonicRegression | None) -> np.ndarray:
    p_side = pd.to_numeric(df["p_side"], errors="coerce").to_numpy(dtype=float)
    return p_side if calibrator is None else np.asarray(calibrator.predict(p_side), dtype=float)


def select_min_ev(
    candidates: dict[float, BacktestResult],
    frame: pd.DataFrame,
    available_count: int,
    min_order_count: int,
) -> tuple[float, list[dict[str, float]]]:
    rows = [
        {"min_ev": value, **backtest_metrics(frame, result, available_count)}
        for value, result in sorted(candidates.items())
    ]
    eligible = [row for row in rows if row["order_count"] >= min_order_count]
    if not eligible:
        raise ValueError(f"No min_ev candidate meets min_order_count={min_order_count}")
    selected = max(
        eligible,
        key=lambda row: (row["mean_accepted_pnl"], row["order_count"], row["min_ev"]),
    )
    return float(selected["min_ev"]), rows


def floor_to_tick(values: np.ndarray, tick_size: float) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=float) / tick_size + 1e-12) * tick_size


def realized_pnl(df: pd.DataFrame, bid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = pd.to_numeric(df["chosen_low"], errors="coerce").to_numpy(dtype=float)
    correct = df["correct"].astype(bool).to_numpy()
    submitted = np.asarray(bid, dtype=float) > 0.0
    printed_filled = submitted & np.isfinite(low) & (low <= (np.asarray(bid, dtype=float) + 1e-12))
    filled = (correct & printed_filled) | ((~correct) & submitted)
    pnl = np.zeros(len(df), dtype=float)
    pnl[filled & correct] = 1.0 - bid[filled & correct]
    pnl[filled & (~correct)] = -bid[filled & (~correct)]
    return pnl, filled, printed_filled


def backtest_with_bid(df: pd.DataFrame, bid: np.ndarray, expected_ev: np.ndarray | None = None, fill_prob: np.ndarray | None = None) -> BacktestResult:
    pnl, filled, printed_filled = realized_pnl(df, bid)
    ev = np.zeros(len(df), dtype=float) if expected_ev is None else np.asarray(expected_ev, dtype=float)
    fp = np.full(len(df), float("nan"), dtype=float) if fill_prob is None else np.asarray(fill_prob, dtype=float)
    return BacktestResult(
        bid=np.asarray(bid, dtype=float),
        expected_ev=ev,
        fill_prob=fp,
        pnl=pnl,
        filled=filled,
        printed_filled=printed_filled,
    )


def backtest_metrics(
    df: pd.DataFrame,
    result: BacktestResult,
    available_sample_count: int | None = None,
) -> dict[str, float]:
    correct = df["correct"].astype(bool).to_numpy()
    filled = result.filled
    pnl = result.pnl
    bid = result.bid
    submitted = bid > 0.0
    wrong_submitted = (~correct) & submitted
    correct_submitted = correct & submitted
    calibrated_subset = correct_submitted & np.isfinite(result.fill_prob)
    available = int(available_sample_count) if available_sample_count is not None else len(df)
    coverage = float(len(df) / available) if available else float("nan")
    accuracy = float(correct.mean()) if len(correct) else float("nan")
    predicted_up = df["selected_side"].astype("string").eq("UP").to_numpy()
    up_count = int(predicted_up.sum())
    down_count = int((~predicted_up).sum())
    precision_up = float(correct[predicted_up].mean()) if predicted_up.any() else float("nan")
    precision_down = float(correct[~predicted_up].mean()) if (~predicted_up).any() else float("nan")
    balanced_precision = float(np.nanmean([precision_up, precision_down]))
    utility = coverage * (2.0 * accuracy - 1.0)
    downside_risk = math.sqrt(max(coverage * (1.0 - accuracy), 0.0))
    p_up = pd.to_numeric(df["p_up"], errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=int)
    metrics = {
        "sample_count": float(available),
        "coverage": coverage,
        "precision_up": precision_up,
        "precision_down": precision_down,
        "balanced_precision": balanced_precision,
        "all_sample_accuracy": float(correct.sum() / available) if available else float("nan"),
        "accepted_sample_accuracy": accuracy,
        "share_up_predictions": float(up_count / len(df)) if len(df) else float("nan"),
        "share_down_predictions": float(down_count / len(df)) if len(df) else float("nan"),
        "selected_t_up": float(pd.to_numeric(df["selected_t_up"], errors="coerce").mean()),
        "selected_t_down": float(pd.to_numeric(df["selected_t_down"], errors="coerce").mean()),
        "accepted_count": float(len(df)),
        "up_prediction_count": float(up_count),
        "down_prediction_count": float(down_count),
        "roc_auc": float(roc_auc_score(target, p_up)) if len(np.unique(target)) > 1 else float("nan"),
        "brier_score": float(brier_score_loss(target, p_up)),
        "log_loss": float(log_loss(target, p_up, labels=[0, 1])),
        "utility": utility,
        "downside_risk": downside_risk,
        "selection_score": float(utility / downside_risk) if downside_risk > 0 else float("nan"),
        "up_signal_count": float(up_count),
        "down_signal_count": float(down_count),
        "total_signal_count": float(len(df)),
        "signal_coverage": coverage,
        "overall_signal_accuracy": accuracy,
        "trade_count": float(filled.sum()),
        "order_count": float(submitted.sum()),
        "order_coverage": float(submitted.mean()) if len(submitted) else float("nan"),
        "fill_rate": float(filled.mean()) if len(filled) else float("nan"),
        "correct_count": float(correct.sum()),
        "correct_fill_rate": float(filled[correct].mean()) if correct.any() else float("nan"),
        "wrong_fill_forced": float(filled[wrong_submitted].mean()) if wrong_submitted.any() else float("nan"),
        "wrong_fill_printed": float(result.printed_filled[wrong_submitted].mean()) if wrong_submitted.any() else float("nan"),
        "sum_pnl": float(pnl.sum()) if len(pnl) else float("nan"),
        "mean_accepted_pnl": float(pnl.mean()) if len(pnl) else float("nan"),
        "mean_pnl_filled": float(pnl[filled].mean()) if filled.any() else float("nan"),
        "win_pnl_sum": float(pnl[correct].sum()) if correct.any() else float("nan"),
        "loss_pnl_sum": float(pnl[~correct].sum()) if (~correct).any() else float("nan"),
        "mean_bid": float(np.mean(bid)) if len(bid) else float("nan"),
        "median_bid": float(np.median(bid)) if len(bid) else float("nan"),
        "mean_expected_ev": float(np.mean(result.expected_ev)) if len(result.expected_ev) else float("nan"),
        "negative_expected_ev_share": float((result.expected_ev < 0.0).mean()) if len(result.expected_ev) else float("nan"),
        "mean_model_fill_prob": float(np.mean(result.fill_prob[calibrated_subset])) if calibrated_subset.any() else float("nan"),
        "realized_correct_fill_rate_submitted": float(result.filled[correct_submitted].mean()) if correct_submitted.any() else float("nan"),
        "submitted_fill_calibration_gap": (
            float(np.mean(result.fill_prob[calibrated_subset]) - result.filled[calibrated_subset].mean())
            if calibrated_subset.any()
            else float("nan")
        ),
    }
    return metrics


def write_predictions(
    df: pd.DataFrame,
    result: BacktestResult,
    path: Path,
    q_used: np.ndarray,
    point_pred: np.ndarray | None = None,
) -> None:
    base_cols = [
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
    out = df[[c for c in base_cols if c in df.columns]].copy()
    out.insert(0, "sample_id", np.arange(len(out), dtype=np.int64))
    if point_pred is not None:
        out["gc_point_pred"] = point_pred
    out["q_used"] = q_used
    out["bid"] = result.bid
    out["expected_ev"] = result.expected_ev
    out["model_fill_prob"] = result.fill_prob
    out["filled"] = result.filled
    out["printed_filled"] = result.printed_filled
    out["realized_pnl"] = result.pnl
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def reliability_table(
    correct_df: pd.DataFrame,
    hazard: np.ndarray,
    gc: np.ndarray,
    tick_grid: np.ndarray,
) -> pd.DataFrame:
    low = pd.to_numeric(correct_df["chosen_low"], errors="coerce").to_numpy(dtype=float)
    if len(low) != len(gc):
        raise ValueError("Reliability rows and predictions are misaligned")
    rows: list[dict[str, float]] = []
    for k, bid in enumerate(tick_grid):
        prior = 0.0 if k == 0 else tick_grid[k - 1]
        at_risk = low > prior + 1e-12
        event = at_risk & (low <= bid + 1e-12)
        rows.append(
            {
                "bid": float(bid),
                "sample_count": float(len(low)),
                "risk_count": float(at_risk.sum()),
                "event_count": float(event.sum()),
                "predicted_hazard": float(np.mean(hazard[at_risk, k])) if at_risk.any() else float("nan"),
                "observed_hazard": float(event.sum() / at_risk.sum()) if at_risk.any() else float("nan"),
                "predicted_cdf": float(np.mean(gc[:, k])) if len(gc) else float("nan"),
                "observed_cdf": float(np.mean(low <= bid + 1e-12)) if len(low) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def point_cdf_matrix(point_pred: np.ndarray, residuals: np.ndarray, tick_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gc = np.vstack([empirical_cdf(residuals, tick_grid - float(value)) for value in point_pred])
    hazard = np.empty_like(gc)
    hazard[:, 0] = gc[:, 0]
    prior_survival = np.maximum(1.0 - gc[:, :-1], 1e-12)
    hazard[:, 1:] = np.clip((gc[:, 1:] - gc[:, :-1]) / prior_survival, 0.0, 1.0)
    return hazard, gc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/expected_return/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_deploy_manifest(config)
    seed = int(config["training"]["random_seed"])
    set_seed(seed)
    device_name = str(config["training"].get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    batch_size = int(config["training"]["batch_size"])

    train_all = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    validation = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    fit_all, calibration_all = split_fit_calibration(train_all, config)
    fit_mask = fit_all["correct"].astype(bool) & fit_all["chosen_low"].notna()
    cal_correct_mask = calibration_all["correct"].astype(bool) & calibration_all["chosen_low"].notna()
    validation_correct_mask = validation["correct"].astype(bool) & validation["chosen_low"].notna()
    fit = fit_all.loc[fit_mask].copy()
    calibration_correct = calibration_all.loc[cal_correct_mask].copy()
    validation_correct = validation.loc[validation_correct_mask].copy()
    if fit.empty or calibration_correct.empty:
        raise ValueError("Correct-only fit/calibration data is empty")

    columns, cat_cols = feature_set(config, fit_all)
    preprocessor = Preprocessor.fit(fit_all, columns, cat_cols)
    x_fit = preprocessor.transform(fit)
    x_cal_correct = preprocessor.transform(calibration_correct)
    x_train_all = preprocessor.transform(train_all)
    x_cal_all = preprocessor.transform(calibration_all)
    x_validation = preprocessor.transform(validation)
    y_fit = pd.to_numeric(fit["chosen_low"], errors="coerce").to_numpy(dtype=float)
    y_cal_correct = pd.to_numeric(calibration_correct["chosen_low"], errors="coerce").to_numpy(dtype=float)

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")
    source_summary_path = resolve_path(config["paths"].get("target_summary_source", str(reports_dir / "target_build_summary.json")))
    target_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}

    tick = float(config["target"]["tick_size"])
    min_bid = float(config["target"].get("min_bid", tick))
    hazard_config = config["model"].get("hazard", {})
    tick_grid = build_tick_grid(tick, float(hazard_config.get("max_price", 0.85)))
    family = str(config["model"]["family"])
    point_predictions: dict[str, np.ndarray] = {}
    residuals: np.ndarray | None = None
    if family == "r9_point_plus_global_residual_cdf":
        model, epoch_rows = train_point_model(x_fit, y_fit, x_cal_correct, y_cal_correct, config, device)
        point_predictions = {
            "fit": predict(model, x_fit, device, batch_size),
            "train": predict(model, x_train_all, device, batch_size),
            "calibration": predict(model, x_cal_all, device, batch_size),
            "calibration_correct": predict(model, x_cal_correct, device, batch_size),
            "validation": predict(model, x_validation, device, batch_size),
        }
        residuals = y_cal_correct - point_predictions["calibration_correct"]
        hazard_train, gc_train = point_cdf_matrix(point_predictions["train"], residuals, tick_grid)
        hazard_cal, gc_cal = point_cdf_matrix(point_predictions["calibration"], residuals, tick_grid)
        hazard_validation, gc_validation = point_cdf_matrix(point_predictions["validation"], residuals, tick_grid)
    elif family == "hazard_survival_cdf":
        model, epoch_rows = train_hazard_model(
            x_fit, y_fit, x_cal_correct, y_cal_correct, tick_grid, config, device
        )
        hazard_train, gc_train = predict_hazard(model, x_train_all, device, batch_size)
        hazard_cal, gc_cal = predict_hazard(model, x_cal_all, device, batch_size)
        hazard_validation, gc_validation = predict_hazard(model, x_validation, device, batch_size)
    else:
        raise ValueError(f"Unsupported model.family: {family}")

    if any(np.any(np.diff(matrix, axis=1) < -1e-7) for matrix in [gc_train, gc_cal, gc_validation]):
        raise AssertionError("Gc monotonicity check failed")
    q_calibrator = fit_q_calibrator(calibration_all) if bool(config["model"].get("q_calibration", {}).get("enabled", False)) else None
    q_train = calibrated_q(train_all, q_calibrator)
    q_cal = calibrated_q(calibration_all, q_calibrator)
    q_validation = calibrated_q(validation, q_calibrator)

    train_mask = train_all["threshold_accepted"].astype(bool).to_numpy()
    cal_mask = calibration_all["threshold_accepted"].astype(bool).to_numpy()
    validation_mask = validation["threshold_accepted"].astype(bool).to_numpy()
    train_accepted, calibration_accepted, validation_accepted = (
        train_all.loc[train_mask].copy(), calibration_all.loc[cal_mask].copy(), validation.loc[validation_mask].copy()
    )

    def run_expected(
        frame: pd.DataFrame, q: np.ndarray, gc: np.ndarray, min_ev: float
    ) -> BacktestResult:
        bid, ev, fill_prob = choose_survival_expected_return_bids(q, gc, tick_grid, min_bid, min_ev)
        return backtest_with_bid(frame, bid, ev, fill_prob)

    min_ev_grid = [float(value) for value in config["target"]["min_ev_grid"]]
    cal_candidates = {
        value: run_expected(calibration_accepted, q_cal[cal_mask], gc_cal[cal_mask], value)
        for value in min_ev_grid
    }
    selected_min_ev, min_ev_search = select_min_ev(
        cal_candidates,
        calibration_accepted,
        len(calibration_all),
        int(config["target"].get("min_ev_min_order_count", 100)),
    )
    train_result = run_expected(train_accepted, q_train[train_mask], gc_train[train_mask], selected_min_ev)
    cal_result = cal_candidates[selected_min_ev]
    validation_result = run_expected(
        validation_accepted, q_validation[validation_mask], gc_validation[validation_mask], selected_min_ev
    )
    validation_frontier = []
    for value in min_ev_grid:
        result = run_expected(validation_accepted, q_validation[validation_mask], gc_validation[validation_mask], value)
        validation_frontier.append({"min_ev": value, **backtest_metrics(validation_accepted, result, len(validation))})
    o0_forced = run_expected(validation_accepted, q_validation[validation_mask], gc_validation[validation_mask], float("-inf"))

    p_validation = pd.to_numeric(validation_accepted["p_side"], errors="coerce").to_numpy(dtype=float)
    baselines: dict[str, dict[str, float]] = {}
    for name, bid in {
        "fixed_0p50_pside": floor_to_tick(0.50 * p_validation, tick),
        "fixed_0p75_pside": floor_to_tick(0.75 * p_validation, tick),
        "pay_pside": floor_to_tick(p_validation, tick),
    }.items():
        bid = np.maximum(bid, min_bid)
        baselines[name] = backtest_metrics(validation_accepted, backtest_with_bid(validation_accepted, bid), len(validation))
    baselines["expected_return"] = backtest_metrics(validation_accepted, validation_result, len(validation))

    train_point = point_predictions.get("train")
    cal_point = point_predictions.get("calibration")
    validation_point = point_predictions.get("validation")
    write_predictions(train_accepted, train_result, resolve_path(config["paths"]["predictions_train"]), q_train[train_mask], None if train_point is None else train_point[train_mask])
    write_predictions(calibration_accepted, cal_result, resolve_path(config["paths"]["predictions_calibration"]), q_cal[cal_mask], None if cal_point is None else cal_point[cal_mask])
    write_predictions(validation_accepted, validation_result, resolve_path(config["paths"]["predictions_validation"]), q_validation[validation_mask], None if validation_point is None else validation_point[validation_mask])
    pd.DataFrame(epoch_rows).to_csv(reports_dir / "epoch_metrics.csv", index=False)
    reliability_path = reports_dir / "reliability_validation.csv"
    reliability_table(
        validation_correct,
        hazard_validation[validation_correct_mask.to_numpy()],
        gc_validation[validation_correct_mask.to_numpy()],
        tick_grid,
    ).to_csv(reliability_path, index=False)
    frontier_path = reports_dir / "validation_frontier.csv"
    pd.DataFrame(validation_frontier).to_csv(frontier_path, index=False)

    checkpoint: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model": config["model"],
        "preprocessor": preprocessor.to_dict(),
        "feature_columns": columns,
        "categorical_columns": cat_cols,
        "tick_grid": tick_grid.tolist(),
        "target": config["target"],
        "selected_min_ev": selected_min_ev,
        "q_calibrator": None if q_calibrator is None else {
            "x_thresholds": q_calibrator.X_thresholds_.tolist(),
            "y_thresholds": q_calibrator.y_thresholds_.tolist(),
        },
    }
    if residuals is not None:
        checkpoint["residual_cdf"] = {"source": "calibration_correct", "residuals": residuals.tolist()}
    checkpoint_path = models_dir / ("hazard_survival_cdf.pt" if family == "hazard_survival_cdf" else "low_cdf_point_model.pt")
    torch.save(checkpoint, checkpoint_path)

    direction_coverage = float(target_summary.get("validation_summary", {}).get("accepted_coverage_vs_source", validation_mask.mean()))
    report: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_training": git_commit(),
        "git_commit": git_commit(),
        "config_path": str(reports_dir / "config_used.yaml"),
        "report_path": str(reports_dir / "summary_metrics.json"),
        "primary_metric": "validation forced mean_accepted_pnl for expected_return policy",
        "model_family": family,
        "q_calibration": "isotonic_calibration_split" if q_calibrator is not None else "none",
        "deploy_experiment_id": manifest.get("experiment_id"),
        "deploy_training_mode": manifest.get("training_mode"),
        "offline_validation_metric_source": manifest.get("source_report_path"),
        "order_window": config["target"]["order_window"],
        "selected_min_ev": selected_min_ev,
        "min_ev_selection_source": "calibration",
        "validation_selection_note": "Validation did not participate in model, q, or min_ev selection; frontier is diagnostic only.",
        "min_ev_search": min_ev_search,
        "validation_frontier": validation_frontier,
        "o0_forced_no_abstain_validation_metrics": backtest_metrics(validation_accepted, o0_forced, len(validation)),
        "train_metrics": backtest_metrics(train_accepted, train_result, len(train_all)),
        "calibration_metrics": backtest_metrics(calibration_accepted, cal_result, len(calibration_all)),
        "validation_metrics": backtest_metrics(validation_accepted, validation_result, len(validation)),
        "validation_baselines": baselines,
        "signal_coverage": direction_coverage,
        "coverage_constraint_satisfied": bool(direction_coverage >= float(config.get("objective", {}).get("min_coverage", 0.70))),
        "coverage_note": "Direction accepted-universe coverage is unchanged; expected-return order coverage is reported separately.",
        "target_build_summary_path": str(source_summary_path),
        "chosen_low_missing_policy": "excluded",
        "gc_monotonicity_check": True,
        "windows": {
            name: {
                "row_count": int(len(frame)),
                "start": str(pd.to_datetime(frame["timestamp"], utc=True).min()),
                "end": str(pd.to_datetime(frame["timestamp"], utc=True).max()),
            }
            for name, frame in [("fit", fit_all), ("calibration", calibration_all), ("validation", validation)]
        },
        "train_window": {"row_count": int(len(train_accepted)), "start": str(pd.to_datetime(train_accepted["timestamp"], utc=True).min()), "end": str(pd.to_datetime(train_accepted["timestamp"], utc=True).max())},
        "validation_window": {"row_count": int(len(validation_accepted)), "start": str(pd.to_datetime(validation_accepted["timestamp"], utc=True).min()), "end": str(pd.to_datetime(validation_accepted["timestamp"], utc=True).max())},
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "config_snapshot": str(reports_dir / "config_used.yaml"),
            "epoch_metrics": str(reports_dir / "epoch_metrics.csv"),
            "reliability_validation": str(reliability_path),
            "validation_frontier": str(frontier_path),
            "train_predictions": str(resolve_path(config["paths"]["predictions_train"])),
            "calibration_predictions": str(resolve_path(config["paths"]["predictions_calibration"])),
            "validation_predictions": str(resolve_path(config["paths"]["predictions_validation"])),
        },
        "feature_count_raw": len(columns),
        "feature_count_encoded": int(x_fit.shape[1]),
        "feature_columns": columns,
        "categorical_columns": cat_cols,
    }
    if residuals is not None:
        report["point_model_metrics"] = {
            "fit_correct": point_metrics(y_fit, point_predictions["fit"]),
            "calibration_correct": point_metrics(y_cal_correct, point_predictions["calibration_correct"]),
            "validation_correct": point_metrics(
                pd.to_numeric(validation_correct["chosen_low"], errors="coerce").to_numpy(dtype=float),
                point_predictions["validation"][validation_correct_mask.to_numpy()],
            ),
        }
    write_json(reports_dir / "summary_metrics.json", report)
    print({"validation_metrics": report["validation_metrics"], "validation_baselines": baselines})


if __name__ == "__main__":
    main()
