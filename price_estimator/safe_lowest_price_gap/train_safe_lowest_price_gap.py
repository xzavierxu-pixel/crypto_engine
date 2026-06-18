#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_ESTIMATOR_DIR = SCRIPT_DIR.parent
ROOT = PRICE_ESTIMATOR_DIR.parent
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "scripts"))
sys.path.insert(0, str(PRICE_ESTIMATOR_DIR / "upper_bound_mlp"))

from price_estimator_common import (  # noqa: E402
    apply_sample_filter,
    load_config,
    load_deploy_manifest,
    resolve_path,
    write_json,
)
from train_upper_bound_mlp import (  # noqa: E402
    PRED_BASE_COLS,
    Preprocessor,
    UpperBoundMLP,
    feature_set,
)


@dataclass(frozen=True)
class PredictionResult:
    p_pred: np.ndarray
    action: np.ndarray
    conf_ok: np.ndarray


@dataclass(frozen=True)
class CandidateResult:
    alpha: float
    delta_norm: float
    delta_quantile: float
    bucket_miss_threshold: float
    bucket_model: dict[str, Any]
    metrics: dict[str, float]
    prediction: PredictionResult


class SafeGapDataset(Dataset):
    def __init__(self, x: np.ndarray, y_safe: np.ndarray, s_eff: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y_safe = torch.from_numpy(y_safe.astype(np.float32)).view(-1, 1)
        self.s_eff = torch.from_numpy(s_eff.astype(np.float32)).view(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y_safe[idx], self.s_eff[idx]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_targets(df: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_col = str(config["target"]["raw_column"])
    buffer = float(config["target"]["buffer"])
    s_floor = float(config["loss"]["s_floor"])
    y_safe = pd.to_numeric(df[raw_col], errors="coerce").to_numpy(dtype=float) + buffer
    p_side = pd.to_numeric(df["p_side"], errors="coerce").to_numpy(dtype=float)
    s = p_side - y_safe
    s_eff = np.maximum(s, s_floor)
    return y_safe.astype(np.float32), s.astype(np.float32), s_eff.astype(np.float32)


def split_fit_calibration(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_col = str(config["split"].get("timestamp_column", "timestamp"))
    days = int(config["split"].get("calibration_tail_days", 31))
    if ts_col not in df.columns:
        raise ValueError(f"Calibration split requires timestamp column {ts_col!r}")
    ordered = df.sort_values(ts_col).reset_index(drop=True)
    ts = pd.to_datetime(ordered[ts_col], utc=True)
    cutoff = ts.max() - pd.Timedelta(days=days)
    fit = ordered.loc[ts < cutoff].copy()
    calibration = ordered.loc[ts >= cutoff].copy()
    if fit.empty or calibration.empty:
        raise ValueError("Calibration tail split produced an empty fit or calibration set")
    return fit, calibration


def normalized_asym_loss(
    z: torch.Tensor,
    y_safe: torch.Tensor,
    s_eff: torch.Tensor,
    alpha: float,
    c: float,
    kappa: float,
) -> torch.Tensor:
    f = torch.sigmoid(z)
    r = f - y_safe
    s = torch.clamp_min(s_eff, 1e-6)
    rn = r / s
    under = float(alpha) * (torch.sqrt(rn * rn + float(c) * float(c)) - float(c))
    t = torch.clamp(rn, min=0.0)
    over_quad = 0.5 * t * t
    over_lin = float(kappa) * (t - 0.5 * float(kappa))
    over = torch.where(t <= float(kappa), over_quad, over_lin)
    return torch.where(r < 0.0, under, over).mean()


@torch.no_grad()
def predict_f(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
        z = model(xb).detach().cpu().numpy().reshape(-1)
        out.append(sigmoid_np(z))
    return np.concatenate(out) if out else np.array([], dtype=float)


def ceil_to_tick(raw: np.ndarray, tick_size: float, tol: float) -> np.ndarray:
    return np.ceil(raw / tick_size - tol) * tick_size


def infer_prices(
    f: np.ndarray,
    p_side: np.ndarray,
    conf_ok: np.ndarray,
    delta_norm: float,
    tick_size: float,
    tick_tol: float,
    s_floor: float,
) -> PredictionResult:
    f = np.asarray(f, dtype=float)
    p_side = np.asarray(p_side, dtype=float)
    conf_ok = np.asarray(conf_ok, dtype=bool)
    s_proxy = np.clip(p_side - f, float(s_floor), None)
    raw = f + float(delta_norm) * s_proxy
    ticked = ceil_to_tick(raw, tick_size, tick_tol)
    ticked = np.maximum(ticked, 0.0)
    p_pred = np.minimum(ticked, p_side)
    action = np.full(len(f), "active", dtype=object)
    clamp = conf_ok & (raw >= p_side)
    action[clamp] = "clamp_over_pside"
    action[~conf_ok] = "abstain_low_conf"
    p_pred[~conf_ok] = p_side[~conf_ok]
    p_pred[clamp] = p_side[clamp]
    return PredictionResult(p_pred=p_pred, action=action.astype(str), conf_ok=conf_ok)


def metric_summary(
    y_safe: np.ndarray,
    p_side: np.ndarray,
    prediction: PredictionResult,
    tolerance: float,
) -> dict[str, float]:
    y = np.asarray(y_safe, dtype=float)
    p = np.asarray(p_side, dtype=float)
    pred = np.asarray(prediction.p_pred, dtype=float)
    feasible = y < p
    covered = (pred + tolerance >= y) & (pred <= p + tolerance)
    covered_feasible = covered & feasible
    denom = np.maximum(p - y, 1e-12)
    gap_norm = (pred - y) / denom
    covered_gap = gap_norm[covered_feasible]

    def q(values: np.ndarray, quantile: float) -> float:
        return float(np.quantile(values, quantile)) if len(values) else float("nan")

    active = prediction.action == "active"
    abstain = prediction.action == "abstain_low_conf"
    clamp = prediction.action == "clamp_over_pside"
    active_cov = covered_feasible & active
    active_gap = gap_norm[active_cov]
    metrics = {
        "sample_count": float(len(y)),
        "feasible_count": float(feasible.sum()),
        "infeasible_count": float((~feasible).sum()),
        "max_possible_coverage": float(feasible.mean()) if len(y) else float("nan"),
        "coverage_overall": float(covered.mean()) if len(y) else float("nan"),
        "coverage_feasible": float(covered[feasible].mean()) if feasible.any() else float("nan"),
        "covered_count": float(covered.sum()),
        "covered_feasible_count": float(covered_feasible.sum()),
        "covered_gap_norm_mean": float(np.mean(covered_gap)) if len(covered_gap) else float("nan"),
        "covered_gap_norm_median": float(np.median(covered_gap)) if len(covered_gap) else float("nan"),
        "covered_gap_norm_q25": q(covered_gap, 0.25),
        "covered_gap_norm_q75": q(covered_gap, 0.75),
        "active_covered_gap_norm_mean": float(np.mean(active_gap)) if len(active_gap) else float("nan"),
        "active_covered_gap_norm_median": float(np.median(active_gap)) if len(active_gap) else float("nan"),
        "active_count": float(active.sum()),
        "active_share": float(active.mean()) if len(active) else float("nan"),
        "abstain_low_conf_count": float(abstain.sum()),
        "abstain_low_conf_share": float(abstain.mean()) if len(abstain) else float("nan"),
        "clamp_over_pside_count": float(clamp.sum()),
        "clamp_over_pside_share": float(clamp.mean()) if len(clamp) else float("nan"),
        "non_active_share": float((~active).mean()) if len(active) else float("nan"),
        "side_violation_count": float((pred > p + tolerance).sum()),
        "side_violation_rate": float((pred > p + tolerance).mean()) if len(pred) else float("nan"),
        "mean_p_pred": float(np.mean(pred)) if len(pred) else float("nan"),
        "mean_p_side": float(np.mean(p)) if len(p) else float("nan"),
    }
    feasible_gap_with_fallback = np.where(covered_feasible, np.clip(gap_norm, 0.0, 1.0), np.nan)
    if np.isfinite(feasible_gap_with_fallback).any():
        metrics["feasible_covered_gap_norm_mean"] = float(np.nanmean(feasible_gap_with_fallback))
    else:
        metrics["feasible_covered_gap_norm_mean"] = float("nan")
    return metrics


def action_diagnostics(
    y_safe: np.ndarray,
    p_side: np.ndarray,
    prediction: PredictionResult,
    tolerance: float,
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    actions = ["active", "abstain_low_conf", "clamp_over_pside"]
    for action in actions:
        mask = prediction.action == action
        if not mask.any():
            rows[action] = {"sample_count": 0.0}
            continue
        rows[action] = metric_summary(y_safe[mask], p_side[mask], PredictionResult(
            p_pred=prediction.p_pred[mask],
            action=prediction.action[mask],
            conf_ok=prediction.conf_ok[mask],
        ), tolerance)
    return rows


def bucket_frame(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    keys = list(config.get("bucket_abstain", {}).get("keys", []))
    out = pd.DataFrame(index=df.index)
    for key in keys:
        if key in df.columns:
            out[key] = df[key].astype("string").fillna("missing").astype(str)
        else:
            out[key] = "missing"
    if not keys:
        out["_global"] = "global"
    return out.reset_index(drop=True)


def bucket_labels(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    frame = bucket_frame(df, config)
    return frame.astype(str).agg("|".join, axis=1)


def fit_bucket_model(
    calibration: pd.DataFrame,
    y_safe: np.ndarray,
    p_side: np.ndarray,
    f_cal: np.ndarray,
    delta_norm: float,
    config: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    tick_size = float(config["target"]["tick_size"])
    tick_tol = float(config["target"]["tick_rounding_tolerance"])
    s_floor = float(config["loss"]["s_floor"])
    base_pred = infer_prices(
        f_cal,
        p_side,
        np.ones(len(f_cal), dtype=bool),
        delta_norm,
        tick_size,
        tick_tol,
        s_floor,
    )
    feasible = y_safe < p_side
    covered = (base_pred.p_pred + tolerance >= y_safe) & (base_pred.p_pred <= p_side + tolerance)
    labels = bucket_labels(calibration, config)
    work = pd.DataFrame({"bucket": labels, "feasible": feasible, "miss": feasible & (~covered)})
    min_count = int(config.get("bucket_abstain", {}).get("min_bucket_count", 30))
    table: dict[str, dict[str, float]] = {}
    feasible_global = work.loc[work["feasible"]]
    global_miss_rate = float(feasible_global["miss"].mean()) if len(feasible_global) else 1.0
    for bucket, part in work.groupby("bucket", dropna=False):
        feasible_part = part.loc[part["feasible"]]
        count = int(len(feasible_part))
        if count >= min_count:
            miss_rate = float(feasible_part["miss"].mean())
        else:
            miss_rate = global_miss_rate
        table[str(bucket)] = {
            "feasible_count": float(count),
            "sample_count": float(len(part)),
            "miss_rate": miss_rate,
        }
    return {
        "enabled": bool(config.get("bucket_abstain", {}).get("enabled", True)),
        "keys": list(config.get("bucket_abstain", {}).get("keys", [])),
        "min_bucket_count": min_count,
        "global_miss_rate": global_miss_rate,
        "table": table,
    }


def bucket_conf_ok(df: pd.DataFrame, bucket_model: dict[str, Any], threshold: float) -> np.ndarray:
    if not bool(bucket_model.get("enabled", True)):
        return np.ones(len(df), dtype=bool)
    labels = bucket_labels(df, {"bucket_abstain": {"keys": bucket_model.get("keys", [])}})
    table = bucket_model.get("table", {})
    global_miss_rate = float(bucket_model.get("global_miss_rate", 1.0))
    miss_rate = labels.map(lambda v: float(table.get(str(v), {}).get("miss_rate", global_miss_rate))).to_numpy()
    return miss_rate <= float(threshold)


def candidate_key(
    candidate: CandidateResult,
    min_coverage: float,
    max_non_active_share: float,
    optimize_metric: str,
    tie_breaker_metric: str,
) -> tuple[float, float, float, float, float]:
    m = candidate.metrics
    primary = float(m.get(optimize_metric, float("nan")))
    tie_breaker = float(m.get(tie_breaker_metric, float("nan")))
    valid = (
        m["coverage_feasible"] >= min_coverage
        and m["side_violation_rate"] == 0.0
        and m["non_active_share"] <= max_non_active_share
        and math.isfinite(primary)
    )
    if not valid:
        return (1.0, -m["coverage_feasible"], float("inf"), float("inf"), float("inf"))
    return (
        0.0,
        primary,
        tie_breaker if math.isfinite(tie_breaker) else float("inf"),
        m["non_active_share"],
        -m["covered_feasible_count"],
    )


def select_calibration_candidate(
    calibration: pd.DataFrame,
    y_safe: np.ndarray,
    p_side: np.ndarray,
    f_cal: np.ndarray,
    alpha: float,
    config: dict[str, Any],
) -> tuple[CandidateResult, list[dict[str, float]]]:
    feasible = y_safe < p_side
    s_floor = float(config["loss"]["s_floor"])
    s_proxy = np.clip(p_side[feasible] - f_cal[feasible], s_floor, None)
    r_norm = (y_safe[feasible] - f_cal[feasible]) / s_proxy
    if len(r_norm) == 0:
        raise ValueError("No feasible calibration rows available")
    tolerance = float(config["target"]["feasibility_tolerance"])
    tick_size = float(config["target"]["tick_size"])
    tick_tol = float(config["target"]["tick_rounding_tolerance"])
    min_coverage = float(
        config["objective"].get(
            "min_calibration_feasible_coverage",
            config["objective"]["min_feasible_coverage"],
        )
    )
    min_coverage += float(config["objective"].get("calibration_coverage_buffer", 0.0))
    max_non_active_share = float(config["objective"].get("max_non_active_share", 1.0))
    max_non_active_share = float(config["objective"].get("max_non_active_share", 1.0))
    frontier: list[dict[str, float]] = []
    candidates: list[CandidateResult] = []
    for q in [float(v) for v in config["calibration"]["delta_quantiles"]]:
        delta_norm = float(np.quantile(r_norm, q))
        bucket_model = fit_bucket_model(calibration, y_safe, p_side, f_cal, delta_norm, config, tolerance)
        for threshold in [float(v) for v in config["calibration"]["bucket_miss_rate_thresholds"]]:
            conf_ok = bucket_conf_ok(calibration, bucket_model, threshold)
            pred = infer_prices(f_cal, p_side, conf_ok, delta_norm, tick_size, tick_tol, s_floor)
            metrics = metric_summary(y_safe, p_side, pred, tolerance)
            row = {
                "alpha": float(alpha),
                "delta_norm": delta_norm,
                "delta_quantile": q,
                "bucket_miss_threshold": threshold,
                **metrics,
            }
            frontier.append(row)
            candidates.append(
                CandidateResult(
                    alpha=float(alpha),
                    delta_norm=delta_norm,
                    delta_quantile=q,
                    bucket_miss_threshold=threshold,
                    bucket_model=bucket_model,
                    metrics=metrics,
                    prediction=pred,
                )
            )
    optimize_metric = str(config["objective"].get("optimize_metric", "active_covered_gap_norm_mean"))
    tie_breaker_metric = str(config["objective"].get("tie_breaker_metric", "covered_gap_norm_mean"))
    best = min(
        candidates,
        key=lambda c: candidate_key(c, min_coverage, max_non_active_share, optimize_metric, tie_breaker_metric),
    )
    return best, frontier


def train_one_alpha(
    alpha: float,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    s_eff_fit: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    calibration: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, CandidateResult, list[dict[str, Any]]]:
    seed = int(config["training"]["random_seed"]) + int(alpha * 10)
    set_seed(seed)
    model = UpperBoundMLP(
        input_dim=x_fit.shape[1],
        hidden_dims=[int(v) for v in config["model"]["hidden_dims"]],
        dropout=[float(v) for v in config["model"]["dropout"]],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    dataset = SafeGapDataset(x_fit, y_fit, s_eff_fit)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
        generator=generator,
    )
    c = float(config["loss"]["c"])
    kappa = float(config["loss"]["kappa"])
    grad_clip = float(config["training"].get("gradient_clip_norm", 5.0))
    patience = int(config["training"].get("early_stop_patience", 20))
    epochs = int(config["training"]["epochs"])
    best_state: dict[str, torch.Tensor] | None = None
    best_candidate: CandidateResult | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    stale = 0
    rows: list[dict[str, Any]] = []
    min_coverage = float(
        config["objective"].get(
            "min_calibration_feasible_coverage",
            config["objective"]["min_feasible_coverage"],
        )
    )
    min_coverage += float(config["objective"].get("calibration_coverage_buffer", 0.0))
    max_non_active_share = float(config["objective"].get("max_non_active_share", 1.0))
    optimize_metric = str(config["objective"].get("optimize_metric", "active_covered_gap_norm_mean"))
    tie_breaker_metric = str(config["objective"].get("tie_breaker_metric", "covered_gap_norm_mean"))
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb, s_eff_b in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            s_eff_b = s_eff_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model(xb)
            loss = normalized_asym_loss(z, yb, s_eff_b, alpha=alpha, c=c, kappa=kappa)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        f_cal = predict_f(model, x_cal, device, int(config["training"]["batch_size"]))
        candidate, _ = select_calibration_candidate(calibration, y_cal, p_cal, f_cal, alpha, config)
        key = candidate_key(candidate, min_coverage, max_non_active_share, optimize_metric, tie_breaker_metric)
        row = {
            "alpha": float(alpha),
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "selected_delta_norm": candidate.delta_norm,
            "selected_delta_quantile": candidate.delta_quantile,
            "selected_bucket_miss_threshold": candidate.bucket_miss_threshold,
            **{f"calibration_{k}": v for k, v in candidate.metrics.items()},
        }
        rows.append(row)
        if best_key is None or key < best_key:
            best_key = key
            best_candidate = candidate
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({
                "alpha": alpha,
                "epoch": epoch,
                "loss": row["loss"],
                "calibration_coverage_feasible": candidate.metrics["coverage_feasible"],
                "calibration_gap_mean": candidate.metrics["covered_gap_norm_mean"],
                "calibration_active_gap_mean": candidate.metrics["active_covered_gap_norm_mean"],
                "calibration_abstain_share": candidate.metrics["abstain_low_conf_share"],
            }, sort_keys=True))
        if stale >= patience:
            break
    if best_state is None or best_candidate is None:
        raise RuntimeError(f"Training failed for alpha={alpha}")
    model.load_state_dict(best_state)
    return model, best_candidate, rows


def evaluate_with_candidate(
    df: pd.DataFrame,
    y_safe: np.ndarray,
    p_side: np.ndarray,
    f: np.ndarray,
    candidate: CandidateResult,
    config: dict[str, Any],
) -> tuple[PredictionResult, dict[str, float], dict[str, dict[str, float]]]:
    conf_ok = bucket_conf_ok(df, candidate.bucket_model, candidate.bucket_miss_threshold)
    pred = infer_prices(
        f,
        p_side,
        conf_ok,
        candidate.delta_norm,
        float(config["target"]["tick_size"]),
        float(config["target"]["tick_rounding_tolerance"]),
        float(config["loss"]["s_floor"]),
    )
    metrics = metric_summary(y_safe, p_side, pred, float(config["target"]["feasibility_tolerance"]))
    diagnostics = action_diagnostics(y_safe, p_side, pred, float(config["target"]["feasibility_tolerance"]))
    return pred, metrics, diagnostics


def _dist_stats(vals: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"count": 0.0}
    out = {
        "count": float(len(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }
    for quantile in quantiles:
        out[f"q{int(round(float(quantile) * 100)):02d}"] = float(np.quantile(vals, float(quantile)))
    out["spread_q90_q50"] = out.get("q90", float("nan")) - out.get("q50", float("nan"))
    return out


def normalized_residual_diagnostics(
    y_safe: np.ndarray,
    p_side: np.ndarray,
    f: np.ndarray,
    s_floor: float,
    quantiles: list[float],
) -> dict[str, Any]:
    y = np.asarray(y_safe, dtype=float)
    p = np.asarray(p_side, dtype=float)
    ff = np.asarray(f, dtype=float)
    feasible = y < p
    yf = y[feasible]
    pf = p[feasible]
    ff_feasible = ff[feasible]
    s_true = np.clip(pf - yf, float(s_floor), None)
    s_proxy = np.clip(pf - ff_feasible, float(s_floor), None)
    residual = yf - ff_feasible
    return {
        "feasible_count": float(feasible.sum()),
        "abs_residual": _dist_stats(residual, quantiles),
        "norm_residual_true_s": _dist_stats(residual / s_true, quantiles),
        "norm_residual_proxy_s": _dist_stats(residual / s_proxy, quantiles),
        "room_s_true": _dist_stats(s_true, quantiles),
        "room_s_proxy": _dist_stats(s_proxy, quantiles),
    }


def pside_bin_edges(config: dict[str, Any]) -> list[float]:
    diagnostics = config.get("diagnostics", {})
    if "pside_bin_step" in diagnostics:
        step = float(diagnostics.get("pside_bin_step", 0.05))
        if step <= 0 or step > 1:
            raise ValueError("diagnostics.pside_bin_step must be in (0, 1]")
        n = int(round(1.0 / step))
        return [round(i * step, 10) for i in range(n + 1)]
    return [float(v) for v in diagnostics.get("p_side_bin_edges", [0.0, 1.0])]


def per_pside_bin_delta_norm(
    y_safe: np.ndarray,
    p_side: np.ndarray,
    f: np.ndarray,
    edges: list[float],
    s_floor: float,
    delta_quantile: float,
) -> list[dict[str, float]]:
    y = np.asarray(y_safe, dtype=float)
    p = np.asarray(p_side, dtype=float)
    ff = np.asarray(f, dtype=float)
    feasible = y < p
    s_proxy = np.clip(p - ff, float(s_floor), None)
    r_norm = (y - ff) / s_proxy
    idx = np.digitize(p, edges, right=False)
    rows: list[dict[str, float]] = []
    for bucket_idx in range(1, len(edges)):
        mask = feasible & (idx == bucket_idx)
        row = {
            "pside_bin": f"[{edges[bucket_idx - 1]:.2f}, {edges[bucket_idx]:.2f})",
            "edge_lo": float(edges[bucket_idx - 1]),
            "edge_hi": float(edges[bucket_idx]),
            "feasible_count": float(mask.sum()),
        }
        if mask.any():
            row.update({
                "norm_residual_q50": float(np.quantile(r_norm[mask], 0.50)),
                "norm_residual_q90": float(np.quantile(r_norm[mask], 0.90)),
                "local_delta_norm": float(np.quantile(r_norm[mask], float(delta_quantile))),
                "mean_room_s_true": float(np.mean(np.clip(p[mask] - y[mask], float(s_floor), None))),
                "mean_room_s_proxy": float(np.mean(s_proxy[mask])),
            })
        rows.append(row)
    return rows


def pside_bin_metrics(
    y_safe: np.ndarray,
    p_side: np.ndarray,
    prediction: PredictionResult,
    edges: list[float],
    tolerance: float,
) -> list[dict[str, float]]:
    p = np.asarray(p_side, dtype=float)
    idx = np.digitize(p, edges, right=False)
    rows: list[dict[str, float]] = []
    for bucket_idx in range(1, len(edges)):
        mask = idx == bucket_idx
        row = {
            "pside_bin": f"[{edges[bucket_idx - 1]:.2f}, {edges[bucket_idx]:.2f})",
            "edge_lo": float(edges[bucket_idx - 1]),
            "edge_hi": float(edges[bucket_idx]),
            "sample_count": float(mask.sum()),
        }
        if mask.any():
            sub = PredictionResult(
                p_pred=prediction.p_pred[mask],
                action=prediction.action[mask],
                conf_ok=prediction.conf_ok[mask],
            )
            row.update(metric_summary(y_safe[mask], p[mask], sub, tolerance))
        rows.append(row)
    return rows


def write_predictions(
    df: pd.DataFrame,
    y_safe: np.ndarray,
    s: np.ndarray,
    f: np.ndarray,
    prediction: PredictionResult,
    path: Path,
) -> None:
    pred = df[[c for c in PRED_BASE_COLS if c in df.columns]].copy()
    pred.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    pred["target_safe"] = y_safe
    pred["safe_gap_room"] = s
    pred["feasible"] = s >= 0.0
    pred["f_model"] = f
    pred["p_pred"] = prediction.p_pred
    pred["action"] = prediction.action
    pred["conf_ok"] = prediction.conf_ok
    denom = np.maximum(pd.to_numeric(pred["p_side"], errors="coerce").to_numpy(dtype=float) - y_safe, 1e-12)
    pred["gap_norm"] = (prediction.p_pred - y_safe) / denom
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(path, index=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_numpy_artifact(
    path: Path,
    model: nn.Module,
    metadata: dict[str, Any],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in model.state_dict().items():
        arrays[name.replace(".", "__")] = tensor.detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, metadata=json.dumps(metadata, sort_keys=True), **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="price_estimator/safe_lowest_price_gap/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    seed = int(config["training"]["random_seed"])
    set_seed(seed)
    device_name = str(config["training"].get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if device.type == "auto":
        device = torch.device("cpu")

    train_all_raw = pd.read_parquet(resolve_path(config["paths"]["train_dataset"]))
    validation_raw = pd.read_parquet(resolve_path(config["paths"]["validation_dataset"]))
    manifest = load_deploy_manifest(config)
    train_all, train_filter = apply_sample_filter(train_all_raw, config, manifest)
    validation, validation_filter = apply_sample_filter(validation_raw, config, manifest)
    fit, calibration = split_fit_calibration(train_all, config)

    columns, cat_cols = feature_set(config, fit)
    preprocessor = Preprocessor.fit(fit, columns, cat_cols)
    x_fit = preprocessor.transform(fit)
    x_cal = preprocessor.transform(calibration)
    x_train_all = preprocessor.transform(train_all)
    x_validation = preprocessor.transform(validation)

    y_fit, s_fit, s_eff_fit = safe_targets(fit, config)
    y_cal, s_cal, _ = safe_targets(calibration, config)
    y_train_all, s_train_all, _ = safe_targets(train_all, config)
    y_validation, s_validation, _ = safe_targets(validation, config)
    p_cal = pd.to_numeric(calibration["p_side"], errors="coerce").to_numpy(dtype=float)
    p_train_all = pd.to_numeric(train_all["p_side"], errors="coerce").to_numpy(dtype=float)
    p_validation = pd.to_numeric(validation["p_side"], errors="coerce").to_numpy(dtype=float)

    reports_dir = resolve_path(config["paths"]["reports_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_path(args.config), reports_dir / "config_used.yaml")

    epoch_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    trained: list[tuple[nn.Module, CandidateResult]] = []
    min_coverage = float(
        config["objective"].get(
            "min_calibration_feasible_coverage",
            config["objective"]["min_feasible_coverage"],
        )
    )
    min_coverage += float(config["objective"].get("calibration_coverage_buffer", 0.0))
    max_non_active_share = float(config["objective"].get("max_non_active_share", 1.0))
    optimize_metric = str(config["objective"].get("optimize_metric", "active_covered_gap_norm_mean"))
    tie_breaker_metric = str(config["objective"].get("tie_breaker_metric", "covered_gap_norm_mean"))
    for alpha in [float(v) for v in config["loss"]["alpha_grid"]]:
        model, candidate, rows = train_one_alpha(
            alpha,
            x_fit,
            y_fit,
            s_eff_fit,
            x_cal,
            y_cal,
            p_cal,
            calibration,
            config,
            device,
        )
        f_cal = predict_f(model, x_cal, device, int(config["training"]["batch_size"]))
        selected, frontier = select_calibration_candidate(calibration, y_cal, p_cal, f_cal, alpha, config)
        epoch_rows.extend(rows)
        frontier_rows.extend(frontier)
        trained.append((model, selected))

    selected_model, selected_candidate = min(
        trained,
        key=lambda item: candidate_key(
            item[1],
            min_coverage,
            max_non_active_share,
            optimize_metric,
            tie_breaker_metric,
        ),
    )
    f_fit = predict_f(selected_model, x_fit, device, int(config["training"]["batch_size"]))
    f_cal = predict_f(selected_model, x_cal, device, int(config["training"]["batch_size"]))
    f_train_all = predict_f(selected_model, x_train_all, device, int(config["training"]["batch_size"]))
    f_validation = predict_f(selected_model, x_validation, device, int(config["training"]["batch_size"]))

    fit_pred, fit_metrics, fit_action_diag = evaluate_with_candidate(
        fit, y_fit, pd.to_numeric(fit["p_side"], errors="coerce").to_numpy(dtype=float), f_fit, selected_candidate, config
    )
    cal_pred, cal_metrics, cal_action_diag = evaluate_with_candidate(
        calibration, y_cal, p_cal, f_cal, selected_candidate, config
    )
    train_pred, train_metrics, train_action_diag = evaluate_with_candidate(
        train_all, y_train_all, p_train_all, f_train_all, selected_candidate, config
    )
    validation_pred, validation_metrics, validation_action_diag = evaluate_with_candidate(
        validation, y_validation, p_validation, f_validation, selected_candidate, config
    )
    tolerance = float(config["target"]["feasibility_tolerance"])
    s_floor = float(config["loss"]["s_floor"])
    diag_quantiles = [
        float(v)
        for v in config.get("diagnostics", {}).get(
            "normalized_residual_quantiles",
            [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        )
    ]
    edges = pside_bin_edges(config)
    normalized_residual_report = {
        "fit": normalized_residual_diagnostics(
            y_fit,
            pd.to_numeric(fit["p_side"], errors="coerce").to_numpy(dtype=float),
            f_fit,
            s_floor,
            diag_quantiles,
        ),
        "calibration": normalized_residual_diagnostics(y_cal, p_cal, f_cal, s_floor, diag_quantiles),
        "train": normalized_residual_diagnostics(y_train_all, p_train_all, f_train_all, s_floor, diag_quantiles),
        "validation": normalized_residual_diagnostics(y_validation, p_validation, f_validation, s_floor, diag_quantiles),
    }
    per_pside_delta_norm_report = {
        "fit": per_pside_bin_delta_norm(
            y_fit,
            pd.to_numeric(fit["p_side"], errors="coerce").to_numpy(dtype=float),
            f_fit,
            edges,
            s_floor,
            selected_candidate.delta_quantile,
        ),
        "calibration": per_pside_bin_delta_norm(
            y_cal,
            p_cal,
            f_cal,
            edges,
            s_floor,
            selected_candidate.delta_quantile,
        ),
        "train": per_pside_bin_delta_norm(
            y_train_all,
            p_train_all,
            f_train_all,
            edges,
            s_floor,
            selected_candidate.delta_quantile,
        ),
        "validation": per_pside_bin_delta_norm(
            y_validation,
            p_validation,
            f_validation,
            edges,
            s_floor,
            selected_candidate.delta_quantile,
        ),
    }
    pside_bin_metric_report = {
        "fit": pside_bin_metrics(
            y_fit,
            pd.to_numeric(fit["p_side"], errors="coerce").to_numpy(dtype=float),
            fit_pred,
            edges,
            tolerance,
        ),
        "calibration": pside_bin_metrics(y_cal, p_cal, cal_pred, edges, tolerance),
        "train": pside_bin_metrics(y_train_all, p_train_all, train_pred, edges, tolerance),
        "validation": pside_bin_metrics(y_validation, p_validation, validation_pred, edges, tolerance),
    }

    model_metadata = {
        "input_dim": x_fit.shape[1],
        "model": config["model"],
        "preprocessor": preprocessor.to_dict(),
        "feature_columns": columns,
        "categorical_columns": cat_cols,
        "target": config["target"],
        "loss": config["loss"],
        "calibration": {
            "alpha": selected_candidate.alpha,
            "delta_norm": selected_candidate.delta_norm,
            "delta_quantile": selected_candidate.delta_quantile,
            "bucket_miss_threshold": selected_candidate.bucket_miss_threshold,
            "bucket_model": selected_candidate.bucket_model,
        },
    }
    checkpoint_path = models_dir / "safe_lowest_price_gap.pt"
    numpy_artifact_path = models_dir / "safe_lowest_price_gap.npz"
    torch.save(
        {
            "state_dict": selected_model.state_dict(),
            **model_metadata,
        },
        checkpoint_path,
    )
    write_numpy_artifact(numpy_artifact_path, selected_model, model_metadata)

    train_pred_path = resolve_path(config["paths"]["predictions_train"])
    cal_pred_path = resolve_path(config["paths"]["predictions_calibration"])
    validation_pred_path = resolve_path(config["paths"]["predictions_validation"])
    write_predictions(train_all, y_train_all, s_train_all, f_train_all, train_pred, train_pred_path)
    write_predictions(calibration, y_cal, s_cal, f_cal, cal_pred, cal_pred_path)
    write_predictions(validation, y_validation, s_validation, f_validation, validation_pred, validation_pred_path)
    write_csv(reports_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(reports_dir / "calibration_frontier.csv", frontier_rows)

    validation_min_coverage = float(config["objective"]["min_feasible_coverage"])
    coverage_constraint_satisfied = bool(validation_metrics["coverage_feasible"] >= validation_min_coverage)
    report = {
        "experiment_id": config["experiment_id"],
        "git_commit_at_training": git_commit(),
        "config_path": args.config,
        "primary_metric": f"validation {optimize_metric} subject to coverage_feasible >= objective.min_feasible_coverage",
        "model_family": "safe_lowest_price_gap_single_mlp_normalized_asym_bucket_abstain",
        "objective": config["objective"],
        "coverage_constraint_satisfied": coverage_constraint_satisfied,
        "selected_calibration": {
            "alpha": selected_candidate.alpha,
            "delta_norm": selected_candidate.delta_norm,
            "delta_quantile": selected_candidate.delta_quantile,
            "bucket_miss_threshold": selected_candidate.bucket_miss_threshold,
        },
        "calibration_metric_note": "delta_norm candidates use feasible normalized residual quantiles; selection is hard-filtered by calibration coverage_feasible plus calibration_coverage_buffer and non_active_share",
        "loss": config["loss"],
        "bucket_abstain": {
            **config["bucket_abstain"],
            "selected_bucket_count": len(selected_candidate.bucket_model.get("table", {})),
            "global_miss_rate": selected_candidate.bucket_model.get("global_miss_rate"),
        },
        "sample_filter": {
            "train": train_filter,
            "validation": validation_filter,
        },
        "fit_metrics": fit_metrics,
        "calibration_metrics": cal_metrics,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "fit_action_diagnostics": fit_action_diag,
        "calibration_action_diagnostics": cal_action_diag,
        "train_action_diagnostics": train_action_diag,
        "validation_action_diagnostics": validation_action_diag,
        "normalized_residual_diagnostics": normalized_residual_report,
        "per_pside_bin_delta_norm": per_pside_delta_norm_report,
        "fit_pside_bin_metrics": pside_bin_metric_report["fit"],
        "calibration_pside_bin_metrics": pside_bin_metric_report["calibration"],
        "train_pside_bin_metrics": pside_bin_metric_report["train"],
        "validation_pside_bin_metrics": pside_bin_metric_report["validation"],
        "fit_window": {
            "row_count": int(len(fit)),
            "start": str(pd.to_datetime(fit["timestamp"], utc=True).min()) if "timestamp" in fit else None,
            "end": str(pd.to_datetime(fit["timestamp"], utc=True).max()) if "timestamp" in fit else None,
        },
        "calibration_window": {
            "row_count": int(len(calibration)),
            "start": str(pd.to_datetime(calibration["timestamp"], utc=True).min()) if "timestamp" in calibration else None,
            "end": str(pd.to_datetime(calibration["timestamp"], utc=True).max()) if "timestamp" in calibration else None,
        },
        "train_window": {
            "row_count": int(len(train_all)),
            "start": str(pd.to_datetime(train_all["timestamp"], utc=True).min()) if "timestamp" in train_all else None,
            "end": str(pd.to_datetime(train_all["timestamp"], utc=True).max()) if "timestamp" in train_all else None,
        },
        "validation_window": {
            "row_count": int(len(validation)),
            "start": str(pd.to_datetime(validation["timestamp"], utc=True).min()) if "timestamp" in validation else None,
            "end": str(pd.to_datetime(validation["timestamp"], utc=True).max()) if "timestamp" in validation else None,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "numpy_artifact": str(numpy_artifact_path),
            "config_snapshot": str(reports_dir / "config_used.yaml"),
            "epoch_metrics": str(reports_dir / "epoch_metrics.csv"),
            "calibration_frontier": str(reports_dir / "calibration_frontier.csv"),
            "train_predictions": str(train_pred_path),
            "calibration_predictions": str(cal_pred_path),
            "validation_predictions": str(validation_pred_path),
        },
        "feature_count_raw": len(columns),
        "feature_count_encoded": int(x_fit.shape[1]),
        "feature_columns": columns,
        "categorical_columns": cat_cols,
        "deploy_artifact_dir": config["paths"]["deploy_artifact_dir"],
        "deploy_experiment_id": manifest.get("experiment_id"),
        "deploy_training_mode": manifest.get("training_mode"),
        "offline_validation_metric_source": manifest.get("source_report_path"),
        "deploy_training_mode_for_this_experiment": "offline_experiment_only",
        "success_criteria": {
            "coverage_feasible_at_least_target": coverage_constraint_satisfied,
            "side_violation_rate_is_zero": validation_metrics["side_violation_rate"] == 0.0,
            "primary_metric": validation_metrics.get(optimize_metric),
        },
    }
    report_path = reports_dir / "summary_metrics.json"
    write_json(report_path, report)
    print(json.dumps({
        "selected_calibration": report["selected_calibration"],
        "train_metrics": train_metrics,
        "calibration_metrics": cal_metrics,
        "validation_metrics": validation_metrics,
        "coverage_constraint_satisfied": coverage_constraint_satisfied,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
