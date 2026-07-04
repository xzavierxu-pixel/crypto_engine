#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
SOURCE = ROOT / ".arbor/sessions/20260703_trade_btc_feature_research/run_minimal_q.py"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]
SEED = 20260704


def load_module():
    spec = importlib.util.spec_from_file_location("bayes_policy_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


base = load_module()


def policy(payload: dict, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a fixed calibrated q and fixed Gc surface to one legal bid per row."""
    q = np.clip(payload["qs"]["raw_minimal_0.25"], 1e-5, 1 - 1e-5)
    gc = payload["gcs"]["hazard"]
    grid = payload["grid"]
    q_eff = expit(theta[0] * logit(q) + theta[1])
    floor = np.clip(theta[2] + theta[3] * (q - 0.65), 0.50, 0.98)
    max_bid = np.clip(theta[4] + theta[5] * (q - 0.65), 0.20, 0.90)
    ev = q_eff[:, None] * gc * (1.0 - grid[None, :]) - (1.0 - q_eff[:, None]) * grid[None, :]
    legal = (gc >= floor[:, None]) & (grid[None, :] <= max_bid[:, None])
    masked = np.where(legal, ev, -np.inf)
    idx = np.argmax(masked, axis=1)
    value = masked[np.arange(len(q)), idx]
    submit = np.isfinite(value) & (value >= theta[6]) & (q >= theta[7])
    bid = np.where(submit, grid[idx], 0.0)
    return bid, np.where(submit, value, 0.0), np.where(submit, gc[np.arange(len(q)), idx], 0.0)


def metrics(payload: dict, theta: np.ndarray) -> dict:
    bid, ev, fill = policy(payload, theta)
    result = base.joint.backtest_with_bid(payload["accepted"], bid, ev, fill)
    return base.joint.backtest_metrics(payload["accepted"], result, len(payload["dev"]))


BOUNDS = np.array([
    [0.60, 1.50], [-0.50, 0.50], [0.60, 0.92], [-0.80, 0.80],
    [0.35, 0.90], [-1.00, 1.00], [-0.01, 0.08], [0.50, 0.82],
])


def scale(x: np.ndarray) -> np.ndarray:
    return BOUNDS[:, 0] + x * (BOUNDS[:, 1] - BOUNDS[:, 0])


def main() -> None:
    rng = np.random.default_rng(SEED)
    prepared = {fold: base.prepare_fold(fold) for fold in FOLDS}
    cache: dict[tuple[float, ...], tuple[float, dict]] = {}

    def objective(x: np.ndarray) -> tuple[float, dict]:
        theta = scale(x)
        key = tuple(np.round(theta, 8))
        if key not in cache:
            fold_metrics = {f: metrics(prepared[f], theta) for f in TUNE}
            pnls = np.array([fold_metrics[f]["sum_pnl"] for f in TUNE])
            score = float(pnls.sum() - pnls.std())
            cache[key] = score, fold_metrics
        return cache[key]

    # Seed with space-filling random designs, then sequential expected improvement.
    xs = rng.random((32, BOUNDS.shape[0]))
    ys = np.array([objective(x)[0] for x in xs])
    kernel = Matern(length_scale=np.ones(BOUNDS.shape[0]), nu=2.5) + WhiteKernel(1e-4)
    for _ in range(48):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=SEED, n_restarts_optimizer=0)
        gp.fit(xs, ys)
        candidates = rng.random((4096, BOUNDS.shape[0]))
        mean, std = gp.predict(candidates, return_std=True)
        z = (mean - ys.max()) / np.maximum(std, 1e-9)
        # Normal EI without a scipy.stats dependency in the hot loop.
        ei = (mean - ys.max()) * ndtr(z)
        ei += std * np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        x = candidates[int(np.argmax(ei))]
        xs = np.vstack([xs, x])
        ys = np.r_[ys, objective(x)[0]]

    best = int(np.argmax(ys))
    theta = scale(xs[best])
    _, tune_metrics = objective(xs[best])
    holdout_metrics = {f: metrics(prepared[f], theta) for f in HOLDOUT}
    all_metrics = {**tune_metrics, **holdout_metrics}
    rows = []
    for x, score in zip(xs, ys):
        row = {f"theta_{i}": float(v) for i, v in enumerate(scale(x))}
        rows.append({**row, "tune_robust": float(score)})
    pd.DataFrame(rows).sort_values("tune_robust", ascending=False).to_csv(
        SESSION / "bayesian_bid_policy_search.csv", index=False
    )
    summary = {
        "method": "gaussian_process_expected_improvement",
        "fixed_q": "raw_minimal_0.25",
        "fixed_gc": "hazard",
        "evaluation_count": len(xs),
        "selection_folds": TUNE,
        "untouched_holdout_folds": HOLDOUT,
        "btest_used": False,
        "theta": theta.tolist(),
        "tune_robust": float(ys[best]),
        "fold_metrics": all_metrics,
        "holdout_sum_pnl": float(sum(holdout_metrics[f]["sum_pnl"] for f in HOLDOUT)),
    }
    (SESSION / "bayesian_bid_policy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
