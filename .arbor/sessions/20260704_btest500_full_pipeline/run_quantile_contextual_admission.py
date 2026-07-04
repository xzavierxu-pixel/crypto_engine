#!/usr/bin/env python3
"""Downside-aware contextual admission for the validated analytic bid policy."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
PREV = ROOT / ".arbor/sessions/20260703_prefinal_rolling"
JOINT_PATH = ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
ANCHOR_PATH = SESSION / "run_anchored_contextual_lcb.py"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]
SEED = 20260704


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("quantile_admission_joint", JOINT_PATH)
anchor = load_module("quantile_admission_context", ANCHOR_PATH)


def training_matrix(train: pd.DataFrame, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-information action rewards; labels never enter the policy feature matrix."""
    x = anchor.context(train)
    action_idx = np.arange(4, min(len(grid), 86), 4)
    bids = grid[action_idx]
    correct = train["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(train["chosen_low"], errors="coerce").fillna(1.0).to_numpy(float)
    reward = np.where(
        correct[:, None],
        np.where(low[:, None] <= bids[None, :], 1.0 - bids[None, :], 0.0),
        -bids[None, :],
    ).astype("float32")
    model_x = np.column_stack([
        np.repeat(x, len(bids), axis=0),
        np.tile(bids, len(train)),
        np.tile(bids * bids, len(train)),
    ])
    return model_x, reward.reshape(-1)


def fit_reward_models(train: pd.DataFrame, grid: np.ndarray) -> dict[str, LGBMRegressor]:
    x, y = training_matrix(train, grid)
    specs = {
        "mean": {"objective": "regression_l2"},
        "q30": {"objective": "quantile", "alpha": 0.30},
        "median": {"objective": "quantile", "alpha": 0.50},
    }
    models = {}
    for offset, (name, objective) in enumerate(specs.items()):
        model = LGBMRegressor(
            **objective, n_estimators=220, learning_rate=0.035,
            num_leaves=15, max_depth=5, min_child_samples=250,
            subsample=0.78, subsample_freq=1, colsample_bytree=0.75,
            reg_lambda=30.0, reg_alpha=5.0, random_state=SEED + offset,
            verbosity=-1, n_jobs=-1,
        )
        model.fit(x, y)
        models[name] = model
    return models


def analytic_bid(prepared: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, accepted, qs, gc, grid, _, _ = prepared
    return joint.choose_survival_expected_return_bids(
        qs["raw_tree_blend"], gc, grid, 0.01, 0.02, min_fill_probability=0.85
    )


def contextual_scores(
    models: dict[str, LGBMRegressor], accepted: pd.DataFrame, bid: np.ndarray
) -> dict[str, np.ndarray]:
    x = anchor.context(accepted)
    query = np.column_stack([x, bid, bid * bid])
    raw = {name: model.predict(query) for name, model in models.items()}

    def z(value: np.ndarray) -> np.ndarray:
        scale = max(float(np.nanstd(value)), 1e-6)
        return (value - float(np.nanmean(value))) / scale

    return {
        **raw,
        "mean_q30": 0.5 * z(raw["mean"]) + 0.5 * z(raw["q30"]),
        "mean_per_risk": raw["mean"] / np.maximum(bid, 0.04),
    }


def admitted_bid(
    bid: np.ndarray, ev: np.ndarray, fill: np.ndarray, score: np.ndarray, keep: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    submitted = bid > 0
    if keep >= 1.0 or not np.any(submitted):
        return bid.copy(), ev.copy(), fill.copy(), float("-inf")
    cutoff = float(np.quantile(score[submitted], 1.0 - keep))
    retain = submitted & (score >= cutoff)
    return np.where(retain, bid, 0.0), np.where(retain, ev, 0.0), np.where(retain, fill, 0.0), cutoff


def evaluate_fold(fold: str) -> tuple[dict[str, dict], dict]:
    d = PREV / "folds" / fold
    prepared = joint.prepare(
        d / "data/train.parquet", d / "data/dev.parquet", d / "models/hazard_survival_cdf.pt", SEED
    )
    dev, accepted, _, _, grid, _, _ = prepared
    train = pd.read_parquet(d / "data/train.parquet")
    train = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    models = fit_reward_models(train, grid)
    base_bid, base_ev, base_fill = analytic_bid(prepared)
    scores = contextual_scores(models, accepted, base_bid)
    variants = {}
    for score_name, score in scores.items():
        for keep in (0.30, 0.45, 0.60, 0.75, 0.90, 1.00):
            bid, ev, fill, cutoff = admitted_bid(base_bid, base_ev, base_fill, score, keep)
            result = joint.backtest_with_bid(accepted, bid, ev, fill)
            metrics = joint.backtest_metrics(accepted, result, len(dev))
            metrics["contextual_score_cutoff"] = cutoff
            variants[f"{score_name}_k{keep:.2f}"] = metrics
    audit = {
        "training_rows": len(train), "reward_models": list(models),
        "feature_allowlist": anchor.FEATURES + ["selected_side_up", "candidate_bid", "candidate_bid_squared"],
        "forbidden_feature_intersection": sorted(set(anchor.FEATURES) & {
            "target", "correct", "chosen_low", "winner", "pnl", "trade_time", "endDate",
            "condition_id", "market_id", "slug", "outcome",
        }),
        "base_order_count": int(np.sum(base_bid > 0)),
    }
    return variants, audit


def main() -> None:
    by_fold, audits = {}, {}
    for fold in FOLDS:
        by_fold[fold], audits[fold] = evaluate_fold(fold)
        print(f"finished {fold}", flush=True)
    rows = []
    for key in by_fold["w1"]:
        pnls = {fold: float(by_fold[fold][key]["sum_pnl"]) for fold in FOLDS}
        tune = np.array([pnls[fold] for fold in TUNE])
        rows.append({
            "variant": key, **{f"{fold}_pnl": pnls[fold] for fold in FOLDS},
            "tune_sum": float(tune.sum()), "tune_worst": float(tune.min()),
            "tune_robust": float(tune.sum() - tune.std()),
            "holdout_sum": float(sum(pnls[fold] for fold in HOLDOUT)),
            "holdout_worst": float(min(pnls[fold] for fold in HOLDOUT)),
        })
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0].to_dict()
    selected = str(winner["variant"])
    summary = {
        "method": "cross_fitted_quantile_contextual_admission",
        "selection_folds": TUNE, "untouched_holdout_folds": HOLDOUT,
        "btest_used": False, "winner": winner,
        "holdout_gate_passed": bool(winner["holdout_worst"] > 0),
        "fold_metrics": {fold: by_fold[fold][selected] for fold in FOLDS},
        "leakage_audit": audits, "top20": table.head(20).to_dict("records"),
    }
    table.to_csv(SESSION / "quantile_contextual_admission_search.csv", index=False)
    (SESSION / "quantile_contextual_admission_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": winner, "holdout_gate_passed": summary["holdout_gate_passed"], "audit": audits["w1"]}, indent=2))


if __name__ == "__main__":
    main()
