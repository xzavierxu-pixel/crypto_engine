#!/usr/bin/env python3
"""Full-information contextual bid policy evaluated on chronological B_dev only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
SOURCE = ROOT / ".arbor/sessions/20260703_trade_btc_feature_research/run_minimal_q.py"
PREV = ROOT / ".arbor/sessions/20260703_prefinal_rolling"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]
SEED = 20260704


def load_base():
    spec = importlib.util.spec_from_file_location("contextual_policy_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


base = load_base()

# Explicit allowlist of decision-time columns. Labels and realized price fields are
# used only to construct training rewards, never as policy inputs.
FEATURES = [
    "p_side", "direction_confidence", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "sl_return_5s", "sl_return_30s", "sl_rv_30s", "sl_taker_imbalance_30s",
    "sl_signed_dollar_flow_ratio_30s", "sl_directional_efficiency_30s",
    "sl_choppiness_30s", "sl_price_minus_vwap_30s", "sl_volume_burst_30s",
    "ret_1", "ret_3", "ret_5", "rv_5", "relative_volume_5", "volume_z_5",
]


def context(frame: pd.DataFrame) -> np.ndarray:
    cols = []
    for name in FEATURES:
        if name in frame:
            cols.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(float))
        else:
            cols.append(np.zeros(len(frame)))
    cols.append((frame["selected_side"].astype(str).str.upper() == "UP").to_numpy(float))
    x = np.column_stack(cols)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def fit_and_score(fold: str) -> tuple[dict, dict]:
    payload = base.prepare_fold(fold)
    train = pd.read_parquet(PREV / "folds" / fold / "data/train.parquet")
    train = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    xtr, xdv = context(train), context(payload["accepted"])
    grid = payload["grid"]
    action_idx = np.arange(4, min(len(grid), 86), 4)  # legal 4-cent action grid
    bids = grid[action_idx]

    correct = train["correct"].astype(bool).to_numpy()
    low = pd.to_numeric(train["chosen_low"], errors="coerce").fillna(1.0).to_numpy(float)
    reward = np.where(
        correct[:, None],
        np.where(low[:, None] <= bids[None, :], 1.0 - bids[None, :], 0.0),
        -bids[None, :],
    ).astype("float32")
    n, a = reward.shape
    train_x = np.column_stack([
        np.repeat(xtr, a, axis=0),
        np.tile(bids, n),
        np.tile(bids * bids, n),
    ])
    model = LGBMRegressor(
        objective="regression_l1", n_estimators=350, learning_rate=0.035,
        num_leaves=20, max_depth=5, min_child_samples=200,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=20.0, reg_alpha=3.0,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    model.fit(train_x, reward.reshape(-1))
    nd = len(xdv)
    dev_x = np.column_stack([
        np.repeat(xdv, a, axis=0),
        np.tile(bids, nd),
        np.tile(bids * bids, nd),
    ])
    values = model.predict(dev_x).reshape(nd, a)

    variants = {}
    for min_value in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
        for max_bid in (0.40, 0.52, 0.64, 0.76, 0.84):
            legal = bids <= max_bid
            masked = np.where(legal[None, :], values, -np.inf)
            idx = np.argmax(masked, axis=1)
            val = masked[np.arange(nd), idx]
            submit = val >= min_value
            bid = np.where(submit, bids[idx], 0.0)
            result = base.joint.backtest_with_bid(payload["accepted"], bid, np.maximum(val, 0.0), np.zeros(nd))
            key = f"v{min_value:.3f}_b{max_bid:.2f}"
            variants[key] = base.joint.backtest_metrics(payload["accepted"], result, len(payload["dev"]))
    audit = {
        "training_rows": int(n), "action_count": int(a), "feature_count": int(xtr.shape[1]),
        "feature_allowlist": FEATURES + ["selected_side_up"],
        "forbidden_feature_intersection": sorted(set(FEATURES) & {"target", "correct", "chosen_low", "winner", "pnl", "stage1_sample_weight"}),
    }
    return variants, audit


def main() -> None:
    fold_variants, audits = {}, {}
    for fold in FOLDS:
        fold_variants[fold], audits[fold] = fit_and_score(fold)
    keys = list(fold_variants["w1"])
    rows = []
    for key in keys:
        pnls = {f: float(fold_variants[f][key]["sum_pnl"]) for f in FOLDS}
        tune = np.array([pnls[f] for f in TUNE])
        rows.append({"variant": key, **{f"{f}_pnl": pnls[f] for f in FOLDS},
                     "tune_sum": float(tune.sum()), "tune_worst": float(tune.min()),
                     "tune_robust": float(tune.sum() - tune.std()),
                     "holdout_sum": float(sum(pnls[f] for f in HOLDOUT))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0].to_dict()
    selected = winner["variant"]
    summary = {
        "method": "full_information_contextual_action_value",
        "selection_folds": TUNE, "untouched_holdout_folds": HOLDOUT, "btest_used": False,
        "winner": winner, "fold_metrics": {f: fold_variants[f][selected] for f in FOLDS},
        "leakage_audit": audits, "top20": table.head(20).to_dict("records"),
    }
    table.to_csv(SESSION / "contextual_action_value_search.csv", index=False)
    (SESSION / "contextual_action_value_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": winner, "leakage_audit": audits["w1"]}, indent=2))


if __name__ == "__main__":
    main()
