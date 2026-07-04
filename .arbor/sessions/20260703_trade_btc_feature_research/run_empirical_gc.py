#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
PREV = ROOT / ".arbor/sessions/20260703_prefinal_rolling"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]
PBINS = np.array([0.0, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01])


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("emp_gc_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
feat = load_module("emp_gc_feat", SESSION / "run_feature_research.py")


def cdf(lows: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.array([(lows <= bid).mean() for bid in grid], dtype=float) if len(lows) else np.zeros(len(grid))


def empirical_gc(train: pd.DataFrame, dev: pd.DataFrame, grid: np.ndarray, days: int | None,
                 stratify: str) -> np.ndarray:
    correct = train.loc[train["correct"].astype(bool)].copy()
    if days is not None:
        ts = pd.to_datetime(correct["timestamp"], utc=True)
        correct = correct.loc[ts >= ts.max() - pd.Timedelta(days=days)]
    global_cdf = cdf(correct["chosen_low"].to_numpy(float), grid)
    out = np.tile(global_cdf, (len(dev), 1))
    if stratify == "none":
        return out
    train_bins = np.digitize(correct["p_side"].to_numpy(float), PBINS[1:-1])
    dev_bins = np.digitize(dev["p_side"].to_numpy(float), PBINS[1:-1])
    train_side = (correct["selected_side"].astype(str).str.upper() == "UP").to_numpy(int)
    dev_side = (dev["selected_side"].astype(str).str.upper() == "UP").to_numpy(int)
    keys = [(b,) for b in range(len(PBINS) - 1)] if stratify == "pbin" else [
        (b, s) for b in range(len(PBINS) - 1) for s in (0, 1)
    ]
    for key in keys:
        mask_train = train_bins == key[0]
        mask_dev = dev_bins == key[0]
        if len(key) == 2:
            mask_train &= train_side == key[1]
            mask_dev &= dev_side == key[1]
        n = int(mask_train.sum())
        if n == 0:
            continue
        local = cdf(correct.loc[mask_train, "chosen_low"].to_numpy(float), grid)
        weight = n / (n + 200.0)
        out[mask_dev] = weight * local + (1.0 - weight) * global_cdf
    return out


def prepare_fold(fold: str) -> dict:
    d = PREV / "folds" / fold
    train_path, dev_path = d / "data/train.parquet", d / "data/dev.parquet"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    prepared = joint.prepare(train_path, dev_path, d / "models/hazard_survival_cdf.pt", 20260703)
    lookup = feat.load_trades([train, dev])
    tr_trade, dv_trade = feat.engineer_trades(train, lookup), feat.engineer_trades(dev, lookup)
    hazard_cols, _, grid, _ = joint.load_hazard(d / "models/hazard_survival_cdf.pt")
    q_trade = feat.fit_predict(pd.concat([train, tr_trade], axis=1), pd.concat([dev, dv_trade], axis=1),
                               hazard_cols + list(tr_trade.columns), 20260703)
    q = 0.5 * prepared[2]["raw_tree_blend"] + 0.5 * q_trade
    tr = train.loc[train["threshold_accepted"].astype(bool)].reset_index(drop=True)
    dv = dev.loc[dev["threshold_accepted"].astype(bool)].reset_index(drop=True)
    variants = {"hazard": prepared[3]}
    for days in (None, 14, 28):
        for stratify in ("none", "pbin", "pbin_side"):
            variants[f"emp_{days or 'all'}_{stratify}"] = empirical_gc(tr, dv, grid, days, stratify)
    correct = dv["correct"].astype(bool).to_numpy()
    target = (dv.loc[correct, "chosen_low"].to_numpy(float)[:, None] <= grid[None, :]).astype(float)
    brier = {name: float(np.mean((gc[correct] - target) ** 2)) for name, gc in variants.items()}
    return {"dev": dev, "accepted": dv, "q": q, "grid": grid, "gc": variants, "brier": brier}


def evaluate(p: dict, gc_name: str, floor: float, min_ev: float, max_bid: float) -> dict:
    allowed = p["grid"] <= max_bid + 1e-9
    bid, ev, fill = joint.choose_survival_expected_return_bids(
        p["q"], p["gc"][gc_name][:, allowed], p["grid"][allowed], 0.01, min_ev, min_fill_probability=floor,
    )
    return joint.backtest_metrics(p["accepted"], joint.backtest_with_bid(p["accepted"], bid, ev, fill), len(p["dev"]))


def main() -> None:
    prepared = {f: prepare_fold(f) for f in FOLDS}
    rows = []
    for gc_name in prepared["w1"]["gc"]:
        for max_bid in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
          for floor in (0.0, 0.50, 0.60, 0.70, 0.80):
            for min_ev in (0.0, 0.005, 0.01, 0.02, 0.03):
                metrics = {f: evaluate(prepared[f], gc_name, floor, min_ev, max_bid) for f in FOLDS}
                pnl = {f: metrics[f]["sum_pnl"] for f in FOLDS}
                tune = np.array([pnl[f] for f in TUNE])
                rows.append({"gc_model": gc_name, "max_bid": max_bid, "fill_floor": floor, "min_ev": min_ev,
                             **{f"{f}_pnl": pnl[f] for f in FOLDS}, "tune_sum": float(tune.sum()),
                             "tune_worst": float(tune.min()), "tune_robust": float(tune.sum() - tune.std()),
                             "holdout_sum": float(sum(pnl[f] for f in HOLDOUT)),
                             "holdout_worst": float(min(pnl[f] for f in HOLDOUT)),
                             "mean_bid_tune": float(np.mean([metrics[f]["mean_bid"] for f in TUNE]))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0]
    table.to_csv(SESSION / "empirical_gc_search.csv", index=False)
    summary = {"experiment_count": len(table), "winner": winner.to_dict(),
               "brier": {f: p["brier"] for f, p in prepared.items()}, "btest_used": False,
               "top20": table.head(20).to_dict("records")}
    (SESSION / "empirical_gc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"winner": summary["winner"], "brier": summary["brier"]}, indent=2))


if __name__ == "__main__":
    main()
