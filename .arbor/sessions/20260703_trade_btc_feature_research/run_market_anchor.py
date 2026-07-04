#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
PREV = ROOT / ".arbor/sessions/20260703_prefinal_rolling"
FOLDS = [f"w{i}" for i in range(1, 7)]
TUNE, HOLDOUT = FOLDS[:4], FOLDS[4:]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


joint = load_module("market_anchor_base", ROOT / ".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
feat = load_module("market_anchor_features", SESSION / "run_feature_research.py")


def anchor_candidates(frame: pd.DataFrame, trade: pd.DataFrame) -> dict[str, np.ndarray]:
    raw = frame["p_side"].to_numpy(float)
    sc = trade["pm_trade_selected_count_60s"].to_numpy(float) > 0
    oc = trade["pm_trade_opposite_count_60s"].to_numpy(float) > 0
    selected = trade["pm_trade_selected_last_60s"].to_numpy(float)
    opposite_complement = 1.0 - trade["pm_trade_opposite_last_60s"].to_numpy(float)
    market = raw.copy()
    market[sc & ~oc] = selected[sc & ~oc]
    market[~sc & oc] = opposite_complement[~sc & oc]
    both = sc & oc
    market[both] = 0.5 * (selected[both] + opposite_complement[both])
    pair = raw.copy()
    denom = selected + trade["pm_trade_opposite_last_60s"].to_numpy(float)
    pair[both] = np.divide(selected[both], denom[both], out=market[both].copy(), where=denom[both] > 1e-8)
    market = np.clip(market, 0.01, 0.99)
    pair = np.clip(pair, 0.01, 0.99)
    out = {"raw": raw, "market": market, "pair_norm": pair}
    for name, anchor in (("market", market), ("pair", pair)):
        for weight in (0.25, 0.50, 0.75):
            out[f"raw_{name}_{weight:.2f}"] = (1.0 - weight) * raw + weight * anchor
    return out


def prepare_fold(fold: str) -> dict:
    d = PREV / "folds" / fold
    train_path, dev_path = d / "data/train.parquet", d / "data/dev.parquet"
    train, dev = pd.read_parquet(train_path), pd.read_parquet(dev_path)
    lookup = feat.load_trades([train, dev])
    dev_trade = feat.engineer_trades(dev, lookup)
    accepted_mask = dev["threshold_accepted"].astype(bool).to_numpy()
    accepted = dev.loc[accepted_mask].reset_index(drop=True)
    accepted_trade = dev_trade.loc[accepted_mask].reset_index(drop=True)
    _, prep, grid, hazard = joint.load_hazard(d / "models/hazard_survival_cdf.pt")
    gc_all = joint.predict_hazard(hazard, prep.transform(dev), torch.device("cpu"), 512)[1]
    qs = anchor_candidates(accepted, accepted_trade)
    y = accepted["correct"].astype(int).to_numpy()
    calibration = {
        name: {"brier": float(brier_score_loss(y, q)),
               "log_loss": float(log_loss(y, np.clip(q, 1e-6, 1 - 1e-6)))}
        for name, q in qs.items()
    }
    return {"dev": dev, "accepted": accepted, "gc": gc_all[accepted_mask], "grid": grid,
            "qs": qs, "calibration": calibration,
            "market_coverage": float(((accepted_trade["pm_trade_selected_count_60s"] > 0) |
                                       (accepted_trade["pm_trade_opposite_count_60s"] > 0)).mean())}


def evaluate(p: dict, q_name: str, floor: float, min_ev: float) -> dict:
    q = p["qs"][q_name]
    bid, ev, fill = joint.choose_survival_expected_return_bids(
        q, p["gc"], p["grid"], 0.01, min_ev, min_fill_probability=floor,
    )
    return joint.backtest_metrics(p["accepted"], joint.backtest_with_bid(p["accepted"], bid, ev, fill), len(p["dev"]))


def main() -> None:
    prepared = {f: prepare_fold(f) for f in FOLDS}
    rows = []
    for q_name in prepared["w1"]["qs"]:
        for floor in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            for min_ev in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
                pnl = {f: evaluate(prepared[f], q_name, floor, min_ev)["sum_pnl"] for f in FOLDS}
                tune = np.array([pnl[f] for f in TUNE])
                rows.append({"q_model": q_name, "gc_floor": floor, "min_ev": min_ev,
                             **{f"{f}_pnl": pnl[f] for f in FOLDS},
                             "tune_sum": float(tune.sum()), "tune_worst": float(tune.min()),
                             "tune_robust": float(tune.sum() - tune.std()),
                             "holdout_sum": float(sum(pnl[f] for f in HOLDOUT))})
    table = pd.DataFrame(rows).sort_values(["tune_robust", "tune_sum"], ascending=False)
    winner = table.iloc[0]
    table.to_csv(SESSION / "market_anchor_search.csv", index=False)
    summary = {"experiment_count": len(table), "winner": winner.to_dict(),
               "market_coverage": {f: p["market_coverage"] for f, p in prepared.items()},
               "calibration": {f: p["calibration"] for f, p in prepared.items()}, "btest_used": False}
    (SESSION / "market_anchor_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
