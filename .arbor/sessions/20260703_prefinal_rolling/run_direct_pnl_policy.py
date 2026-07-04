#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
module_path = ROOT/".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py"
spec = importlib.util.spec_from_file_location("joint", module_path)
joint = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(joint)

FOLDS = [f"w{i}" for i in range(1,7)]
BIDS = np.round(np.arange(.05,.86,.05),2)
THRESHOLDS = [0,.001,.0025,.005,.0075,.01,.015,.02,.03,.04,.05]


def realized_pnl(correct: np.ndarray, low: np.ndarray, bid: float) -> np.ndarray:
    return np.where(correct, np.where(low <= bid + 1e-12, 1-bid, 0.0), -bid)


def fit_fold(fold: str):
    d = SESSION/"folds"/fold
    train = pd.read_parquet(d/"data/train.parquet")
    dev = pd.read_parquet(d/"data/dev.parquet")
    ck = torch.load(d/"models/hazard_survival_cdf.pt",map_location="cpu",weights_only=False)
    cols = list(ck["feature_columns"])
    bad = [c for c in cols if c in joint.FORBIDDEN_EXACT or c.startswith("future_") or "sample_weight" in c.lower()]
    if bad: raise RuntimeError(f"forbidden features {bad}")
    tr = train.loc[train.threshold_accepted.astype(bool)].copy()
    dv = dev.loc[dev.threshold_accepted.astype(bool)].copy()
    xtr_all, xdv_all = joint.matrix(tr,cols), joint.matrix(dv,cols)
    ycorrect = tr.correct.astype(int).to_numpy()
    selector = LGBMClassifier(n_estimators=250,learning_rate=.03,num_leaves=15,max_depth=5,min_child_samples=100,
                              colsample_bytree=.4,reg_lambda=10,reg_alpha=2,verbosity=-1,n_jobs=-1,random_state=20260703)
    selector.fit(xtr_all,ycorrect)
    top_idx = np.argsort(selector.feature_importances_)[-60:]
    top = [cols[i] for i in top_idx]
    xtr = xtr_all[top].to_numpy(np.float32)
    xdv = xdv_all[top].to_numpy(np.float32)
    correct = tr.correct.astype(bool).to_numpy()
    low = tr.chosen_low.to_numpy(float)
    # One shared action-value model learns E[PnL | X, bid].
    train_x = np.vstack([np.c_[xtr,np.full(len(xtr),b,dtype=np.float32)] for b in BIDS])
    train_y = np.concatenate([realized_pnl(correct,low,b) for b in BIDS])
    model = LGBMRegressor(objective="huber",n_estimators=500,learning_rate=.025,num_leaves=31,max_depth=7,
                          min_child_samples=150,colsample_bytree=.65,subsample=.8,reg_lambda=15,reg_alpha=3,
                          verbosity=-1,n_jobs=-1,random_state=20260703)
    model.fit(train_x,train_y)
    pred = np.column_stack([model.predict(np.c_[xdv,np.full(len(xdv),b,dtype=np.float32)]) for b in BIDS])
    best_idx = pred.argmax(axis=1)
    return dev,dv,BIDS[best_idx],pred[np.arange(len(dv)),best_idx],top


def evaluate(dev,dv,bid,score,threshold):
    chosen = np.where(score>=threshold,bid,0.0)
    result = joint.backtest_with_bid(dv,chosen,score,np.full(len(dv),np.nan))
    return joint.backtest_metrics(dv,result,len(dev))


def main():
    fitted={f:fit_fold(f) for f in FOLDS}
    rows=[]
    for t in THRESHOLDS:
        ms={f:evaluate(*fitted[f][:4],t) for f in FOLDS}
        rows.append({"threshold":t,**{f"{f}_pnl":ms[f]["sum_pnl"] for f in FOLDS},
                     **{f"{f}_orders":ms[f]["order_count"] for f in FOLDS},
                     "tune_sum":sum(ms[f]["sum_pnl"] for f in FOLDS[:4]),
                     "tune_worst":min(ms[f]["sum_pnl"] for f in FOLDS[:4]),
                     "holdout_sum":sum(ms[f]["sum_pnl"] for f in FOLDS[4:])})
    table=pd.DataFrame(rows).sort_values(["tune_sum","tune_worst"],ascending=False)
    winner=table.iloc[0]
    payload={"mechanism":"direct action-value regression over 17 bid actions","selection_folds":FOLDS[:4],
             "holdout_folds":FOLDS[4:],"winner":winner.to_dict(),
             "holdout_gate_passed":bool(winner.holdout_sum>0 and winner.w5_pnl>0 and winner.w6_pnl>0),
             "top_features_by_fold":{f:fitted[f][4] for f in FOLDS},"all_thresholds":table.to_dict("records")}
    table.to_csv(SESSION/"direct_pnl_thresholds.csv",index=False)
    (SESSION/"direct_pnl_summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
