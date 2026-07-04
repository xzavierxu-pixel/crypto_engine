#!/usr/bin/env python3
from __future__ import annotations

import importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

ROOT=Path(__file__).resolve().parents[3]
SESSION=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("joint",ROOT/".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
joint=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(joint)
FOLDS=[f"w{i}" for i in range(1,7)]

def fit_fold(fold):
    fd=SESSION/"folds"/fold
    raw_train=pd.read_parquet(fd/"data/train.parquet",columns=["timestamp"])
    raw_dev=pd.read_parquet(fd/"data/dev.parquet",columns=["timestamp"])
    all_data=pd.read_parquet(SESSION/"pretest_both_side_lows.parquet")
    ts=pd.to_datetime(all_data.timestamp,utc=True)
    tr=all_data.loc[ts.isin(pd.to_datetime(raw_train.timestamp,utc=True))].copy()
    dv=all_data.loc[ts.isin(pd.to_datetime(raw_dev.timestamp,utc=True))].copy()
    cols,prep,grid,hazard=joint.load_hazard(fd/"models/hazard_survival_cdf.pt")
    xtr,xdv=joint.matrix(tr,cols),joint.matrix(dv,cols)
    y=tr.target.astype(int).to_numpy()
    lgb=LGBMClassifier(n_estimators=450,learning_rate=.025,num_leaves=15,max_depth=5,min_child_samples=150,
                       colsample_bytree=.4,reg_lambda=10,reg_alpha=2,verbosity=-1,n_jobs=-1,random_state=20260703).fit(xtr,y)
    cat=CatBoostClassifier(iterations=450,depth=5,learning_rate=.03,l2_leaf_reg=12,loss_function="Logloss",
                           verbose=False,allow_writing_files=False,random_seed=20260703).fit(xtr,y)
    raw=dv.p_up.to_numpy(float); lp=lgb.predict_proba(xdv)[:,1]; cp=cat.predict_proba(xdv)[:,1]; tree=.5*lp+.5*cp
    probs={"raw":raw,"lgbm":lp,"catboost":cp,"tree":tree}
    for a in [.25,.5,.75]: probs[f"raw_tree_{a}"]=(1-a)*raw+a*tree
    prepared={}
    fixed_mask=dv.threshold_accepted.astype(bool).to_numpy()
    for name,p in probs.items():
        side=np.where(p>=.5,"UP","DOWN")
        work=dv.copy(); work["selected_side"]=side; work["p_up"]=p; work["p_side"]=np.maximum(p,1-p)
        work["direction_confidence"]=np.abs(p-.5)
        work["correct"]=(side==np.where(work.target.astype(int).to_numpy()==1,"UP","DOWN"))
        original=dv.selected_side.str.upper().to_numpy()
        alternative_low=np.where(side=="UP",work.up_low,work.down_low)
        work["chosen_low"]=np.where(side==original,dv.chosen_low,alternative_low)
        gc=joint.predict_hazard(hazard,prep.transform(work),torch.device("cpu"),512)[1][fixed_mask]
        accepted=work.loc[fixed_mask].copy()
        prepared[name]=(work,accepted,gc,grid)
    return prepared

def evaluate(p,shrink,floor,min_ev):
    full,a,gc,grid=p; q=(1-shrink)*a.p_side.to_numpy(float)+shrink*.5
    bid,ev,fill=joint.choose_survival_expected_return_bids(q,gc,grid,.01,min_ev,min_fill_probability=floor)
    missing=~np.isfinite(a.chosen_low.to_numpy(float)); bid[missing]=0
    return joint.backtest_metrics(a,joint.backtest_with_bid(a,bid,ev,fill),len(full))

def main():
    folds={f:fit_fold(f) for f in FOLDS}
    baseline={f:evaluate(folds[f]["raw"],0,.75,0)["sum_pnl"] for f in FOLDS}
    rows=[]
    for direction in folds["w1"]:
      for shrink in [0,.1,.2,.3]:
       for floor in [.7,.75,.8,.85,.9]:
        for mev in [0,.005,.01,.02,.03,.05]:
         ms={f:evaluate(folds[f][direction],shrink,floor,mev) for f in FOLDS}; pnl={f:ms[f]["sum_pnl"] for f in FOLDS}
         delta={f:pnl[f]-baseline[f] for f in FOLDS}; tune=np.array([pnl[f] for f in FOLDS[:4]])
         rows.append({"direction_model":direction,"q_shrink":shrink,"gc_floor":floor,"min_ev":mev,
                      **{f"{f}_pnl":pnl[f] for f in FOLDS},**{f"{f}_delta":delta[f] for f in FOLDS},
                      "tune_sum":tune.sum(),"tune_worst":tune.min(),"tune_positive_delta_weeks":int(sum(delta[f]>0 for f in FOLDS[:4])),
                      "tune_robust":float(tune.sum()-tune.std()),"holdout_sum":pnl["w5"]+pnl["w6"],
                      "holdout_delta":delta["w5"]+delta["w6"],
                      "w5_accuracy":ms["w5"]["accepted_sample_accuracy"],"w6_accuracy":ms["w6"]["accepted_sample_accuracy"]})
    table=pd.DataFrame(rows); eligible=table[table.tune_positive_delta_weeks>=3]
    winner=eligible.sort_values(["tune_robust","tune_sum","tune_worst"],ascending=False).iloc[0]
    table.sort_values(["tune_robust","tune_sum"],ascending=False).to_csv(SESSION/"joint_direction_search.csv",index=False)
    payload={"experiment_count":len(table),"baseline":baseline,"winner":winner.to_dict(),
             "holdout_gate_passed":bool(winner.holdout_delta>0 and winner.w5_pnl>0 and winner.w6_pnl>0),
             "top20":table.sort_values(["tune_robust","tune_sum"],ascending=False).head(20).to_dict("records")}
    (SESSION/"joint_direction_summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
