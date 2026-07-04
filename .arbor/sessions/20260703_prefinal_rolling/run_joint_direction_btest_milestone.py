#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

ROOT=Path(__file__).resolve().parents[3]; SESSION=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("joint",ROOT/".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
joint=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(joint)
summary=json.loads((SESSION/"joint_direction_summary.json").read_text()); winner=summary["winner"]
if not summary["holdout_gate_passed"]: raise RuntimeError("rolling holdout gate failed")
tr=pd.read_parquet(SESSION/"pretest_both_side_lows.parquet"); dv=pd.read_parquet(SESSION/"btest_both_side_lows.parquet")
ck_path=ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt"
cols,prep,grid,hazard=joint.load_hazard(ck_path); xtr,xdv=joint.matrix(tr,cols),joint.matrix(dv,cols); y=tr.target.astype(int)
lgb=LGBMClassifier(n_estimators=450,learning_rate=.025,num_leaves=15,max_depth=5,min_child_samples=150,colsample_bytree=.4,
                   reg_lambda=10,reg_alpha=2,verbosity=-1,n_jobs=-1,random_state=20260703).fit(xtr,y)
cat=CatBoostClassifier(iterations=450,depth=5,learning_rate=.03,l2_leaf_reg=12,loss_function="Logloss",verbose=False,
                       allow_writing_files=False,random_seed=20260703).fit(xtr,y)
raw=dv.p_up.to_numpy(float); tree=.5*lgb.predict_proba(xdv)[:,1]+.5*cat.predict_proba(xdv)[:,1]; p=.25*raw+.75*tree
side=np.where(p>=.5,"UP","DOWN"); work=dv.copy(); original=work.selected_side.str.upper().to_numpy()
work["selected_side"]=side; work["p_up"]=p; work["p_side"]=np.maximum(p,1-p); work["direction_confidence"]=np.abs(p-.5)
work["correct"]=(side==np.where(work.target.astype(int).to_numpy()==1,"UP","DOWN"))
alt=np.where(side=="UP",work.up_low,work.down_low); work["chosen_low"]=np.where(side==original,work.chosen_low,alt)
mask=work.threshold_accepted.astype(bool).to_numpy(); accepted=work.loc[mask].copy(); gc=joint.predict_hazard(hazard,prep.transform(work),torch.device("cpu"),512)[1][mask]
q=accepted.p_side.to_numpy(float); bid,ev,fill=joint.choose_survival_expected_return_bids(q,gc,grid,.01,0,min_fill_probability=.85)
bid[~np.isfinite(accepted.chosen_low.to_numpy(float))]=0
metrics=joint.backtest_metrics(accepted,joint.backtest_with_bid(accepted,bid,ev,fill),len(work))
payload={"evaluation_kind":"joint-direction milestone after six rolling pre-test folds","baseline_sum_pnl":27.44,"target":100,
         "candidate":winner,"metrics":metrics,"direction_changed_count":int((side!=original).sum()),
         "reconstructed_opposite_low_note":"raw sell-taker trades, decision_time through endDate; original chosen_low retained when side unchanged"}
(SESSION/"joint_direction_btest_milestone.json").write_text(json.dumps(payload,indent=2,allow_nan=True),encoding="utf-8");print(json.dumps(payload,indent=2,allow_nan=True))
