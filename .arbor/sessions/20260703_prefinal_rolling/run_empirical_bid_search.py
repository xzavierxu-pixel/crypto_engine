#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];SESSION=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("joint",ROOT/".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
joint=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(joint)
F=[f"w{i}" for i in range(1,7)]
frames={f:pd.read_parquet(SESSION/"folds"/f/"data/dev.parquet") for f in F}
def score(df,bid):
 a=df.loc[df.threshold_accepted.astype(bool)].copy();r=joint.backtest_with_bid(a,bid(a),None,None);return joint.backtest_metrics(a,r,len(df))
rows=[]
for qmin in np.round(np.arange(.50,.81,.025),3):
 for fixed in np.round(np.arange(.10,.81,.025),3):
  ms={f:score(frames[f],lambda a,q=qmin,b=fixed:np.where(a.p_side.to_numpy(float)>=q,b,0.)) for f in F};v=np.array([ms[f]["sum_pnl"] for f in F[:4]])
  rows.append({"family":"fixed","q_min":qmin,"bid":fixed,"slope":0.,**{f"{f}_pnl":ms[f]["sum_pnl"] for f in F},
               "tune_sum":v.sum(),"tune_worst":v.min(),"tune_robust":float(v.sum()-v.std()),"holdout_sum":ms['w5']['sum_pnl']+ms['w6']['sum_pnl']})
for qmin in np.round(np.arange(.50,.76,.05),3):
 for slope in [.5,.75,1.,1.25]:
  for offset in [-.25,-.15,-.05,.05]:
   def fn(a,q=qmin,s=slope,o=offset):
    p=a.p_side.to_numpy(float);b=np.clip(np.round((s*p+o)*100)/100,.01,.85);return np.where(p>=q,b,0.)
   ms={f:score(frames[f],fn) for f in F};v=np.array([ms[f]["sum_pnl"] for f in F[:4]])
   rows.append({"family":"affine_q","q_min":qmin,"bid":offset,"slope":slope,**{f"{f}_pnl":ms[f]["sum_pnl"] for f in F},
                "tune_sum":v.sum(),"tune_worst":v.min(),"tune_robust":float(v.sum()-v.std()),"holdout_sum":ms['w5']['sum_pnl']+ms['w6']['sum_pnl']})
t=pd.DataFrame(rows).sort_values(["tune_robust","tune_sum"],ascending=False);w=t.iloc[0]
payload={"experiment_count":len(t),"winner":w.to_dict(),"holdout_gate_passed":bool(w.holdout_sum>0 and w.w5_pnl>0 and w.w6_pnl>0),"top20":t.head(20).to_dict('records')}
t.to_csv(SESSION/"empirical_bid_search.csv",index=False);(SESSION/"empirical_bid_summary.json").write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
