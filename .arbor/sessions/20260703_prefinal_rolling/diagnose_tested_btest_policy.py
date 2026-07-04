#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3]; SESSION=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("joint",ROOT/".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
joint=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(joint)
p=joint.prepare(ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_train.parquet",
                ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data/expected_return_validation.parquet",
                ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_h2_hazard_smooth/models/hazard_survival_cdf.pt",20260703)
full,a,qs,gc,grid,_,_=p
def run(q,floor,mev):
 b,e,fp=joint.choose_survival_expected_return_bids(q,gc,grid,.01,mev,min_fill_probability=floor)
 return joint.backtest_with_bid(a,b,e,fp)
results={"baseline":run(qs["raw"],.75,0),"rolling_winner":run(qs["raw_tree_blend"],.85,.02)}
ts=pd.to_datetime(a.timestamp,utc=True); start=pd.Timestamp("2026-04-11",tz="UTC")
week=((ts-start).dt.days//7).clip(0,3)
rows=[]
for name,r in results.items():
 for w in range(4):
  m=week.eq(w).to_numpy(); submitted=r.bid[m]>0; correct=a.correct.astype(bool).to_numpy()[m]
  rows.append({"policy":name,"week":w+1,"start":str((start+pd.Timedelta(days=7*w)).date()),"rows":int(m.sum()),
               "accuracy":float(correct.mean()),"orders":int(submitted.sum()),"trades":int(r.filled[m].sum()),
               "sum_pnl":float(r.pnl[m].sum()),"win_pnl":float(r.pnl[m][r.pnl[m]>0].sum()),
               "loss_pnl":float(r.pnl[m][r.pnl[m]<0].sum()),"mean_bid_submitted":float(r.bid[m][submitted].mean()) if submitted.any() else np.nan})
pd.DataFrame(rows).to_csv(SESSION/"tested_btest_weekly_diagnostics.csv",index=False)
(SESSION/"tested_btest_weekly_diagnostics.json").write_text(json.dumps(rows,indent=2),encoding="utf-8");print(json.dumps(rows,indent=2))
