#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];SESSION=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("joint",ROOT/".arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py")
joint=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(joint)
F=[f"w{i}" for i in range(1,7)]
prep={f:joint.prepare(SESSION/"folds"/f/"data/train.parquet",SESSION/"folds"/f/"data/dev.parquet",SESSION/"folds"/f/"models/hazard_survival_cdf.pt",20260703) for f in F}
def eval_fold(p,lam,floor,mev):
 full,a,qs,gc,grid,_,_=p;base=qs['raw_tree_blend'];dis=np.std(np.c_[qs['raw'],qs['lgbm'],qs['catboost']],axis=1);q=np.clip(base-lam*dis,.5,.999)
 bid,ev,fill=joint.choose_survival_expected_return_bids(q,gc,grid,.01,mev,min_fill_probability=floor)
 return joint.backtest_metrics(a,joint.backtest_with_bid(a,bid,ev,fill),len(full))
baseline={f:joint.evaluate(prep[f],'raw',0,.75,0)['sum_pnl'] for f in F};rows=[]
for lam in [0,.25,.5,.75,1,1.5,2,3]:
 for floor in [.7,.75,.8,.85,.9]:
  for mev in [0,.005,.01,.02,.03,.05]:
   ms={f:eval_fold(prep[f],lam,floor,mev) for f in F};pnl={f:ms[f]['sum_pnl'] for f in F};delta={f:pnl[f]-baseline[f] for f in F};v=np.array([pnl[f] for f in F[:4]])
   rows.append({'uncertainty_penalty':lam,'gc_floor':floor,'min_ev':mev,**{f'{f}_pnl':pnl[f] for f in F},**{f'{f}_delta':delta[f] for f in F},
                'tune_sum':v.sum(),'tune_worst':v.min(),'tune_positive_delta_weeks':int(sum(delta[f]>0 for f in F[:4])),
                'tune_robust':float(v.sum()-v.std()),'holdout_sum':pnl['w5']+pnl['w6'],'holdout_delta':delta['w5']+delta['w6']})
t=pd.DataFrame(rows);e=t[t.tune_positive_delta_weeks>=3].sort_values(['tune_robust','tune_sum'],ascending=False);w=e.iloc[0]
payload={'experiment_count':len(t),'baseline':baseline,'winner':w.to_dict(),'holdout_gate_passed':bool(w.holdout_delta>0 and w.w5_pnl>0 and w.w6_pnl>0),'top20':e.head(20).to_dict('records')}
t.sort_values(['tune_robust','tune_sum'],ascending=False).to_csv(SESSION/'uncertainty_policy_search.csv',index=False);(SESSION/'uncertainty_policy_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
