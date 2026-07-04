#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
ROOT=Path(__file__).resolve().parents[3];SESSION=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('joint',ROOT/'.arbor/sessions/20260703_joint_q_policy/run_joint_q_policy.py')
joint=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(joint)
F=[f'w{i}' for i in range(1,7)]; REC=[14,21,28,999]
def prepare(f):
 d=SESSION/'folds'/f;train=pd.read_parquet(d/'data/train.parquet');dev=pd.read_parquet(d/'data/dev.parquet');cols,prep,grid,hazard=joint.load_hazard(d/'models/hazard_survival_cdf.pt')
 mask=dev.threshold_accepted.astype(bool).to_numpy();a=dev.loc[mask].copy();xdv=joint.matrix(a,cols);gc=joint.predict_hazard(hazard,prep.transform(dev),torch.device('cpu'),512)[1][mask];out={}
 atr=train.loc[train.threshold_accepted.astype(bool)].copy();ts=pd.to_datetime(atr.timestamp,utc=True)
 for days in REC:
  r=atr if days==999 else atr.loc[ts>=ts.max()-pd.Timedelta(days=days)].copy();x=joint.matrix(r,cols);y=r.correct.astype(int)
  l=LGBMClassifier(n_estimators=350,learning_rate=.025,num_leaves=15,max_depth=5,min_child_samples=100,colsample_bytree=.4,reg_lambda=10,reg_alpha=2,verbosity=-1,n_jobs=-1,random_state=20260703).fit(x,y)
  c=CatBoostClassifier(iterations=350,depth=5,learning_rate=.035,l2_leaf_reg=12,loss_function='Logloss',verbose=False,allow_writing_files=False,random_seed=20260703).fit(x,y)
  out[days]=.5*l.predict_proba(xdv)[:,1]+.5*c.predict_proba(xdv)[:,1]
 return dev,a,gc,grid,out
def ev(p,days,alpha,shrink,floor,mev):
 full,a,gc,grid,trees=p;q=(1-alpha)*a.p_side.to_numpy(float)+alpha*trees[days];q=(1-shrink)*q+shrink*.5
 bid,e,fill=joint.choose_survival_expected_return_bids(q,gc,grid,.01,mev,min_fill_probability=floor);return joint.backtest_metrics(a,joint.backtest_with_bid(a,bid,e,fill),len(full))
def main():
 p={f:prepare(f) for f in F};base={f:ev(p[f],999,0,0,.75,0)['sum_pnl'] for f in F};rows=[]
 for days in REC:
  for alpha in [.25,.5,.75,1]:
   for shrink in [0,.1,.2]:
    for floor in [.7,.75,.8,.85,.9]:
     for mev in [0,.005,.01,.02,.03,.05]:
      ms={f:ev(p[f],days,alpha,shrink,floor,mev) for f in F};pn={f:ms[f]['sum_pnl'] for f in F};de={f:pn[f]-base[f] for f in F};v=np.array([pn[f] for f in F[:4]])
      rows.append({'recency_days':days,'tree_alpha':alpha,'shrink':shrink,'gc_floor':floor,'min_ev':mev,**{f'{f}_pnl':pn[f] for f in F},**{f'{f}_delta':de[f] for f in F},
                   'tune_sum':v.sum(),'tune_worst':v.min(),'positive_weeks':int(sum(de[f]>0 for f in F[:4])),'tune_robust':float(v.sum()-v.std()),
                   'holdout_sum':pn['w5']+pn['w6'],'holdout_delta':de['w5']+de['w6']})
 t=pd.DataFrame(rows);e=t[t.positive_weeks>=3].sort_values(['tune_robust','tune_sum'],ascending=False);w=e.iloc[0]
 payload={'experiment_count':len(t),'baseline':base,'winner':w.to_dict(),'holdout_gate_passed':bool(w.holdout_delta>0 and w.w5_pnl>0 and w.w6_pnl>0),'top20':e.head(20).to_dict('records')}
 t.sort_values(['tune_robust','tune_sum'],ascending=False).to_csv(SESSION/'recency_q_search.csv',index=False);(SESSION/'recency_q_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
