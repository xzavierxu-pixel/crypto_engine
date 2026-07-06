#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
OUT = SESSION / "experiments" / "M3_limit_market_hybrid"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


m = load_module("m_shared_m3", SESSION / "run_m_series.py")
g = m.gmod
joint = m.joint


def hybrid(prepared, mframe, edge: float, max_limit_fill: float):
    anchor_metrics, base, q = m.anchor(prepared)
    accepted = prepared[1].reset_index(drop=True)
    market_price = mframe["market_price"].to_numpy(float)
    legal = np.isfinite(market_price)
    market_ev = q - market_price
    use_market = legal & (market_ev > edge) & (base.fill_prob < max_limit_fill)
    bid = base.bid.copy(); bid[use_market] = market_price[use_market]
    expected_ev = base.expected_ev.copy(); expected_ev[use_market] = market_ev[use_market]
    fill_prob = base.fill_prob.copy(); fill_prob[use_market] = 1.0
    filled = base.filled.copy(); filled[use_market] = True
    printed = base.printed_filled.copy(); printed[use_market] = True
    correct = accepted["correct"].astype(bool).to_numpy()
    pnl = base.pnl.copy(); pnl[use_market] = np.where(correct[use_market], 1.0-market_price[use_market], -market_price[use_market])
    result = m.BacktestResult(bid, expected_ev, fill_prob, pnl, filled, printed)
    metrics = joint.backtest_metrics(accepted, result, len(prepared[0]))
    submitted = bid > 0; correct_submitted = correct & submitted
    metrics.update({
        "q_brier": float(brier_score_loss(correct.astype(int), q)),
        "gc_brier": float(brier_score_loss(filled[correct_submitted].astype(int), fill_prob[correct_submitted])) if correct_submitted.any() else float("nan"),
        "wrong_submitted_count": int((~correct & submitted).sum()),
        "avg_loser_cost": float(bid[~correct & submitted].mean()) if (~correct & submitted).any() else float("nan"),
        "market_order_count": int(use_market.sum()), "limit_order_count": int((submitted & ~use_market).sum()),
        "market_order_share": float(use_market.sum()/submitted.sum()) if submitted.any() else float("nan"),
    })
    return metrics, result, q, use_market, market_ev, anchor_metrics


def main() -> None:
    data = {}
    for fold in g.FOLDS:
        prepared = g.load_prepared(fold)
        data[fold] = (prepared, m.build_market_prices(fold, prepared))
    rows, detail = [], {}
    for edge in (0.05, 0.075, 0.10):
        for ceiling in (0.75, 0.80, 0.85):
            metrics = {f: hybrid(*data[f], edge, ceiling)[0] for f in g.TUNE}
            pnl = {f:metrics[f]["sum_pnl"] for f in g.TUNE}; tune=np.asarray([pnl[f] for f in g.TUNE])
            key=f"{edge}|{ceiling}"
            rows.append({"market_edge":edge,"max_limit_fill":ceiling,**{f"{f}_pnl":pnl[f] for f in g.TUNE},"tune_sum":float(tune.sum()),"tune_std":float(tune.std()),"tune_robust":float(tune.sum()-tune.std())})
            detail[key]=metrics
    table=pd.DataFrame(rows).sort_values(["tune_robust","tune_sum"],ascending=False); winner=table.iloc[0].to_dict(); key=f"{winner['market_edge']}|{winner['max_limit_fill']}"
    for fold in g.HOLDOUT:
        detail[key][fold] = hybrid(*data[fold], float(winner["market_edge"]), float(winner["max_limit_fill"]))[0]
        winner[f"{fold}_pnl"] = detail[key][fold]["sum_pnl"]
    winner["holdout_sum"] = float(sum(winner[f"{f}_pnl"] for f in g.HOLDOUT))
    winner["holdout_both_positive"] = bool(all(winner[f"{f}_pnl"] > 0 for f in g.HOLDOUT))
    gate=bool(winner["holdout_sum"]>0 and winner["holdout_both_positive"])
    OUT.mkdir(parents=True,exist_ok=True); table.to_csv(OUT/"search.csv",index=False)
    bdev={"winner":winner,"winner_fold_metrics":detail[key],"holdout_gate_passed":gate,"prerequisite":"M2 positive edge passed"}
    prepared=g.load_prepared("btest"); mf=m.build_market_prices("btest",prepared)
    if gate:
        metrics,result,q,use_market,market_ev,anchor_metrics=hybrid(prepared,mf,float(winner["market_edge"]),float(winner["max_limit_fill"])); metrics["anchor_delta"]=metrics["sum_pnl"]-anchor_metrics["sum_pnl"]; metrics["btest_evaluation_count"]=1
        pred=m.predictions(prepared,mf,q,market_ev,result); pred["order_type"]=np.where(use_market,"market",np.where(result.bid>0,"limit","abstain")); pred["realized_pnl"]=result.pnl
        status="evaluated_once"; score=metrics["sum_pnl"]
    else:
        metrics={"status":"skipped_holdout_gate_failed","sum_pnl":None,"btest_evaluation_count":0}; pred=pd.DataFrame(columns=["sample_id","decision_time","selected_side","q","bid","market_price","expected_ev","filled","correct","realized_pnl","order_type"]); status="skipped"; score=None; anchor_metrics={"sum_pnl":42.43}
    config=f"experiment_id: M3_limit_market_hybrid\nmarket_edge: {winner['market_edge']}\nmax_limit_fill: {winner['max_limit_fill']}\nq_model: raw_tree_blend\n"
    report=f"# M3 limit/market hybrid\n\nSelected edge={winner['market_edge']} and max anchor fill={winner['max_limit_fill']} on w1-w4. Holdout gate `{gate}`. B_test status `{status}`; sum_pnl `{score}` versus full-universe anchor `{anchor_metrics['sum_pnl']}`.\n"
    m.artifacts(OUT,prepared,config,bdev,{"metrics":metrics,"sum_pnl":score,"btest_evaluation_count":metrics["btest_evaluation_count"]},pred,report)
    ledger=SESSION/"gc_market_stop_btest_ledger.csv"; old=pd.read_csv(ledger); old=old.loc[old.experiment_id!="M3_limit_market_hybrid"]
    row={"experiment_id":"M3_limit_market_hybrid","track":"M","btest_status":status,"btest_sum_pnl":score,"anchor_same_universe":anchor_metrics["sum_pnl"],"delta":None if score is None else score-anchor_metrics["sum_pnl"],"btest_reads":int(gate),"holdout_gate_passed":gate}
    pd.concat([old,pd.DataFrame([row])],ignore_index=True,sort=False).to_csv(ledger,index=False)
    print(json.dumps(row,indent=2,allow_nan=True))


if __name__ == "__main__":
    main()
