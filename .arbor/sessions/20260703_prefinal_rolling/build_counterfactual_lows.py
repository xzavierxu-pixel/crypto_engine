#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SESSION=Path(__file__).resolve().parent
DATA=ROOT/"price_estimator/expected_return/experiments/20260619_expected_return_trade_coverage_start/data"
TRADES=ROOT/"price_estimator/data/sell_taker_trades_daily"

def enrich(frame:pd.DataFrame)->pd.DataFrame:
    refs=frame[["condition_id","decision_time","endDate"]].drop_duplicates("condition_id").copy()
    refs["decision_time"]=pd.to_datetime(refs.decision_time,utc=True)
    refs["endDate"]=pd.to_datetime(refs.endDate,utc=True)
    pieces=[]
    dates=pd.to_datetime(frame.decision_time,utc=True).dt.strftime("%Y-%m-%d").unique()
    # Files can contain markets straddling UTC midnight, so include the following day.
    dates=sorted(set(dates)|{(pd.Timestamp(x)+pd.Timedelta(days=1)).strftime("%Y-%m-%d") for x in dates})
    for date in dates:
        path=TRADES/f"date={date}.parquet"
        if not path.exists(): continue
        t=pd.read_parquet(path,columns=["condition_id","outcome","price","trade_time"])
        t=t.merge(refs,on="condition_id",how="inner")
        if t.empty: continue
        t["trade_time"]=pd.to_datetime(t.trade_time,utc=True)
        t=t.loc[(t.trade_time>=t.decision_time)&(t.trade_time<=t.endDate)]
        t["outcome"]=t.outcome.str.lower()
        pieces.append(t.groupby(["condition_id","outcome"],as_index=False).price.min())
    lows=pd.concat(pieces,ignore_index=True).groupby(["condition_id","outcome"],as_index=False).price.min()
    wide=lows.pivot(index="condition_id",columns="outcome",values="price").rename(columns={"up":"up_low","down":"down_low"})
    out=frame.merge(wide,on="condition_id",how="left")
    selected=np.where(out.selected_side.str.upper().eq("UP"),out.up_low,out.down_low)
    valid=np.isfinite(selected)&np.isfinite(out.chosen_low)
    report={"rows":len(out),"up_low_available":int(out.up_low.notna().sum()),"down_low_available":int(out.down_low.notna().sum()),
            "both_available":int((out.up_low.notna()&out.down_low.notna()).sum()),
            "selected_low_exact_match_rate":float(np.isclose(selected[valid],out.loc[valid,"chosen_low"],atol=1e-9).mean()),
            "selected_low_mean_abs_error":float(np.abs(selected[valid]-out.loc[valid,"chosen_low"]).mean())}
    return out,report

def main():
    reports={}
    for name,file in [("pretest","expected_return_train.parquet"),("btest","expected_return_validation.parquet")]:
        out,report=enrich(pd.read_parquet(DATA/file))
        out.to_parquet(SESSION/f"{name}_both_side_lows.parquet",index=False)
        reports[name]=report
    (SESSION/"counterfactual_low_build.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
    print(json.dumps(reports,indent=2))
if __name__=="__main__":main()
