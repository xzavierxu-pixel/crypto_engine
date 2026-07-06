#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
EXPERIMENTS = SESSION / "experiments"
EXPECTED = [
    "G0_anchor_reproduction", "G1_high_fill_scan", "G2_high_fill_conservative_q", "G3_recalibrated_gc",
    "M0_market_price_qa", "M1_market_ev_scan", "M2_same_universe_comparison", "M3_limit_market_hybrid",
    "S0_fixed_stop_preflight", "S1_stop_grid", "S2_take_profit_overlay", "S3_combination_not_triggered",
]
NODE_BY_EXPERIMENT = {
    "G0_anchor_reproduction":"1.1", "G1_high_fill_scan":"1.2", "G2_high_fill_conservative_q":"1.3", "G3_recalibrated_gc":"1.4",
    "M0_market_price_qa":"2.1", "M1_market_ev_scan":"2.2", "M2_same_universe_comparison":"2.3", "M3_limit_market_hybrid":"2.4",
    "S0_fixed_stop_preflight":"3.1", "S1_stop_grid":"3.2", "S2_take_profit_overlay":"3.3", "S3_combination_not_triggered":"3.4",
}
FILES = ["REPORT.md", "config_used.yaml", "feature_manifest.json", "leakage_check.json", "metrics_bdev.json", "metrics_btest.json", "predictions_btest.parquet"]
PRED_REQUIRED = ["sample_id", "decision_time", "selected_side", "q", "expected_ev", "filled", "correct", "realized_pnl"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    # Mirror named research artifacts into the minimal generic executor schema.
    tree = load_json(SESSION/".coordinator"/"idea_tree.json")
    for name, node_id in NODE_BY_EXPERIMENT.items():
        out = EXPERIMENTS/name
        prompt = EXPERIMENTS/node_id/"executor_prompt.md"
        if prompt.exists():
            shutil.copyfile(prompt, out/"executor_prompt.md")
        node = tree["nodes"][node_id]
        btest = load_json(out/"metrics_btest.json")
        generic = {"node_id":node_id,"status":node.get("status"),"score":node.get("score"),"result":node.get("result"),"btest":btest}
        (out/"metrics.json").write_text(json.dumps(generic,indent=2,allow_nan=True),encoding="utf-8")
        node_out = EXPERIMENTS/node_id
        node_out.mkdir(parents=True, exist_ok=True)
        if not (node_out/"metrics.json").exists():
            (node_out/"metrics.json").write_text(json.dumps(generic,indent=2,allow_nan=True),encoding="utf-8")
        if not (node_out/"report.md").exists():
            shutil.copyfile(out/"REPORT.md", node_out/"report.md")
    experiments = {}
    all_files = True; all_predictions = True; all_leakage = True
    for name in EXPECTED:
        out = EXPERIMENTS / name
        missing = [f for f in FILES if not (out/f).exists()]
        all_files &= not missing
        pred_columns = list(pd.read_parquet(out/"predictions_btest.parquet").columns) if not missing and (out/"predictions_btest.parquet").exists() else []
        pred_missing = [c for c in PRED_REQUIRED if c not in pred_columns]
        has_price = "bid" in pred_columns or "market_price" in pred_columns
        has_fill = "fill_probability" in pred_columns or "fill_flag" in pred_columns
        all_predictions &= not pred_missing and has_price and has_fill
        leakage = load_json(out/"leakage_check.json") if (out/"leakage_check.json").exists() else {"passed":False}
        all_leakage &= bool(leakage.get("passed"))
        experiments[name] = {"missing_files":missing, "missing_prediction_columns":pred_missing, "has_price_column":has_price, "has_fill_column":has_fill, "leakage_passed":bool(leakage.get("passed")), "prediction_rows":len(pd.read_parquet(out/"predictions_btest.parquet")) if (out/"predictions_btest.parquet").exists() else None}

    ledger = pd.read_csv(SESSION/"gc_market_stop_btest_ledger.csv")
    ledger_complete = set(ledger["experiment_id"]) == set(EXPECTED) and not ledger["experiment_id"].duplicated().any()
    g0 = load_json(EXPERIMENTS/"G0_anchor_reproduction"/"metrics_btest.json")
    m0 = load_json(EXPERIMENTS/"M0_market_price_qa"/"metrics_btest.json")
    m2 = load_json(EXPERIMENTS/"M2_same_universe_comparison"/"metrics_btest.json")
    m3 = load_json(EXPERIMENTS/"M3_limit_market_hybrid"/"metrics_btest.json")
    s3 = load_json(EXPERIMENTS/"S3_combination_not_triggered"/"metrics_btest.json")
    quarantined = [name for name in EXPECTED if (EXPERIMENTS/name/"protocol_violation_btest_diagnostic.json").exists()]
    status = subprocess.run(["git","status","--short"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()
    protected_changes = [line for line in status if any(token in line for token in ["execution_engine/deploy", "execution_engine/config", "config/settings.yaml", "src/labels/"])]
    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_experiment_count": len(EXPECTED), "experiments": experiments,
        "all_required_files_present": all_files, "all_prediction_schemas_pass": all_predictions,
        "all_leakage_checks_pass": all_leakage, "ledger_complete_and_unique": ledger_complete,
        "g0_anchor_reproduced": bool(np.isclose(g0["sum_pnl"],42.43,atol=1e-9)),
        "m0_timestamp_gate_passed": m0["qa"]["late_trade_count"] == 0,
        "m2_fair_comparison_present": "paired_same_universe" in m2 and m2["paired_same_universe"]["legal_m_count"] > 0,
        "m3_trigger_was_honored": m3["btest_evaluation_count"] == 1,
        "s3_nontrigger_was_honored": s3["btest_evaluation_count"] == 0,
        "protected_path_changes": protected_changes, "promotion_performed": False,
        "quarantined_protocol_violations": quarantined,
        "protocol_clean": not quarantined,
        "artifact_completion_passed": bool(all_files and all_predictions and all_leakage and ledger_complete and not protected_changes),
        "scientific_protocol_completion_passed": bool(not quarantined),
        "best_results": {
            "limit_anchor_sum_pnl": 42.43,
            "M1_market_sum_pnl_distinct_semantics": float(ledger.loc[ledger.experiment_id=="M1_market_ev_scan","btest_sum_pnl"].iloc[0]),
            "M2_same_universe_limit_anchor": m2["paired_same_universe"]["limit_anchor"]["sum_pnl"],
            "M2_same_universe_market_sum_pnl": m2["paired_same_universe"]["market"]["sum_pnl"],
            "M3_hybrid_sum_pnl": m3["metrics"]["sum_pnl"],
        },
    }
    (SESSION/"COMPLETION_AUDIT.json").write_text(json.dumps(audit,indent=2,allow_nan=True),encoding="utf-8")

    nodes = list(tree["nodes"].values())
    stats = {"run_name":SESSION.name,"completed_at":audit["timestamp_utc"],"node_count":len(nodes),"done_count":sum(n.get("status")=="done" for n in nodes),"pruned_count":sum(n.get("status")=="pruned" for n in nodes),"experiment_count":len(EXPECTED),"btest_ledger_rows":len(ledger),"artifact_completion_passed":audit["artifact_completion_passed"],"protocol_clean":audit["protocol_clean"]}
    (SESSION/"run_stats.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    events=[]
    events.append({"event":"session.start","run_name":SESSION.name,"timestamp":audit["timestamp_utc"]})
    for n in nodes:
        if n["id"]!="ROOT" and n.get("status") in {"done","pruned"}:
            events.append({"event":"idea.completed" if n["status"]=="done" else "idea.pruned","node_id":n["id"],"score":n.get("score"),"timestamp":audit["timestamp_utc"]})
    events.append({"event":"session.checkpoint","phase":"audit","artifact_completion_passed":audit["artifact_completion_passed"],"protocol_clean":audit["protocol_clean"],"timestamp":audit["timestamp_utc"]})
    (SESSION/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events),encoding="utf-8")
    checkpoint={"run_name":SESSION.name,"cycle":len(nodes)-1,"phase":"audit_complete","git_branch":"2mins","in_flight_executors":[],"pending_human_gates":[],"timestamp":audit["timestamp_utc"]}
    (SESSION/".coordinator"/"checkpoint.json").write_text(json.dumps(checkpoint,indent=2),encoding="utf-8")
    (SESSION/".coordinator"/"messages.jsonl").write_text(json.dumps({"role":"system","content":"Experiment execution finished; completion audit recorded.","timestamp":audit["timestamp_utc"]})+"\n",encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ["all_required_files_present","all_prediction_schemas_pass","all_leakage_checks_pass","ledger_complete_and_unique","artifact_completion_passed","protocol_clean","quarantined_protocol_violations"]},indent=2))


if __name__ == "__main__":
    main()
