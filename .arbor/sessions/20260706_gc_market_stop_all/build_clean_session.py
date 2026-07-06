#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
TARGET = SOURCE.parent / "20260706_gc_market_stop_all_clean"

if TARGET.exists():
    raise RuntimeError(f"refusing to overwrite existing clean session: {TARGET}")
TARGET.mkdir(parents=True)

# Re-run G1-G3 in the clean session using only rolling prepared caches. Deliberately
# do not copy the frozen B_test prepared cache, so failed gates cannot access it.
shutil.copyfile(SOURCE/"run_g_series.py", TARGET/"run_g_series.py")
cache_out = TARGET/"cache"/"g_prepared"; cache_out.mkdir(parents=True)
for fold in [f"w{i}" for i in range(1,7)]:
    shutil.copyfile(SOURCE/"cache"/"g_prepared"/f"{fold}.pkl", cache_out/f"{fold}.pkl")

# G0 is an authorized fixed anchor replay. Other G outputs are regenerated later.
shutil.copytree(SOURCE/"experiments"/"G0_anchor_reproduction", TARGET/"experiments"/"G0_anchor_reproduction")

# Copy protocol-valid M/S results and all executor state artifacts.
for name in [
    "M0_market_price_qa","M1_market_ev_scan","M2_same_universe_comparison","M3_limit_market_hybrid",
    "S0_fixed_stop_preflight","S1_stop_grid","S2_take_profit_overlay","S3_combination_not_triggered",
]:
    shutil.copytree(SOURCE/"experiments"/name, TARGET/"experiments"/name)
for node_dir in ["1.1","1.2","1.3","1.4","2.1","2.2","2.3","2.4","3.1","3.2","3.3","3.4"]:
    shutil.copytree(SOURCE/"experiments"/node_dir, TARGET/"experiments"/node_dir)

for name in ["run_g0_anchor.py","run_s_series.py","run_m_series.py","run_m3_hybrid.py","finalize_conditionals.py","audit_session.py"]:
    shutil.copyfile(SOURCE/name, TARGET/name)

# Keep only valid or correctly skipped ledger rows; G1-G3 will be added by the
# corrected runner with zero B_test reads.
ledger = pd.read_csv(SOURCE/"gc_market_stop_btest_ledger.csv")
ledger = ledger.loc[~ledger["experiment_id"].isin(["G1_high_fill_scan","G2_high_fill_conservative_q","G3_recalibrated_gc"])]
ledger.to_csv(TARGET/"gc_market_stop_btest_ledger.csv", index=False)

# Copy Arbor state, removing references to quarantined diagnostic values from the
# authoritative clean tree. Scores remain B_dev scores.
shutil.copytree(SOURCE/".coordinator", TARGET/".coordinator")
tree_path = TARGET/".coordinator"/"idea_tree.json"
tree = json.loads(tree_path.read_text(encoding="utf-8"))
clean_notes = {
    "1.2": ("Nominal Gc>=0.90 improved only one of four tune weeks, so G1 failed selection and B_test was not read.", "G1 failed w1-w4; B_test skipped."),
    "1.3": ("q shrinkage did not win; shrink=0 remained best and only one of four tune weeks improved, so B_test was not read.", "G2 failed w1-w4; B_test skipped."),
    "1.4": ("The empirical blend improved Gc Brier but zero of four tune weeks improved PnL, so B_test was not read.", "G3 failed w1-w4; B_test skipped."),
}
for node_id,(insight,result) in clean_notes.items():
    tree["nodes"][node_id]["insight"] = insight
    tree["nodes"][node_id]["result"] = result
tree["nodes"]["1"]["insight"] = "G0 reproduced 42.43. G1-G3 all failed the required w1-w4 selection gate and did not read B_test."
tree["nodes"]["ROOT"]["insight"] = "All G/M/S nodes completed under the corrected evidence set. M1/M2 market orders and M3 hybrid passed; no promotion was performed."
tree_path.write_text(json.dumps(tree,indent=2,allow_nan=True),encoding="utf-8")

# The source report's invalid diagnostics remain disclosed as provenance, but are
# not part of this clean session's selection or result ledger.
(TARGET/"PROVENANCE.md").write_text(
    "# Clean-session provenance\n\n"
    "This session is the authoritative corrected evidence package. G1-G3 were rerun from rolling w1-w6 caches without a B_test cache and correctly stopped at the failed tune gate. "
    "Protocol-valid G0, M0-M3, and S0-S3 artifacts were migrated byte-for-byte from `20260706_gc_market_stop_all`; their parameters had been selected without B_test and each evaluated candidate had one authorized B_test read. "
    "The source session retains three quarantined G diagnostics from an implementation fallback bug; they are excluded here and were not used by any selection.\n",
    encoding="utf-8",
)

print(TARGET)
