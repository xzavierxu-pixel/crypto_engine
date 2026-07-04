#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SESSION = Path(__file__).resolve().parent
TOOLS = Path(r"C:\Users\ROG\.codex\skills\arbor-agent-tools\scripts\arbor_state.py")
BASE = [sys.executable, str(TOOLS), "--cwd"]


def call(command: str, *args: str) -> str:
    cmd = [sys.executable, str(TOOLS), command, "--cwd", str(ROOT), "--run-name", "20260703_sum_pnl_100", *args]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def main() -> None:
    table = pd.read_csv(SESSION / "dev_search_100.csv").sort_values("experiment_id")
    for _, row in table.iterrows():
        exp_id = int(row.experiment_id)
        hypothesis = "\n".join([
            f"Mechanism: Joint policy candidate {exp_id} with q-alpha={row.q_isotonic_alpha:g}, Gc-floor={row.candidate_gc_floor:g}, min-EV={row.min_ev:g}",
            "Hypothesis: This interaction may control calibration bias, fill confidence, and admission risk more robustly than one-axis selection.",
            "Observable: Positive sum_pnl delta on both chronological B_dev folds, ranked by worst-fold delta then mean delta.",
            "Conflicts: performance-first grid member under the robust joint-search parent; retained for complete 100-run evidence.",
        ])
        output = call("add", "--parent-id", "1", "--hypothesis", hypothesis)
        node_id = output.split("Added node ", 1)[1].split(" ", 1)[0]
        result = (
            f"main={row.main_sum_pnl:.2f} (delta {row.main_delta:+.2f}); "
            f"earlier={row.earlier_sum_pnl:.2f} (delta {row.earlier_delta:+.2f}); "
            f"worst_delta={row.worst_delta:+.2f}"
        )
        insight = "Robust winner across both folds." if exp_id == 37 else "Recorded as part of the complete two-fold interaction search."
        report = f"Idea: experiment {exp_id}\nBaseline vs Result: {result}\nScore: {row.main_sum_pnl:.2f}\nInsights: {insight}"
        call("record", "--node-id", node_id, "--score", str(row.main_sum_pnl), "--result", result, "--insight", insight, "--raw-report", report)
    call(
        "update", "--node-id", "1", "--status", "done", "--score", "98.1",
        "--result", "100 candidates completed; experiment 37 selected by worst-fold delta; frozen B_test=5.48 vs 27.44 baseline.",
        "--insight", "Joint search found a two-fold B_dev winner, but it failed the frozen month; no candidate is promoted.",
    )


if __name__ == "__main__":
    main()
