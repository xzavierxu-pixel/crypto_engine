# Two-minute decision-time validation proposal

- Decision timestamp: `market_t0 + 2 minutes`.
- Feature cutoff: feature rows are aligned to the two-minute decision timestamp; source builders must not consume observations after that timestamp.
- Price target: selected-side minimum sell-taker price on `[decision_time, endDate]` (`include_start=true`, `include_end=true`).
- Direction model: retrain the accepted CatBoost calendar model on shifted features, while freezing the accepted UTC day/session threshold policy from `execution_engine/deploy/baseline/artifact_manifest.json`.
- Calibration/policy: retain `raw_tree_blend`, `shrink=0`, `Gc floor=0.85`, and `min_ev=0.02`.
- Validation protocol: preserve the six chronological pretest folds; w1-w4 are diagnostics and w5-w6 are the untouched pretest gate. Evaluate the original validation/B_test only after the gate passes.
- Fill semantics: a correct order fills when `winner_low <= bid`; every submitted wrong order is forced filled.
- Leakage guards: reject forbidden target/future/identifier columns and `stage1_sample_weight` from model features. Fit q and Gc models only on rows preceding each validation window.
- Output isolation: write only below `.arbor/sessions/20260704_two_minute_shift`; existing data, deploy artifacts, configs, and reports are read-only.
- Promotion: none. Report the resulting validation `sum_pnl`; promotion requires a separate user decision.

