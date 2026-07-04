# Arbor Research Contract

- Target: `C:\Users\ROG\Desktop\crypto_engine_version1`, starting from branch `pmdata`.
- Objective: optimize the complete direction/calibration/price-policy workflow until frozen `B_test sum_pnl > 500` under the existing forced-wrong-fill semantics.
- Baseline anchor: `20260619_expected_return_h14_h2_gc_gt_0p75`, `validation.sum_pnl = 27.44`.
- Development discipline: build chronological `B_dev` splits using data ending no later than 2026-04-10. The 2026-04-11 through 2026-05-10 month is frozen `B_test` and must not drive routine iteration.
- Scope: performance-first mixed research across direction, calibration, Gc, bid policy, new decision-time features, alternative model families, ensembles, and staged use of relevant skills from `.codex/skills_disabled`.
- Data: `artifacts/` and `price_estimator/data/` may be read. Raw data, existing reports, deploy artifacts, and live execution configuration are protected from modification.
- Leakage guard: `stage1_sample_weight` is forbidden as a feature and must fail feature-matrix validation; other label/future/sample-weight fields are default-deny pending lineage audit.
- Edit surface: isolated dated experiment code/config/report paths and Arbor session/worktree state. Internal experiment branches and commits are allowed; no merge or promotion into `pmdata` without explicit approval.
- Budget: persistent real run. Continue across sessions until the verified frozen B_test target is reached. GPU, dependency installation, and internet access are allowed.
- Deliverables: exact data universe and split windows, leakage audit, configs, predictions/artifacts, complete PnL/calibration/direction diagnostics, and comparison against 27.44.
