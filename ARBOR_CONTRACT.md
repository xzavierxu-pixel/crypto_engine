# Arbor Research Contract — G/M/S Research Plan 0706

- Target: `C:\Users\ROG\Desktop\crypto_engine_version1`, branch `2mins`.
- Objective: complete every required G, M, and S experiment in `docs/requirement0706.md`.
- Primary metric: maximize frozen `B_test sum_pnl`; limit-order anchor is `42.43`.
- Evaluation: use B_dev and rolling w1–w4 for selection, w5–w6 only as the holdout gate, and run each surviving candidate on B_test once.
- Fair comparisons: changed fill or holding semantics must be reported as separate tracks and compared with the anchor on the same universe and fills.
- Data discipline: all fitting and parameter selection precedes each evaluation window; B_test is read-only and never used for tuning.
- Scope: isolated session/experiment code, configs, reports, predictions, manifests, worktrees, and experiment commits. Long training, GPU use, package installation, internet access, and worktrees are authorized.
- Protected paths: source datasets, deploy artifacts, live execution configuration, accepted baselines, label semantics, and the user-authored requirement document.
- Promotion: no merge into `2mins`, artifact deployment, or live behavior change without separate user approval.
- Required outputs: each named experiment must satisfy section 6 of the plan and append its single B_test result to `gc_market_stop_btest_ledger.csv`.
- Budget: persistent real run; stop only after all mandatory nodes and all conditionally triggered stretch nodes are complete and audited, or at a genuine external blocker.
