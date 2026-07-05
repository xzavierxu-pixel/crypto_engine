# Arbor Research Contract — Trade/L2 T0–T9

- Target: `C:\Users\ROG\Desktop\crypto_engine_version1`, based on branch `2mins` at `a2f1528`.
- Objective: completely implement and evaluate T0–T9 from `docs/trade l2 optimization-0705.md`.
- Primary metric: maximize frozen `B_test sum_pnl`; anchor is `42.43`.
- Evaluation: every named experiment must record B_dev, rolling w1–w6, and B_test, as explicitly required by the design document.
- Fill semantics: correct orders fill only when `winner_low <= bid`; every submitted wrong order is forced filled; printed wrong-side fill is diagnostic only.
- Data discipline: fit models, calibration maps, abstention tables, gates, and posterior bounds only on chronological train/calibration/B_dev data earlier than their evaluation window. B_test is evaluation-only and must not tune any candidate.
- Scope: isolated experiment/session code, configs, reports, predictions, manifests, and Arbor branches/worktrees. Long training, GPU use, package installation, internet access, worktrees, and experiment commits are authorized.
- Protected paths: raw data, accepted/deployed artifacts, live execution configuration, label/fill semantics, existing historical reports, and the user-authored design document.
- Promotion: no merge into `2mins`, deploy artifact replacement, or live behavior change without separate user approval.
- Required per-experiment outputs: `REPORT.md`, `config_used.yaml`, `feature_manifest.json`, `leakage_check.json`, `metrics_bdev.json`, `metrics_btest.json`, `predictions_btest.parquet`, plus the session `trade_l2_btest_ledger.csv`.
- Budget: persistent real run with no fixed cycle or wall-clock cap; stop only after T0–T9 and the completion audit are finished, or at a genuine external blocker.

