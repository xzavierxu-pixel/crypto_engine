# Experiment T0 - frozen trade/L2 baseline reproduction

## Hypothesis

Exact node 1/1.1 code, data, and frozen policy reuse distinguishes genuine rolling reproducibility from implementation drift.

## What Changed

No modeling mechanism changed. This run added reproducible orchestration, leakage checks, diagnostics, and exactly-once B_test accounting.

## What Did Not Change

- Direction artifact: accepted legacy direction columns in the frozen datasets
- Calibration method: node 1 `oldnew_trade_lgbm_blend`
- Fill semantics: correct may not fill; every wrong submitted order is forced filled
- Accepted universe: `threshold_accepted == true`

## Data Windows

- Train / B_dev: chronological rolling w1-w6 under `C:\Users\ROG\Desktop\crypto_engine_version1\.arbor\sessions\20260703_prefinal_rolling\folds`
- Calibration: contained in each frozen training artifact; no evaluation rows fitted
- B_test: frozen 7,468-row validation / 5,228 accepted rows

## Leakage Check

- Forbidden columns intersection: []
- Feature cutoff check: trade_time <= decision_time
- Fit-on-validation check: passed

## Metrics

| Split | sum_pnl | delta_vs_anchor | order_count | order_coverage | win_pnl | loss_pnl | fill_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_dev w1-w4 | 290.31 | n/a | n/a | n/a | n/a | n/a | n/a |
| B_test | 14.78 | -27.65 | 2842 | 0.543611 | 502.42 | -487.64 | 0.245168 |

## B_test Required Result

- btest_sum_pnl: 14.78
- btest_delta_vs_42p43: -27.65
- btest_wrong_submitted_rate: 0.284659
- btest_avg_loser_bid: 0.602769
- btest_submitted_fill_calibration_gap: 0.245168

## Diagnosis

The frozen rolling result reproduced. B_test underperforms the 42.43 anchor. q Brier moved from rolling mean 0.189884 to 0.205080; Gc Brier moved from 0.125942 to 0.141787; wrong-submitted rate rose from 0.195228 to 0.284659. The dominant realized failure is loss exposure (-487.64) amplified by Gc/submitted-fill overconfidence.

## Decision

- Continue / modify / stop: continue to T1 audit; do not promote T0 automatically
- Reason: T0 is the required frozen baseline and B_test diagnostic, not a new candidate.
