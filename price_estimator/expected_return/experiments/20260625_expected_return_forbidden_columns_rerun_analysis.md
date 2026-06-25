# 20260625 Expected Return Forbidden Columns Rerun Analysis

Run date: 2026-06-25

## Scope

Reran all 20 `price_estimator/expected_return/experiments/20260625_*` experiments with `features.forbidden_columns` present in each experiment config and enforced by `price_estimator/expected_return/run_lgbm_ev_policy.py`.

Primary comparison metric for this analysis: validation `sum_pnl`.

Coverage constraint: all rerun experiments satisfy `validation_metrics.coverage >= objective.min_coverage`.

## Leakage Control

Each experiment config now includes the expected return forbidden columns from `price_estimator/expected_return/config.yaml`.

The runner excludes both:

- columns matching `LEAKAGE_FEATURE_PATTERN`
- exact columns listed in `features.forbidden_columns`

The rerun reports record:

- `excluded_feature_forbidden_columns`
- `forbidden_columns_present_in_dataset`
- `feature_count`

Observed feature count after rerun: `1823` for every experiment.

Forbidden columns present in the source dataset but excluded from features:

```text
abs_return
chosen_low
chosen_low_reason
chosen_low_trade_time
correct
predicted_outcome
predicted_side
signed_return
target
target_raw
threshold_accepted
```

## Result Ranking

Sorted by rerun validation `sum_pnl` descending.

| Rank | Experiment | Before sum_pnl | After sum_pnl | Delta | Coverage | Selection score | Accepted acc | Signals | Orders |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `20260625_expected_return_xgb_lowbid_isotonic_no_leak` | 945.02 | 186.37 | -758.65 | 0.700054 | 0.641374 | 0.707345 | 5228 | 4373 |
| 2 | `20260625_expected_return_xgb_catboost_blend_fine_groups_no_leak` | 961.68 | 175.81 | -785.87 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3642 |
| 3 | `20260625_expected_return_xgb_lowbid_group_min_ev_no_leak` | 946.91 | 175.18 | -771.73 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3772 |
| 4 | `20260625_expected_return_xgb_lowbid_offset_group_no_leak` | 946.91 | 175.18 | -771.73 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3772 |
| 5 | `20260625_expected_return_xgb_catboost_blend_lowbid_group_no_leak` | 960.90 | 172.81 | -788.09 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3379 |
| 6 | `20260625_expected_return_xgb_catboost_blend_fine_lowbid_group_no_leak` | 169.19 | 169.19 | 0.00 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3576 |
| 7 | `20260625_expected_return_xgb_catboost_blend_finebid_no_leak` | 950.53 | 166.95 | -783.58 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3706 |
| 8 | `20260625_expected_return_xgb_catboost_blend_highbid_no_leak` | 961.84 | 164.64 | -797.20 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3544 |
| 9 | `20260625_expected_return_xgb_catboost_blend_qgate_no_leak` | 961.84 | 164.64 | -797.20 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3544 |
| 10 | `20260625_expected_return_xgb_lowbid_group_cal14_no_leak` | 929.23 | 163.02 | -766.21 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3567 |
| 11 | `20260625_expected_return_catboost_lgbm_blend_no_leak` | 949.67 | 160.38 | -789.29 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3581 |
| 12 | `20260625_expected_return_catboost_lowbid_group_no_leak` | 948.75 | 158.81 | -789.94 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3474 |
| 13 | `20260625_expected_return_xgb_lowbid_group_gciso_no_leak` | 930.72 | 154.97 | -775.75 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3725 |
| 14 | `20260625_expected_return_lgbm_lowbid_offset_group_no_leak` | 903.33 | 153.59 | -749.74 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3345 |
| 15 | `20260625_expected_return_catboost_deeper_lowbid_group_no_leak` | 938.88 | 152.79 | -786.09 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3728 |
| 16 | `20260625_expected_return_xgb_ev_isotonic_no_leak` | 943.70 | 151.65 | -792.05 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3796 |
| 17 | `20260625_expected_return_xgb_lowbid_group_cal3_no_leak` | 928.07 | 146.36 | -781.71 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3348 |
| 18 | `20260625_expected_return_xgb_lowbid_group_noiso_no_leak` | 936.91 | 143.92 | -792.99 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3486 |
| 19 | `20260625_expected_return_xgb_lowbid_group_qiso_no_leak` | 940.31 | 142.72 | -797.59 | 0.700054 | 0.641374 | 0.707345 | 5228 | 3575 |
| 20 | `20260625_expected_return_lgbm_ev_isotonic_no_leak` | 936.55 | 125.50 | -811.05 | 0.700054 | 0.641374 | 0.707345 | 5228 | 2913 |

## Selected Experiment

Best validation `sum_pnl` after forbidden-column enforcement:

```text
experiment_id: 20260625_expected_return_xgb_lowbid_isotonic_no_leak
report_path: price_estimator/expected_return/experiments/20260625_expected_return_xgb_lowbid_isotonic_no_leak/reports/summary_metrics.json
validation_sum_pnl: 186.37
validation_coverage: 0.7000535619
validation_selection_score: 0.6413740846
validation_utility: 0.2903053026
validation_accepted_sample_accuracy: 0.7073450650
validation_signal_count: 5228
validation_order_count: 4373
coverage_constraint_satisfied: true
feature_count: 1823
```

## Interpretation

The large drop in validation `sum_pnl` across nearly the full batch indicates that the previous EV/fill policy results were materially affected by columns now excluded through `features.forbidden_columns`.

Direction-selection metrics are identical across this batch after rerun because the same accepted signal set is being evaluated:

```text
coverage: 0.7000535619
selection_score: 0.6413740846
accepted_sample_accuracy: 0.7073450650
signal_count: 5228
```

The remaining differences are execution-policy differences: selected bid policy, order count, fill behavior, and realized PnL. Under the corrected no-leak feature set, `20260625_expected_return_xgb_lowbid_isotonic_no_leak` is the highest validation `sum_pnl` candidate.

