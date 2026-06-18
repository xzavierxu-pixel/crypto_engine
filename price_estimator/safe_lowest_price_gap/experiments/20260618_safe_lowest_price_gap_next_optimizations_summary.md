# Safe Lowest Price Gap Next Optimizations Summary

Date: 2026-06-18

Primary metric: validation `active_covered_gap_norm_mean`, subject to `coverage_feasible >= 0.70` and `side_violation_rate == 0`.

Deploy training mode: `offline_experiment_only`

Offline validation metric source: local `price_estimator/data` train/validation datasets.

Deploy artifact update: not performed.

## Results

| experiment | change | alpha | delta_mode | validation active gap | coverage_feasible | non_active_share | side_violation_rate | covered_feasible_count | val - train active gap |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 20260618_r2_cov70_active | baseline | 0.5 | global | 0.516326182 | 0.746951791 | 0.404063509 | 0.0 | 3982 | 0.040222428 |
| 20260618_r6_low_alpha | lower alpha grid | 0.01 | global | 0.361100665 | 0.756143313 | 0.338132400 | 0.0 | 4031 | 0.021783411 |
| 20260618_r7_low_alpha_regularized | R6 + dropout/weight_decay | 0.01 | global | 0.367159422 | 0.713187019 | 0.287540366 | 0.0 | 3802 | 0.028264917 |
| 20260618_r8_pside_bin_delta | R7 + p_side-bin delta | 0.01 | pside_bin | 0.352617037 | 0.732695554 | 0.435548977 | 0.0 | 3906 | 0.030527165 |
| 20260618_r9_sfloor005_room_weight | R7 + s_floor 0.05 + room weight | 0.01 | global | 0.336995717 | 0.735696867 | 0.352798708 | 0.0 | 3922 | 0.016504496 |
| 20260618_r10_low_alpha_filter_off | R6 filter off control | 0.01 | global | 0.361100665 | 0.756143313 | 0.338132400 | 0.0 | 4031 | 0.021783411 |
| 20260618_r10_low_alpha_filter_on | R6 filter on A/B | 0.01 | global | 0.367110094 | 0.795211411 | 0.399112654 | 0.0 | 3122 | 0.009698641 |
| 20260618_r11_sfloor005_lower_alpha | R9 lower alpha follow-up | 0.005 | global | 0.339252785 | 0.736822360 | 0.353067815 | 0.0 | 3928 | 0.014364928 |

## Best Result

Best accepted experiment: `20260618_r9_sfloor005_room_weight`

Config path: `price_estimator/safe_lowest_price_gap/experiments/20260618_r9_sfloor005_room_weight/config.yaml`

Report path: `price_estimator/safe_lowest_price_gap/experiments/20260618_r9_sfloor005_room_weight/reports/summary_metrics.json`

Signal coverage / feasible coverage: `coverage_feasible = 0.735696867`

Coverage constraint satisfied: yes

Validation active gap improved from `0.516326182` to `0.336995717`.

## Checks

- `pytest -q tests/test_price_estimator_safe_lowest_price_gap.py`: 12 passed.
- All new experiments have `coverage_feasible >= 0.70`.
- All new experiments have `side_violation_rate == 0`.
- All new experiments have no forbidden feature overlap.
- All new experiments have zero `coverage=1.0` and `covered_gap_norm_mean=1.0` degenerate rows in `epoch_metrics.csv`.
- R10 filter A/B did not improve active gap: filter on `0.367110094` vs filter off `0.361100665`.
- R11 confirmed lower alpha search: selected alpha moved to `0.005`, but did not beat R9.
