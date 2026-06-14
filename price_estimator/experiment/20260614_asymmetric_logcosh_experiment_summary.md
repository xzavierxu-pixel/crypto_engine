# 20260614 Upper-Bound MLP Asymmetric Log-Cosh Experiments

## Objective

Minimize validation `mean_gap` subject to validation `coverage >= 0.90`.

Baseline from `price_estimator/upper_bound_mlp/reports/upper_bound_mlp_metrics.json`:

- validation coverage: `0.9025834230`
- validation mean_gap: `0.2175521255`
- validation max_violation: `0.3959452212`

## Implemented Variants

- Variant B: `mean_gap_soft_violation`
  - grid run: `price_estimator/experiment/20260614_soft_logcosh_c50_scales_upper_bound_mlp_grid_summary.json`
  - tested: `C=50`, `scale=0.03, 0.05, 0.08`
- Variant A: `asymmetric_logcosh`
  - grid run: `price_estimator/experiment/20260614_asym_logcosh_wu20_upper_bound_mlp_grid_summary.json`
  - tested: `w_under=20`, `scale=0.08, 0.05, 0.03, 0.02`
  - grid run: `price_estimator/experiment/20260614_asym_logcosh_wu10_upper_bound_mlp_grid_summary.json`
  - tested: `w_under=10`, `scale=0.08, 0.05, 0.03, 0.02`

Each run has its own:

- `config.yaml`
- `models/upper_bound_mlp.pt`
- `reports/summary_metrics.json`
- `reports/epoch_metrics.csv`
- `reports/predictions_train.parquet`
- `reports/predictions_validation.parquet`
- train/validation diagnostics CSVs by `price_bin`, `p_bin`, and `selected_side`

## Best Results

Best passing result:

- experiment_id: `20260614_asym_logcosh_wu20_upper_bound_mlp_asym_wu20p0_s0p02`
- validation coverage: `0.9032561895`
- validation mean_gap: `0.2540963888`
- validation p99_violation: `0.0936721563`
- validation max_violation: `0.2442458272`
- coverage constraint satisfied: yes

Best non-passing low-gap result:

- experiment_id: `20260614_asym_logcosh_wu10_upper_bound_mlp_asym_wu10p0_s0p08`
- validation coverage: `0.7962863294`
- validation mean_gap: `0.1963874549`
- validation p99_violation: `0.1494286060`
- validation max_violation: `0.3104477525`
- coverage constraint satisfied: no

Best Variant B result by mean_gap with nonzero coverage:

- experiment_id: `20260614_soft_logcosh_c50_scales_upper_bound_mlp_mean_gap_C50p0_s0p08`
- validation coverage: `0.5819429494`
- validation mean_gap: `0.0756547973`
- validation p99_violation: `0.2404368967`
- validation max_violation: `0.6848136783`
- coverage constraint satisfied: no

## Conclusion

No tested asymmetric log-cosh or mean-gap soft-violation experiment improved on the old ALM validation baseline under the `coverage >= 0.90` constraint.

The only passing run reduced tail violation versus the ALM baseline but increased mean_gap:

- before mean_gap: `0.2175521255`
- after mean_gap: `0.2540963888`
- before coverage: `0.9025834230`
- after coverage: `0.9032561895`

Do not promote these artifacts to deploy.
