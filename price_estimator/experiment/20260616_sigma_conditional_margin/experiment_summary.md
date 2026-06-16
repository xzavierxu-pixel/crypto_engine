# 20260616 Sigma Conditional Margin

git_commit: d9259f780fde40b2b424fa7ed276f48cd2d8c2a4
config_path: price_estimator/configs/sigma_conditional_margin.yaml
report_path: price_estimator/experiment/20260616_sigma_conditional_margin/reports/summary_metrics.json
primary_metric: covered_normalized_lowest_safe_gap.mean
coverage_constraint_satisfied: yes
deploy_training_mode: price_estimator_experiment_only
offline_validation_metric_source: price_estimator/experiment/20260614_capped_quantile_signal_filter/reports/summary_metrics.json

## Objective

Optimize validation covered_normalized_lowest_safe_gap.mean subject to coverage >= 0.70.

## Feature Set

price_estimator datasets were rebuilt with sampled second-level features from:

artifacts/data_v2/second_level/version=second_level_v2/market=BTCUSDT

Rebuilt dataset shapes:

- train: 15657 rows, 1826 columns, 789 sl_/fm_ columns
- validation: 7432 rows, 1826 columns, 789 sl_/fm_ columns

Requested sigma features present:

- sl_rv_30s
- sl_mirror_rv_10s
- sl_mirror_rv_30s
- sl_mirror_rv_60s
- sl_taker_imbalance_30s

Requested sigma features missing from the current second-level store:

- sl_ofi_30s
- sl_spread_bps
- sl_bid_ask_qty_imbalance
- sl_microprice_premium
- sl_depth_imbalance_5
- sl_weighted_depth_imbalance_5

## Model Settings

Center model:

- CatBoost quantile q50
- chronological OOF with 3 splits
- target: target_logit, prediction transformed by sigmoid

Sigma models:

- sigma_v0: deploy feature set only
- sigma_v1: deploy feature set plus available sigma_feature_set columns
- target: abs(target_raw - OOF mu)
- loss: Huber:delta=0.05
- sigma_floor: 0.01

Calibration:

- q_quantiles: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
- sigma_high_quantiles: 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00
- p_side_cap_mode: exact_p_side
- tick_size: 0.01

## Results

Selected variant: sigma_v0

Baseline capped quantile:

- coverage: 0.3142361111
- covered_normalized_lowest_safe_gap.mean: 0.3773864987

sigma_v0:

- coverage: 0.7000385802
- covered_count: 3629
- legal_rate: 1.0
- covered_normalized_lowest_safe_gap.mean: 0.7528510515
- q_quantile: 0.90
- q_value: 1.2703857754
- sigma_high_quantile: 0.85
- sigma_high: 0.1236307684
- fallback_rate: 0.1402391975
- fallback_coverage: 0.6753782669
- non_fallback_coverage: 0.7040610276

sigma_v1:

- coverage: 0.7229938272
- covered_count: 3748
- legal_rate: 1.0
- covered_normalized_lowest_safe_gap.mean: 0.8281069331
- q_quantile: 0.95
- q_value: 1.8481187817
- sigma_high_quantile: 1.00
- fallback_rate: 0.0

## Conclusion

The experiment satisfies coverage >= 0.70, but it does not improve covered_normalized_lowest_safe_gap.mean versus the capped quantile baseline. sigma_v1 used the available sl features, but its selected validation gap was worse than sigma_v0, so the sl-enhanced conditional-margin variant should not be promoted.
