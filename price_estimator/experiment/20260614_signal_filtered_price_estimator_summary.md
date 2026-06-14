# 20260614 Signal-Filtered Price Estimator Experiments

## Objective

Evaluate price upper-bound estimators only on samples that would enter the order stage:

```text
p_up >= deploy t_up OR p_up <= deploy t_down
```

Deploy thresholds were loaded from `execution_engine/deploy/baseline/artifact_manifest.json`.

Additional metric added:

```text
gap_p95_p05_range = quantile(gap, 0.95) - quantile(gap, 0.05)
gap = p_pred - target_raw
```

Lower `gap_p95_p05_range` is better because the prediction error band is tighter.

## Filtered Sample Counts

- train rows after signal filter: `11073`
- validation rows after signal filter: `5184`

## Experiment 1: Capped Quantile Calibration

Config:

```text
price_estimator/experiment/20260614_capped_quantile_signal_filter/config.yaml
```

Report:

```text
price_estimator/experiment/20260614_capped_quantile_signal_filter/reports/summary_metrics.json
```

Selected candidate:

- base column: `pred_q70`
- margin: `-0.08`
- cap: `p_side + 0.01`
- validation mean_gap: `0.0015104494`
- validation coverage: `0.4076003086`
- validation gap_p95_p05_range: `0.5163955112`
- validation p99_violation: `0.2983462737`
- validation max_violation: `0.4459929247`

## Experiment 2: Signal-Filtered Quantile Retrain

Config:

```text
price_estimator/experiment/20260614_quantile_signal_filter/config.yaml
```

Report:

```text
price_estimator/experiment/20260614_quantile_signal_filter/reports/quantile_metrics.json
```

Validation metrics on signal-filtered samples:

| output | mean_gap | coverage | gap_p95_p05_range | p99_violation |
|---|---:|---:|---:|---:|
| q70 | `0.0905369819` | `0.6224922840` | `0.5205512164` | `0.2119988380` |
| q80 | `0.1245412932` | `0.7170138889` | `0.5183836883` | `0.1742787195` |
| q90 | `0.1669910061` | `0.8092206790` | `0.5264436245` | `0.1334536608` |

## Experiment 3: Bounded MLP

Config:

```text
price_estimator/experiment/20260614_bounded_mlp_signal_filter/config.yaml
```

Report:

```text
price_estimator/experiment/20260614_bounded_mlp_signal_filter/reports/summary_metrics.json
```

Model output:

```text
p_pred = p_side * sigmoid(z)
```

Selected checkpoint metrics:

- validation mean_gap: `0.0006651615`
- validation coverage: `0.4342206790`
- validation gap_p95_p05_range: `0.5953564937`
- validation p99_violation: `0.3996924690`
- validation max_violation: `0.5975707845`

Closest checkpoint to `mean_gap ~= 0.02` from epoch metrics:

- epoch: `26`
- validation mean_gap: `0.020156`
- validation coverage: `0.474923`
- validation gap_p95_p05_range: `0.578920`
- validation p99_violation: `0.368769`
- validation max_violation: `0.580192`

## Conclusion

Filtering to accepted-signal samples is the correct evaluation scope for order-stage price estimation.

The new approaches can reach `mean_gap ~= 0.02`, but current versions do it by accepting low coverage and wide error bands. The best low-gap candidate is the capped quantile calibration, with the tightest observed range among these three experiments:

- mean_gap: `0.0015104494`
- gap_p95_p05_range: `0.5163955112`
- coverage: `0.4076003086`

The next useful optimization should explicitly minimize:

```text
mean_gap + lambda * gap_p95_p05_range + tail_violation_penalty
```

on accepted-signal samples, with a configurable minimum coverage floor.
