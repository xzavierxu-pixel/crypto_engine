# Validation Balanced Precision Optimization - 2026-05-02

## Goal

Raise validation `balanced_precision` from the AGENTS.md baseline `0.5627267617` to `>= 0.6000`, while preserving:

- `signal_coverage >= 0.60`
- `up_signal_count >= 50`
- `down_signal_count >= 50`
- `total_signal_count >= 150`

Validation is threshold-tuned and used as the current project acceptance set, so these results are optimistic by design.

## Runs

Primary reports are under ignored `artifacts/` paths and config snapshots are under tracked `experiments/configs/`.

| run | variants | best variant | balanced_precision | precision_up | precision_down | coverage | target reached |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `20260502_validation_bp_optimization` | 37 | `drop_second_level_interaction_bank` | 0.564247 | 0.605578 | 0.522916 | 0.612088 | no |
| `20260502_validation_bp_optimization_round2` | 45 | `feature_select_top_500` | 0.570259 | 0.618182 | 0.522337 | 0.600289 | no |
| `20260502_validation_bp_optimization_round3` | 61 | `feature_select_top_550` | 0.581355 | 0.645455 | 0.517256 | 0.605586 | no |

Best report:

```text
artifacts/data_v2/experiments/20260502_validation_bp_optimization_round3/feature_select_top_550/report.json
```

Best config snapshot:

```text
experiments/configs/20260502_validation_bp_optimization_round3/feature_select_top_550.yaml
```

Best feature set:

```text
artifacts/data_v2/experiments/20260502_validation_bp_optimization_round3/feature_select_top_550/feature_set.json
```

## Findings

- Threshold range expansion did not improve the baseline. The selected thresholds stayed at the same effective boundary for the full-feature model.
- Recomputing sample weights made the conservative and aggressive variants real experiments, but neither exceeded feature selection.
- The strongest single improvement was feature selection from baseline LightGBM gain importance.
- `feature_select_top_550` improved validation balanced precision by `+0.018628`, but still missed the `0.6000` target by `0.018645`.
- The limiting side remains DOWN precision: best `precision_down=0.517256`, only slightly below or near baseline, while UP precision improved to `0.645455`.
- Several planned ablations were no-ops on the cached training frame because those feature pack names did not map to present columns, especially book/depth-related packs in the 1s-backbone-only feature store.

## Conclusion

The target was not reached. Do not update the AGENTS.md baseline to this run as a successful `>= 0.6000` result.

Recommended next experiment direction:

1. Build an explicit feature selection mechanism in config instead of relying on runner-only top-N filtering.
2. Investigate DOWN false positives with slice reports before adding more model capacity.
3. Consider label/no-trade boundary analysis only if label changes are explicitly approved.
