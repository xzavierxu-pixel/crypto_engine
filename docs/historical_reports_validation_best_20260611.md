# Historical Reports Validation Best

Generated on 2026-06-11 from `artifacts/data_v2/reports`.

## Conclusion

The best validation-set experiment currently present in `artifacts/data_v2/reports` is:

`20260611_reversal_auxiliary_ablation`

Report:
`artifacts/data_v2/reports/reversal_hybrid/20260611_reversal_auxiliary_ablation/report.json`

Best variant:
`impulse_flow_subset`

This is the top ranked result because it has the highest validation `selection_score` among reports with `coverage >= 0.70`.

## Selection Rule

The ranking follows the project objective:

```text
selection_score = utility / downside_risk
coverage >= 0.70
```

Primary ranking:

1. validation `selection_score` with `coverage >= 0.70`
2. positive `utility` and `accepted_sample_accuracy > 0.50`
3. coverage and accepted count
4. implementation and leakage risk noted separately

Diagnostics such as YES/NO balance, AUC, logloss, Brier, and generic accuracy were not used as the primary objective.

## Best Validation Result

| Metric | Value |
| --- | ---: |
| experiment_id | `20260611_reversal_auxiliary_ablation` |
| accepted | `true` |
| best_variant | `impulse_flow_subset` |
| selection_score | `0.5899471884` |
| coverage | `0.7047402250` |
| coverage constraint satisfied | `yes` |
| accepted_sample_accuracy | `0.6942808284` |
| utility | `0.2738350295` |
| downside_risk | `0.4641687169` |
| sample_count | `7468` |
| accepted_count | `5263` |
| up_prediction_count | `2936` |
| down_prediction_count | `2327` |
| selected_t_up | `0.55` |
| selected_t_down | `0.15` |
| precision_up | `0.6941416894` |
| precision_down | `0.6944563816` |
| balanced_precision | `0.6942990355` |
| validation_window_start | `2026-04-11 00:15:00+00:00` |
| validation_window_end | `2026-05-10 23:50:00+00:00` |
| config_path | `experiments/configs/20260611_reversal_auxiliary_ablation.yaml` |

## Comparison Against Recorded Accepted Baseline

The project AGENTS baseline is:

`20260520_polymarket_resolved_extended_history_baseline`

| Metric | Accepted baseline | Best reports result | Delta |
| --- | ---: | ---: | ---: |
| selection_score | `0.5748509217` | `0.5899471884` | `+0.0150962667` |
| utility | `0.2674076058` | `0.2738350295` | `+0.0064274237` |
| accepted_sample_accuracy | `0.6909542934` | `0.6942808284` | `+0.0033265350` |
| coverage | `0.7001874665` | `0.7047402250` | `+0.0045527585` |
| accepted_count | `5229` | `5263` | `+34` |
| up_prediction_count | `2648` | `2936` | `+288` |
| down_prediction_count | `2581` | `2327` | `-254` |

Under the required coverage constraint, the best reports result improves validation `selection_score` versus the recorded accepted baseline.

Important deployment note: this document identifies the best validation report in `artifacts/data_v2/reports`; it does not by itself prove that `execution_engine/deploy/baseline` has been regenerated from this experiment.

## Ranked Standard Validation Reports

Only JSON reports with top-level `validation_metrics.selection_score` and `validation_metrics.coverage` were included in this main ranking.

| Rank | Experiment | Accepted | Selection score | Coverage | Accepted accuracy | Utility | Accepted count | Report |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `20260611_reversal_auxiliary_ablation` | `true` | `0.5899471884` | `0.7047402250` | `0.6942808284` | `0.2738350295` | `5263` | `artifacts/data_v2/reports/reversal_hybrid/20260611_reversal_auxiliary_ablation/report.json` |
| 2 | `20260611_four_bucket_abstain_gate` | `true` | `0.5562216733` | `0.7880289234` | `0.6778249788` | `0.2802624531` | `5885` | `artifacts/data_v2/reports/reversal_hybrid/20260611_four_bucket_abstain_gate/report.json` |
| 3 | `baseline_coverage_090_report` | n/a | `0.5144593007` | `0.9027852169` | `0.6582616434` | `0.2857525442` | `6742` | `artifacts/data_v2/reports/baseline_coverage_090_report.json` |
| 4 | `20260611_reversal_direction_meta_gate` | `false` | `0.3553145772` | `0.7404927691` | `0.6262206148` | `0.1869309052` | `5530` | `artifacts/data_v2/reports/reversal_hybrid/20260611_reversal_direction_meta_gate/report.json` |
| 5 | `20260611_reversal_precision_gate` | `false` | `0.2116756669` | `0.7957953937` | `0.5771495878` | `0.1227905731` | `5943` | `artifacts/data_v2/reports/reversal_hybrid/20260611_reversal_precision_gate/report.json` |
| 6 | `20260611_reversal_rescue_gate` | `false` | `-0.1561816959` | `0.5502142475` | `0.4198101728` | `-0.0882431709` | `4109` | `artifacts/data_v2/reports/reversal_hybrid/20260611_reversal_rescue_gate/report.json` |

`20260611_reversal_rescue_gate` fails the `coverage >= 0.70` constraint and is invalid for acceptance ranking even before considering its negative utility.

## Frontier CSV Cross-Check

The frontier CSV files under `artifacts/data_v2/reports/reversal_hybrid` were checked separately. The best `selection_score` with `coverage >= 0.70` in each frontier was:

| Frontier | Best selection score | Coverage | Accepted accuracy | Notes |
| --- | ---: | ---: | ---: | --- |
| `20260611_reversal_auxiliary_ablation/impulse_flow_subset_frontier.csv` | `0.5899471884` | `0.7047402250` | `0.6942808284` | Matches the best report result |
| `20260611_reversal_auxiliary_ablation/all_features_frontier.csv` | `0.5781833368` | `0.7478575254` | `0.6870188004` | Below best |
| `20260611_reversal_auxiliary_ablation/minus_broad_volatility_frontier.csv` | `0.5781833368` | `0.7478575254` | `0.6870188004` | Below best |
| `20260611_four_bucket_abstain_gate/gate_frontier.csv` | `0.5562216733` | `0.7880289234` | `0.6778249788` | Below best |
| `20260611_reversal_direction_meta_gate/frontier.csv` | `0.5458120406` | `0.7074183182` | `0.6827560098` | Below best |
| `20260611_reversal_rescue_gate/frontier.csv` | `0.5376133608` | `0.7565613283` | `0.6759292035` | Below best |
| `20260611_reversal_precision_gate/frontier.csv` | `0.5078232614` | `0.8525709695` | `0.6602795665` | Below best |

Older threshold-frontier reports from 2026-05-21 were also below the best report result. The best validation candidate observed in `regime_reversal_threshold_exact_frontier_replay_gate_20260521.json` was approximately `0.5411193355`, which does not beat `0.5899471884`.

## Reports Not Used For Main Ranking

The following files were present but not used as standard validation-ranking inputs:

| Path | Reason |
| --- | --- |
| `artifacts/data_v2/reports/price_edge/price_edge_evaluation_20260607.json` | Price-edge evaluation report, not a standard model validation `selection_score` report |
| `artifacts/data_v2/reports/price_edge/price_edge_evaluation_min_ev_010_20260607.json` | Price-edge evaluation report, not a standard model validation `selection_score` report |
| `artifacts/data_v2/reports/regime_reversal_prd_audit_20260521.json` | Audit/checklist report, no top-level validation metrics |
| `artifacts/data_v2/reports/reversal_diagnostics/baseline_reversal_trend_slices_090.json` | Diagnostic slice report, not an acceptance ranking report |
| `artifacts/data_v2/reports/reversal_hybrid/20260611_oof_meta_gate_blocked/report.json` | Blocked experiment report without comparable validation `selection_score` |

## Interpretation

`20260611_reversal_auxiliary_ablation` is a narrow validation improvement over the recorded accepted baseline. It preserves the minimum coverage constraint, improves accepted-sample accuracy, and increases accepted signal count slightly.

The margin is real in the saved validation report, but it should be treated as validation-tuned and optimistic under the project protocol. Before using it for deployment, the offline artifact and threshold source should be verified, then `train_online_full_train.py` should be run only from the accepted split artifact if deployment regeneration is intended.
