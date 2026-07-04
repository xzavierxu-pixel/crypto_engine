# crypto_engine

`crypto_engine` is a BTCUSDT 5-minute Polymarket trading research and execution repository. It builds market features, trains a selective UP/DOWN direction model, estimates an order price for accepted signals, and feeds both artifacts into a Polymarket execution adapter.
The system is intentionally narrow: one asset, one main horizon, one resolved Polymarket label family, one shared feature pipeline, one accepted direction artifact, and one execution-facing price-estimator artifact.

## Current Production State

Direction classifier:

```yaml
experiment_id: 20260611_catboost_calendar_coordinate_search
artifact_dir: execution_engine/deploy/baseline
config_path: experiments/configs/20260611_catboost_calendar_coordinate_search.yaml
report_path: artifacts/data_v2/reports/reversal_hybrid/20260611_catboost_calendar_coordinate_search/report.json
model_plugin: catboost
calibration_plugin: none
feature_count: 569
threshold_policy: utc_day_session_coordinate
threshold_source: offline_validation_calendar_coordinate
validation_selection_score: 0.6413740846
validation_coverage: 0.7000535619
validation_accepted_sample_accuracy: 0.7073450650
validation_accepted_count: 5228
fallback_t_up: 0.5792857143
fallback_t_down: 0.4314285714
```

Execution price estimator:

```yaml
active_artifact: expected_return_h14
artifact_dir: execution_engine/deploy/price_estimator_expected_return_h14
experiment_id: 20260619_expected_return_h14_h2_gc_gt_0p75
artifact_type: price_estimator_expected_return_hazard
prediction_column: expected_return_bid
order_price_policy: min(best_ask - 0.01, expected_return_optimal_bid)
validation_sum_pnl: 27.44
validation_mean_accepted_pnl: 0.0052486611
```

Execution template:

```yaml
runtime.mode: live
orders.enabled: true
orders.first.enabled: true
orders.second.enabled: false
price_estimator.enabled: true
```

Artifact manifests and config files are the source of truth. Older experiment documents are useful history, but they may not describe the current deploy state.

## Stage Relationships

```text
Binance and Polymarket source data -> normalized source tables and label stores -> shared feature builders and training frames -> direction classifier validation and threshold policy -> accepted direction artifact in execution_engine/deploy/baseline -> price-estimator / bid-policy artifact for accepted signals -> execution_engine runtime feature prep, market mapping, guards, and orders -> audit logs, live summaries, and PnL analysis
```

Each stage has a different job:

| Stage | Primary question | Main output | Main metric |
| ---|---|---|--- |
| Data and labels | Are the inputs reproducible and timestamp-safe? | `artifacts/data_v2/*` | QA manifests, leakage checks, row coverage |
| Direction model | Should the system trade, and on which side? | `p_up`, UP/DOWN/NO-SIGNAL thresholds, direction artifact | `selection_score` with coverage >= 0.70 |
| Price estimator | At what limit price should an accepted signal bid? | `expected_return_bid` or abstain | realized/simulated PnL, fill diagnostics |
| Execution | Can the artifacts be applied safely online? | orders, audit logs, summaries | order success, fill rate, realized PnL, guard behavior |

The direction model defines the accepted signal universe. The price estimator operates only after that universe is selected. A bid-policy PnL result should therefore always be read together with the direction coverage and accepted accuracy that produced the accepted rows.

## Prediction Target

The default direction target is the resolved Polymarket BTC 5-minute UP/DOWN outcome:

```yaml
label_builder: polymarket_resolved
label_version: polymarket_resolved_gamma_v1
label_store_path: artifacts/data_v2/labels/polymarket_resolved/btc_updown_5m.parquet
```

The historical BTC OHLCV direction label, such as `close[t0 + 4m] >= open[t0]`, is diagnostic only unless an experiment explicitly selects it.

## Feature Leakage Guardrails

Feature columns must be available at decision time and must not encode the label, future price path, settlement result, or post-decision order outcome.

For the direction model, `src/data/dataset_builder.py` is the main feature-column gate. It excludes base data columns, raw metadata, label audit metadata, and label-derived fields. These columns are forbidden as model features:

```text
target
future_close
abs_return
signed_return
stage1_target
stage2_target
stage1_sample_weight
polymarket_slug
polymarket_label_status
original_target
original_btc_direction_target
polymarket_target
label_mismatch
label_mismatch_vs_btc_direction
market_id
condition_id
question
endDate
closedTime
closed
umaResolutionStatus
outcomes
outcomePrices
winner
label_source
label_store_path
unresolved_or_missing_label_count
fetched_at
source
```

Raw/source/checksum metadata is also forbidden, including columns such as `raw_timestamp`, `source_file`, `source_date`, `checksum_status`, and any column prefixed with `raw_`, `source_`, or `checksum_`.

For expected-return, bid-policy, and price-estimator experiments, be stricter. Do not use columns or patterns that reveal post-decision fills, future lows, correctness, realized PnL, market identity, or timestamp keys. The current expected-return runners reject feature names matching this leakage pattern:

```text
target|label|winner|correct|chosen_low|future|closed|endDate|condition|market_id|question|slug|outcome|fetched|source|time_to|trade_time|timestamp|date|pnl
```

Common high-risk columns include:

```text
correct
chosen_low
chosen_low_next4
chosen_low_trade_time
lowest_trade_price_next4
lowest_trade_time_next4
time_to_chosen_low_sec
time_to_lowest_trade_sec
target_raw
threshold_accepted
predicted_side
predicted_outcome
realized_pnl
filled
printed_filled
```

Some of these columns are valid labels, diagnostics, grouping keys, or report fields. They are not valid model features unless an experiment explicitly proves they are available before the decision and documents why they are safe.

## Changing Objectives

Do not change the project objective by editing `AGENTS.md` alone. `AGENTS.md` tells coding agents how to behave; it is not the source of truth for labels, metrics, or deploy behavior.

Use this order instead:

1. Decide which stage is changing: direction selection, price estimation, execution policy, or label construction.
2. Update the real source of truth: `config/settings.yaml`, a new experiment config under `experiments/configs/`, a price-estimator config, or `execution_engine/config.example.yaml`.
3. Update the code only if the current builders, metrics, or reports do not support the new objective.
4. Run a no-leak validation report against the previous baseline.
5. Promote artifacts only after validation is accepted.
6. Then update `README.md` for project state and `AGENTS.md` for agent instructions/current facts.

Keep experiments isolated by default. Do not modify the main workflow, default configs, deploy artifacts, or execution policy for an experiment unless the change has been reviewed and explicitly accepted for promotion. Put each experiment under a self-contained directory named with a date and short description, for example:

```text
experiments/configs/20260703_short_description.yaml
artifacts/data_v2/reports/<experiment_family>/20260703_short_description/
price_estimator/expected_return/experiments/20260703_short_description/
```

An experiment can read from the current baseline artifacts, but it should write its own configs, reports, predictions, models, and summaries. After the result is accepted, promote the minimal required changes into the real main flow and update the deploy/config documentation in the same change.

Examples:

| Goal change | Primary files to change first | Documentation to update after validation |
| ---|---|--- |
| Change direction metric or coverage constraint | Experiment config or `config/settings.yaml` | `README.md`, `AGENTS.md`, experiment report |
| Change direction label semantics | `src/labels/`, horizon config, label store build scripts | `README.md`, `AGENTS.md`, data docs |
| Change bid/PnL objective | `price_estimator/*/config.yaml`, bid-policy code/reporting | `README.md`, price-estimator docs, `AGENTS.md` if it becomes current |
| Change live order behavior | `execution_engine/config.example.yaml`, order-plan/runtime code | `README.md`, `execution_engine/README.md`, `AGENTS.md` current facts |
| Change agent working rules only | `AGENTS.md` | Usually no project config change needed |

## Repository Map

| Path | Role |
| ---|--- |
| `config/settings.yaml` | Default offline data, feature, label, objective, and split configuration. |
| `src/core/` | Schemas, time grid logic, constants, and validation helpers. |
| `src/features/` | Shared feature packs and registry used offline and online. |
| `src/labels/` | Polymarket resolved labels and diagnostic labels. |
| `src/data/` | Loaders, preprocessing, and training-frame assembly. |
| `src/model/` | Model plugins, evaluation, training, and artifact helpers. |
| `scripts/` | Offline data/model/runtime commands. See `scripts/README.md`. |
| `execution_engine/` | Runtime adapter, artifact loading, order planning, CLOB client, and deploy artifacts. |
| `price_estimator/` | Historical and current bid/price-estimator experiments and artifacts. |
| `artifacts/data_v2/` | Local raw, normalized, label, feature, dataset, report, and experiment outputs. |
| `tests/` | Focused unit and integration tests for data/model/execution behavior. |

## Offline to Online Flow

1. Build or refresh raw and normalized data under `artifacts/data_v2`.
2. Build the Polymarket resolved label store.
3. Build second-level feature stores when configured. Current `second_level.enabled` is `true`.
4. Build the 5-minute training frame with the shared feature and label builders.
5. Run direction-model validation. The current accepted direction artifact comes from `scripts/analysis/catboost_calendar_coordinate_search.py`.
6. Promote only validation-accepted direction artifacts. Full-train or deployment metrics are diagnostics, not acceptance scores.
7. Train or select a price-estimator artifact for accepted signals. The current execution template uses expected-return hazard H14.
8. Configure `execution_engine/config.example.yaml` or the server-local equivalent to load both artifacts.
9. Run shadow, paper, or live execution and analyze logs with `scripts/analysis/` helpers.

## Main Commands

Build a training frame shape:

```bash
rtk python scripts/data/step4_features/build_dataset.py \
  --input artifacts/data_v2/normalized/binance/spot/BTCUSDT/klines/BTCUSDT-1m.parquet \
  --config config/settings.yaml \
  --horizon 5m \
  --data-root artifacts/data_v2
```

Reproduce the current accepted direction run:

```bash
rtk python scripts/analysis/catboost_calendar_coordinate_search.py \
  --config experiments/configs/20260611_catboost_calendar_coordinate_search.yaml
```

Run one execution-engine cycle in paper/audit mode:

```bash
rtk python execution_engine/run_once.py \
  --config execution_engine/config.example.yaml \
  --mode paper \
  --print-json
```

Use a server-local `execution_engine/config.yaml` for live execution. Use `--help` on each script for exact required inputs. Some commands intentionally require local artifact paths or server-specific runtime config.

## Validation

Useful focused checks:

```bash
rtk python -m pytest -q \
  tests/test_model_pipeline.py::test_online_full_train_script_uses_accepted_thresholds_and_writes_deploy_artifacts \
  tests/test_model_artifacts.py::test_load_binary_selective_artifacts_from_manifest_and_directory \
  tests/test_execution_engine.py::test_execution_config_example_loads
```

For docs-only changes:

```bash
git diff --check -- <changed-files>
```

## Notes for Contributors

- Keep feature and label logic centralized in `src/`.
- Keep execution adapters thin and config-driven.
- Do not hard-code thresholds or runtime business rules in scripts.
- Compare direction experiments against the current validation baseline.
- Compare price-estimator and execution-policy experiments on PnL while preserving direction metrics.
- Read `AGENTS.md` before asking an agent to modify code or run experiments.
