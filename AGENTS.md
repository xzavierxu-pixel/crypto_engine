# AGENTS.md

This file is for Codex-style coding agents working in this repository. For a project overview, read `README.md`. For script commands, read `scripts/README.md`.

## Mission

Improve an existing BTCUSDT 5-minute Polymarket trading system with small, measurable, low-risk changes. Do not treat this as a greenfield project.

The target workflow has five connected stages:

1. Data and features produce reproducible offline/online inputs.
2. The direction model selects UP or DOWN with one threshold and no trade/no-trade filtering.
3. A calibration layer calibrates the selected-side probability.
4. The price estimator / bid-policy stage chooses a limit price or policy action from the selected side and calibrated probability.
5. The execution layer applies the artifacts, guards, market mapping, and order submission/audit logic.

Keep these stages separate. Direction accuracy improvements, bid-policy improvements, and live execution changes use different metrics and must not be mixed without an explicit report.

## Current Facts

As of 2026-07-03, the accepted direction baseline is:

```yaml
experiment_id: 20260611_catboost_calendar_coordinate_search
config_path: experiments/configs/20260611_catboost_calendar_coordinate_search.yaml
report_path: artifacts/data_v2/reports/reversal_hybrid/20260611_catboost_calendar_coordinate_search/report.json
deploy_artifact_dir: execution_engine/deploy/baseline
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

The execution config template currently enables:

```yaml
runtime.mode: live
orders.enabled: true
price_estimator.active_artifact: expected_return_h14
price_estimator.artifact_dir: execution_engine/deploy/price_estimator_expected_return_h14
```

The active execution price estimator manifest is:

```yaml
artifact_type: price_estimator_expected_return_hazard
experiment_id: 20260619_expected_return_h14_h2_gc_gt_0p75
model_file: expected_return_hazard.npz
prediction_column: expected_return_bid
feature_count: 576
order_price_policy: min(best_ask - 0.01, expected_return_optimal_bid)
validation_sum_pnl: 27.44
validation_mean_accepted_pnl: 0.0052486611
validation_order_coverage: 0.4967482785
wrong_fill_forced: 1.0
```

Use the artifact manifests and config files as source of truth for the current deployed/accepted state. The target workflow below is a proposed objective redesign for new experiments; it is not promoted into the main flow until validation results are reviewed and accepted.

## Target Objective Redesign

Do not overwrite the current deploy facts with this redesign until it has a dated proposal, isolated experiment outputs, no-leak validation, and explicit user approval to promote.

The intended direction is:

1. Direction model: select only UP or DOWN using one threshold. This stage no longer filters samples out of the trade universe, so direction coverage is `100%`.
2. Direction evaluation: optimize validation accuracy. Keep existing direction metrics such as coverage, selection_score, utility, precision_up, precision_down, AUC, Brier, and logloss as diagnostics, but do not use selective `selection_score` as the primary target for this redesign.
3. Calibration layer: after selecting a side, calibrate the probability that the selected side is correct. Evaluate calibration primarily with Brier score, plus reliability curves/logloss when available. Feed the calibrated selected-side probability into the price estimator.
4. Price estimator / bid policy: evaluate multiple schemes. Expected-return EV is the current leading direction, but other bid-policy or direct-PnL approaches are allowed when they are isolated and reported clearly.
5. EV mechanics: use the fill model where a correct prediction may not fill, while a wrong submitted order is forced filled. With calibrated correctness probability `q`, bid `b`, and winner-fill CDF `Gc(b|X)`, the base EV form is `q * Gc(b|X) * (1 - b) - (1 - q) * b`.
6. Gc evaluation: if a scheme estimates `Gc`, evaluate it as a probability forecast with Brier score and reliability diagnostics for the event `winner_low <= bid`, in addition to downstream PnL.
7. Final objective: maximize validation-set `sum_pnl` under the documented fill rule. If the current EV/Gc path stops improving validation `sum_pnl`, consider other isolated approaches rather than forcing more changes into the same path.

## Stage Boundaries

### Data and Feature Stage

- Source config: `config/settings.yaml`.
- Main outputs: `artifacts/data_v2/raw`, `normalized`, `labels`, `second_level`, `datasets`, and `manifests`.
- Main scripts: `scripts/data/` and `src/data/`.
- Current `second_level.enabled` is `true`; do not assume second-level features are disabled.
- Feature logic belongs in `src/features/`; label logic belongs in `src/labels/`.

### Direction Model Stage

- Purpose: produce `p_up` and a selected UP/DOWN side.
- Target-redesign behavior: use one threshold to split UP vs DOWN and do not create NO-SIGNAL rows; direction coverage is `100%`.
- Target-redesign primary metric: validation accuracy.
- Legacy/selective metrics such as `selection_score`, coverage, accepted accuracy, and utility remain diagnostics for comparison with existing artifacts.
- Current accepted direction baseline is `20260611_catboost_calendar_coordinate_search`.
- Thresholds must come from config or artifact manifests, never from hard-coded `0.5` logic.

### Calibration Layer Stage

- Purpose: convert the selected-side raw model probability into a calibrated probability of being correct.
- Primary metric: validation Brier score for selected-side correctness.
- Secondary diagnostics: reliability tables/plots, logloss, calibration by probability bucket, and calibration drift by time bucket.
- The calibrated probability is an input to the price estimator; do not let the price estimator silently reuse uncalibrated probabilities unless the experiment is explicitly testing that ablation.

### Price Estimator Stage

- Purpose: choose the limit price or bid-policy action after direction selection and probability calibration.
- Primary metric is validation `sum_pnl` under the documented fill rule, not direction `selection_score`.
- Current leading approach is expected-return EV with calibrated correctness probability and a winner-fill model, but direct-PnL, empirical, grouped, or policy-learning alternatives may be explored in isolated experiments.
- If the estimator models `Gc(b|X)`, evaluate `Gc` with Brier score/reliability for the fill event as well as downstream PnL.
- Current execution template uses expected-return hazard artifact `expected_return_h14`.
- Older `safe_lowest_price_gap` material is historical unless the task explicitly asks for that line of work.

### Execution Stage

- Purpose: load the direction artifact, load the price-estimator artifact, prepare runtime features, map the Polymarket market, and submit or audit orders.
- Runtime order behavior is config-driven through `execution_engine/config.example.yaml` and server-local config.
- Do not move model training, feature definitions, or label definitions into `execution_engine/`.

## Non-Negotiable Rules

- Preserve offline/online feature parity.
- Do not introduce future-looking features or label-derived feature columns.
- Do not silently change label, horizon, timestamp alignment, side semantics, or fill semantics.
- Do not duplicate feature, label, model, or threshold logic in scripts or execution adapters.
- Keep business parameters in `config/settings.yaml`, experiment configs, artifact manifests, or execution config.
- Use chronological splits for validation unless the user explicitly requests another protocol.
- Treat full-train metrics as deployment diagnostics only; never compare them against validation acceptance metrics.
- Do not commit changes unless the user explicitly asks.
- Prefer `rtk python ...` for long local commands when `rtk` is available.
- DO NOT send optional commentary.

## Direction Metrics

For the target redesign, the direction stage uses one UP/DOWN threshold, has `coverage = 1.0`, and optimizes validation accuracy. Report at least:

```text
sample_count
accuracy
selected_threshold
up_prediction_count
down_prediction_count
share_up_predictions
share_down_predictions
precision_up
precision_down
balanced_precision
roc_auc
brier_score
log_loss
```

For legacy/selective comparisons, also report the existing fields:

```text
sample_count
coverage
accepted_count
accepted_sample_accuracy
precision_up
precision_down
balanced_precision
share_up_predictions
share_down_predictions
selected_t_up
selected_t_down
up_prediction_count
down_prediction_count
roc_auc
brier_score
log_loss
utility
downside_risk
selection_score
```

The target-redesign ranking rule is:

1. maximize validation accuracy with coverage fixed at `100%`
2. prefer better side balance and stability across time splits when accuracy is tied
3. keep legacy selective metrics as diagnostics, not the primary target

The legacy/selective ranking rule remains:

1. maximize validation `selection_score` with `coverage >= 0.70`
2. prefer positive utility and accepted accuracy above `0.50`
3. prefer more stable time splits, simpler thresholds, and lower leakage risk

Do not mix these ranking rules in the same claim.

## Calibration Metrics

For selected-side probability calibration, report:

```text
sample_count
brier_score
log_loss
calibration_method
raw_probability_column
calibrated_probability_column
reliability_by_probability_bucket
accuracy_by_probability_bucket
```

The calibrated selected-side probability is the probability input to price-estimator and EV experiments.

## PnL Metrics

For price-estimator, bid-policy, or execution-policy experiments, report:

```text
accepted_count
order_count
order_coverage
trade_count
fill_rate
sum_pnl
mean_accepted_pnl
mean_pnl_filled
mean_bid
win_pnl_sum
loss_pnl_sum
correct_fill_rate
wrong_fill_forced
wrong_fill_printed, if available as a diagnostic
```

Keep direction and calibration metrics in the same report so a PnL gain is not accepted if it comes from changing side selection, probability calibration, or the signal universe unexpectedly.

## Leakage Checks

Before claiming an improvement, verify that feature columns exclude:

```text
target
future_*
abs_return
signed_return
stage1_target
stage2_target
chosen_low
correct
winner
pnl
trade_time
endDate
condition_id
market_id
slug
outcome
```

Also verify that scalers, imputers, encoders, feature selectors, calibration maps, and policy thresholds are not fitted on validation rows unless the document explicitly labels the run as a leaky diagnostic.

## Change Protocol

Do not change the project objective by editing this file alone. `AGENTS.md` records how agents should work and what the current accepted facts are; objective changes must start in the relevant config, code path, and validation report.

Keep experiments isolated. Do not edit the main workflow, default configs, deploy artifacts, or live execution behavior for a trial unless the user has accepted the result and asked to promote it. Use date-plus-description experiment directories such as `20260703_short_description`, and keep that run's config, report, predictions, models, and notes together.

For the target objective redesign above, create a dated proposal/design document before changing the main workflow. The proposal should define the exact direction threshold rule, calibration method, price-estimator candidates, EV/fill assumptions, validation windows, leakage guards, and promotion criteria. After the proposal is reviewed, implement experiments in isolated dated directories. Promote only the minimal winning pieces after validation and user approval.

Use the stage as the routing key:

- Direction objective or coverage rule: update an isolated experiment config first, then validate against the current direction baseline.
- Calibration objective: update an isolated calibration experiment/config and report Brier score before feeding calibrated probabilities downstream.
- Label semantics: update `src/labels/`, horizon config, label-store generation, and tests.
- Price-estimator or bid-policy objective: update the relevant `price_estimator/` config and report PnL metrics with direction metrics preserved.
- Live order behavior: update `execution_engine/config.example.yaml` or execution code, then verify runtime config loading and order-plan tests.

After validation, update `README.md` for project state and update this file only for agent instructions/current facts.

Before editing code, identify:

- the stage being changed
- the metric being improved
- the files and configs affected
- the validation command or report that can falsify the change

After editing, run the narrowest useful validation. For docs-only changes, at least run `git diff --check` on the touched files.

For completed experiments, save or reference:

- config path
- report path
- split windows
- feature set
- artifact path, if generated
- before/after metrics against the current baseline

## Useful Entry Points

| Task | Path |
|---|---|
| Project overview | `README.md` |
| Script reference | `scripts/README.md` |
| Default config | `config/settings.yaml` |
| Direction deploy manifest | `execution_engine/deploy/baseline/artifact_manifest.json` |
| Execution config template | `execution_engine/config.example.yaml` |
| Feature builder | `src/features/builder.py` |
| Feature registry | `src/features/registry.py` |
| Polymarket label builder | `src/labels/polymarket_resolved.py` |
| Training frame assembly | `src/data/dataset_builder.py` |
| Direction training | `scripts/model/train_model.py` |
| Current accepted direction runner | `scripts/analysis/catboost_calendar_coordinate_search.py` |
| Runtime artifact loading | `execution_engine/artifacts.py` |

## Definition of Done

A task is done when the relevant stage still has clear inputs and outputs, the current baseline is not misrepresented, validation has been run or explicitly skipped with a reason, and the final report states whether the requested metric actually improved.