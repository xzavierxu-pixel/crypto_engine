# Requirement: Stacked Continuation + Reversal Meta Model Experiment

## 1. Background

The previous `20260612_catboost_continuation_reversal_hybrid_router` experiment showed that hard routing is too brittle:

* `conflict_margin` nearly preserved coverage but destroyed accepted accuracy and utility.
* `gate_hybrid` improved accepted accuracy but collapsed coverage to about 38%, far below the required `coverage >= 0.70`.
* The gate variant mostly behaved like a high-confidence continuation filter and did not create a usable balance between continuation and reversal signals.

The likely problem is not that the continuation and reversal experts are useless. The problem is that hard routing forces the system to choose one expert before the final UP/DOWN decision. That loses information from the other expert and makes conflicts too coarse.

This document proposes a soft stacking experiment: use both expert models as feature generators, then train a shallow final model to predict the final Polymarket UP/DOWN target.

---

## 2. Objective

Primary objective remains unchanged:

```text
maximize validation selection_score
subject to coverage >= 0.70
```

Acceptance is still based on the current validation baseline:

```yaml
baseline_experiment_id: 20260520_polymarket_resolved_extended_history_baseline
baseline_selection_score: 0.5748509217
baseline_utility: 0.2674076058
baseline_accepted_sample_accuracy: 0.6909542934
baseline_accepted_count: 5229
baseline_coverage: 0.7001874665
```

The experiment is successful only if:

```text
validation coverage >= 0.70
validation selection_score > 0.5748509217
utility > 0
accepted_sample_accuracy > 0.50
```

Diagnostics such as YES/NO balance, AUC, Brier, logloss, continuation accuracy, and reversal accuracy must be reported but must not replace the primary objective.

---

## 3. Proposed Design

### 3.1 High-Level Idea

Train a second-level shallow CatBoost model using:

* continuation expert probability outputs
* reversal expert probability outputs
* expert accept/abstain flags
* expert confidence values
* first-minute side and same/opposite-side indicators
* calendar/session features
* a small, explicitly controlled set of important pre-decision market features

The second-level model outputs:

```text
stacked_p_up = P(Polymarket resolved target is UP)
```

Then run the normal selective binary threshold search:

```text
UP if stacked_p_up >= t_up
DOWN if stacked_p_up <= t_down
ABSTAIN otherwise
```

This keeps the final decision mechanism compatible with the existing project signal rule.

### 3.2 Why This May Improve Selection Score

Hard routing asks:

```text
Should we trust continuation or reversal for this row?
```

Stacking asks a more useful question:

```text
Given both experts' probabilities, confidence, direction constraints, and regime context, what is the final UP probability?
```

This can improve the score because:

* The final model can learn that a weak reversal signal may reduce continuation confidence without fully flipping direction.
* It can learn asymmetric behavior, for example `first_minute_side == YES` may need different reversal evidence than `first_minute_side == NO`.
* It can preserve broad coverage by searching thresholds on the stacked output instead of requiring either expert to accept directly.
* It avoids making `continuation_accept` or `reversal_accept` hard final gates.

---

## 4. Input Expert Models

### 4.1 Continuation Expert

Source:

```text
artifacts\data_v2\reports\reversal_hybrid\20260611_catboost_continuation_side_coordinate_search
```

Expected fields to produce for each row:

```text
continuation_p_up
continuation_accept
continuation_decision
continuation_confidence
continuation_same_side_signal
continuation_opposite_side_signal
```

Decision semantics:

* `continuation_accept` should follow the continuation expert's existing per-regime thresholds.
* The continuation expert remains same-side by construction:
  * `first_minute_side == YES` can only produce UP.
  * `first_minute_side == NO` can only produce DOWN.

### 4.2 Reversal Expert

Source:

```text
artifacts\data_v2\reports\reversal_hybrid\20260611_catboost_reversal_sample_weight_search
```

Use the `reversal_weight_8` settings:

```yaml
reversal_sample_weight: 8.0
reversal_t_up: 0.45
reversal_t_down: 0.30
```

Expected fields to produce for each row:

```text
reversal_p_up
reversal_accept
reversal_decision
reversal_confidence
reversal_same_side_signal
reversal_opposite_side_signal
```

Decision semantics:

* `reversal_accept` should follow reversal-only direction constraints:
  * If `first_minute_side == YES`, reversal direction is DOWN and requires `reversal_p_up <= reversal_t_down`.
  * If `first_minute_side == NO`, reversal direction is UP and requires `reversal_p_up >= reversal_t_up`.
* The reversal expert must not be allowed to produce same-side final decisions when creating `reversal_accept`.

---

## 5. Stacking Feature Set

### 5.1 Required Expert Output Features

```text
continuation_p_up
reversal_p_up
continuation_accept
reversal_accept
continuation_confidence
reversal_confidence
continuation_minus_reversal_p_up
continuation_confidence_minus_reversal_confidence
max_expert_confidence
min_expert_confidence
both_accept
neither_accept
only_continuation_accept
only_reversal_accept
experts_agree_direction
experts_disagree_direction
```

Confidence definition:

```python
confidence = abs(p_up - 0.5)
```

### 5.2 Direction and First-Minute Features

```text
first_minute_side
first_minute_side_yes
first_minute_side_no
continuation_same_side_indicator
reversal_opposite_side_indicator
cont_decision_is_up
cont_decision_is_down
rev_decision_is_up
rev_decision_is_down
```

### 5.3 Calendar and Session Features

```text
dayofweek
hour
session_id
session_open
session_asia
session_europe
session_us
```

Use the existing project definitions if available. Do not invent a new session definition if the feature frame already has one.

### 5.4 Optional Important Market Features

Add a small allowlist of pre-decision features that were already available in the baseline frame. Candidate groups:

```text
fm_*
ret_1*
slope*
flow*
imbalance*
divergence*
wick*
momentum_acceleration*
second_level_first_minute_impulse*
second_level_microstructure*
range*
volume*
volatility*
liquidity*
```

Feature selection must still exclude leakage columns:

```text
target
future_close
abs_return
signed_return
stage1_target
stage2_target
original_btc_direction_target
any column containing "target"
```

Recommended first implementation:

```text
expert stack features + first_minute/calendar/session + top 50 important pre-decision market features
```

This keeps the experiment isolated and reduces the risk that the stacker simply becomes a larger untracked base model.

---

## 6. Target and Model

### 6.1 Target

Use the unchanged project label:

```text
target = resolved Polymarket BTC 5m UP/DOWN settlement outcome
label_builder = polymarket_resolved
label_version = polymarket_resolved_gamma_v1
```

Do not switch to BTC OHLCV direction labels.

### 6.2 Final Model

Recommended model:

```yaml
model_type: CatBoostClassifier
depth: 3
learning_rate: 0.03
iterations: 300
l2_leaf_reg: 10
loss_function: Logloss
eval_metric: Logloss
random_seed: 42
allow_writing_files: false
```

Reasoning:

* Shallow depth limits overfitting.
* Expert outputs carry most of the high-level signal.
* The final model should learn calibration and interaction effects, not become a new complex feature model.

Optional ablation:

```yaml
depth_values: [2, 3, 4]
l2_leaf_reg_values: [5, 10, 20]
```

The first run should keep this grid small.

---

## 7. Validation and Threshold Search

Use the same chronological development/validation split as the previous hybrid experiment.

Important leakage rule:

* The continuation and reversal expert predictions used as stacker features must be generated in a way that does not train and predict on the same rows for the stacker training set.

Recommended implementation:

1. For validation rows, use expert models trained on development rows only.
2. For development stacker training rows, create out-of-fold expert predictions using chronological folds inside development.
3. Train the final stacker on development rows with out-of-fold expert predictions.
4. Evaluate on validation rows using expert predictions from models trained only on development.
5. Search final `t_up` / `t_down` on validation to maximize `selection_score` subject to `coverage >= 0.70`.

If out-of-fold expert predictions are too expensive for the first pass, the fallback must be explicitly marked as a diagnostic-only run and cannot be claimed as an accepted improvement.

Threshold search must report every required metric:

```text
sample_count
coverage
precision_up
precision_down
balanced_precision
all_sample_accuracy
accepted_sample_accuracy
share_up_predictions
share_down_predictions
selected_t_up
selected_t_down
accepted_count
up_prediction_count
down_prediction_count
roc_auc
brier_score
log_loss
utility
downside_risk
selection_score
up_signal_count
down_signal_count
total_signal_count
signal_coverage
overall_signal_accuracy
```

---

## 8. Proposed Experiments

### Experiment A: Expert-Only Stacker

Feature set:

```text
expert output features
first_minute_side features
day/session features
same/opposite-side indicators
```

Purpose:

* Test whether the two experts can be combined without extra market features.
* Lowest leakage risk and easiest interpretation.

### Experiment B: Expert + Top Market Features Stacker

Feature set:

```text
Experiment A features
top 50 important pre-decision market features
```

Purpose:

* Let the final model learn when expert probabilities should be trusted.
* Still keep the second-level model compact.

### Experiment C: Expert + Market Features With Regularization Grid

Feature set:

```text
same as Experiment B
```

Small grid:

```yaml
depth: [2, 3, 4]
l2_leaf_reg: [5, 10, 20]
```

Purpose:

* Check whether the stacker needs slightly more interaction capacity.
* Avoid a broad model search until A/B shows promise.

---

## 9. Expected Outputs

Create a new experiment:

```text
20260612_catboost_continuation_reversal_stacked_meta_model
```

Expected files:

```text
experiments/configs/20260612_catboost_continuation_reversal_stacked_meta_model.yaml
scripts/analysis/catboost_continuation_reversal_stacked_meta_model.py
artifacts/data_v2/reports/reversal_hybrid/20260612_catboost_continuation_reversal_stacked_meta_model/report.json
artifacts/data_v2/reports/reversal_hybrid/20260612_catboost_continuation_reversal_stacked_meta_model/threshold_search_summary.csv
artifacts/data_v2/reports/reversal_hybrid/20260612_catboost_continuation_reversal_stacked_meta_model/stacked_feature_importance.csv
artifacts/data_v2/reports/reversal_hybrid/20260612_catboost_continuation_reversal_stacked_meta_model/validation_audit.csv
```

The report must include:

```text
git_commit
config_path
report_path
primary_metric
signal_coverage
coverage_constraint_satisfied
deploy_training_mode
offline_validation_metric_source
baseline_comparison
leakage_feature_check
stacker_feature_list
expert_prediction_generation_method
```

No deploy artifact should be regenerated unless the validation result beats the accepted baseline under the coverage constraint.

---

## 10. Risks

### 10.1 Stacker Leakage

The biggest risk is training the stacker on expert predictions generated by experts that were trained on the same rows. That would inflate development performance and may distort threshold search.

Mitigation:

* Use chronological out-of-fold expert predictions for development stacker rows.
* Use development-trained experts only for validation predictions.

### 10.2 Coverage Collapse

The model may become very selective if threshold search chases high accuracy.

Mitigation:

* Keep `coverage >= 0.70` as a hard threshold-search constraint.
* Rank only candidates satisfying the coverage constraint.
* Report best unconstrained diagnostic separately, not as the official result.

### 10.3 Expert Features Overpower Raw Market Features

The stacker could learn to ignore reversal features or simply reproduce the continuation expert.

Mitigation:

* Report feature importance.
* Report metrics by:
  * continuation actual regime
  * reversal actual regime
  * first_minute_side
  * session
  * expert agreement/disagreement bucket

### 10.4 Overfitting From Too Many Raw Features

Adding too many market features can turn the experiment into another base-model search.

Mitigation:

* Start with expert-only stacker.
* Add only a small top-feature allowlist.
* Keep the model shallow and regularized.

---

## 11. Items Requiring Confirmation

Please confirm the following before implementation:

1. Is this experiment allowed to train a final stacked model with target `UP/DOWN`, while keeping the original Polymarket resolved label unchanged? yes, use Polymarket resolved label

2. For the official implementation, should development stacker features use chronological out-of-fold expert predictions? This is slower but is the cleanest way to avoid stacking leakage.for example,model A predicts using features ending at time0,model B predicts using features ending at time0,stacking model using model A output and model B output and more features ending at time0

3. Should the first run include only `Experiment A` and `Experiment B`, or should it also include the small regularization grid in `Experiment C`? include experiment C

4. For market features in Experiment B/C, should I use a fixed allowlist from existing feature names, or derive top features from the previous expert/base model importance files? derive top features from the previous expert/base model importance files 

5. Should the final stacked threshold search use the existing baseline threshold grid range, or a wider grid around the stacked probabilities? use exsisting baseline threshold

6. Should `continuation_accept` and `reversal_accept` be included as soft features only, with no hard rule that accepted experts must be followed? yes

7. Should the official acceptance still require `coverage >= 0.70` and validation `selection_score > 0.5748509217`, with no deploy artifact regenerated unless both are true?yes

8. Should I create a completed-experiment git commit after the implementation and run, including the config, script, and report artifacts?yes


