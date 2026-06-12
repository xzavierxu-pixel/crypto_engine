# Requirement: Continuation + Reversal Hybrid Router Experiment

## 1. Objective

Build and evaluate two hybrid routing experiments that combine the strongest continuation expert and the strongest reversal expert.

Known evidence to record in experiment metadata:

* `artifacts\data_v2\reports\reversal_hybrid\20260611_catboost_reversal_sample_weight_search\reversal_weight_8` has the highest `reversal_accepted_accuracy` among current reversal experiments.
* `artifacts\data_v2\reports\reversal_hybrid\20260611_catboost_continuation_side_coordinate_search` has the highest `continuation_accepted_accuracy` among all current experiments.

Goal:

* Preserve the strong continuation performance from the continuation-side model.
* Add reversal coverage using `reversal_weight_8`.
* Avoid the previous seesaw issue where reversal accuracy improves only by destroying continuation accuracy.
* Optimize overall hybrid accuracy / utility while monitoring regime balance and downside risk.

---

## 2. Input Models

### 2.1 Continuation Expert

Source:

```text
artifacts\data_v2\reports\reversal_hybrid\20260611_catboost_continuation_side_coordinate_search
```

Role:

```text
Continuation expert
```

Decision constraint:

* If `first_minute_side == YES`, only allow `UP`.
* If `first_minute_side == NO`, only allow `DOWN`.
* Otherwise `ABSTAIN`.

Use its existing per-regime threshold logic from the coordinate search result.

---

### 2.2 Reversal Expert

Source:

```text
artifacts\data_v2\reports\reversal_hybrid\20260611_catboost_reversal_sample_weight_search\reversal_weight_8
```

Role:

```text
Reversal expert candidate
```

Important constraint:

Although `reversal_weight_8` has the best `reversal_accepted_accuracy`, it must be used as a reversal-only expert in the hybrid.

Decision constraint:

* If `first_minute_side == YES`, reversal direction is `DOWN`.

  * Only allow `DOWN` when `reversal_p_up <= reversal_t_down`.
* If `first_minute_side == NO`, reversal direction is `UP`.

  * Only allow `UP` when `reversal_p_up >= reversal_t_up`.
* Otherwise `ABSTAIN`.

Do not allow the reversal expert to trade in the same direction as `first_minute_side`.

---

## 3. Experiment 3: Conflict Margin Hybrid

Create experiment:

```text
20260612_catboost_continuation_reversal_conflict_margin_hybrid
```

### 3.1 Base Decision Logic

For each validation sample:

```python
cont_decision = continuation_expert_decision(row)
rev_decision = reversal_expert_reversal_only_decision(row)

cont_accept = cont_decision != "ABSTAIN"
rev_accept = rev_decision != "ABSTAIN"
```

Decision rules:

```python
if cont_accept and not rev_accept:
    final_decision = cont_decision
    used_expert = "continuation"
    routing_reason = "cont_only"

elif rev_accept and not cont_accept:
    final_decision = rev_decision
    used_expert = "reversal"
    routing_reason = "rev_only"

elif not cont_accept and not rev_accept:
    final_decision = "ABSTAIN"
    used_expert = "abstain"
    routing_reason = "both_abstain"

else:
    # both experts accept but disagree by construction
    apply confidence margin logic
```

### 3.2 Conflict Margin Logic

When both experts accept:

```python
cont_conf = abs(continuation_p_up - 0.5)
rev_conf = abs(reversal_p_up - 0.5)

if cont_conf - rev_conf >= conflict_margin:
    final_decision = cont_decision
    used_expert = "continuation"
    routing_reason = "conflict_cont_win"

elif rev_conf - cont_conf >= conflict_margin:
    final_decision = rev_decision
    used_expert = "reversal"
    routing_reason = "conflict_rev_win"

else:
    final_decision = "ABSTAIN"
    used_expert = "abstain"
    routing_reason = "conflict_abstain"
```

### 3.3 Search Space

Use grid search over:

```python
conflict_margin_values = [0.03, 0.05, 0.07, 0.10]
```

Optional: also test wider values if needed:

```python
conflict_margin_values_extra = [0.12, 0.15]
```

---

## 4. Experiment 4: Gate Hybrid

Create experiment:

```text
20260612_catboost_continuation_reversal_gate_hybrid
```

### 4.1 Gate Target

Train a separate gate model.

The gate does not predict final `UP` / `DOWN`.

It predicts whether the sample is continuation or reversal.

Define:

```python
first_minute_direction = 1 if first_minute_side == "YES" else 0
actual_direction = 1 if target == "UP" else 0

is_continuation = int(first_minute_direction == actual_direction)
is_reversal = 1 - is_continuation
```

Gate target:

```text
is_continuation
```

Gate output:

```python
p_continuation = gate_model.predict_proba(X_gate)[:, 1]
p_reversal = 1 - p_continuation
```

### 4.2 Gate Features

Use only features available before trade decision.

Recommended feature groups:

```text
first_minute_side
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
early volatility / range / liquidity features
calendar day
session
```

Do not use any future-looking or post-outcome features.

### 4.3 Gate Routing Logic

For each validation sample:

```python
if p_continuation >= tau_cont:
    use continuation expert

elif p_reversal >= tau_rev:
    use reversal expert

else:
    ABSTAIN
```

Expert constraints still apply:

* Continuation expert can only trade same-side as `first_minute_side`.
* Reversal expert can only trade opposite-side from `first_minute_side`.

If the selected expert returns `ABSTAIN`, final decision remains `ABSTAIN`.

### 4.4 Search Space

Search routing thresholds:

```python
tau_cont_values = [0.60, 0.65, 0.70, 0.75]
tau_rev_values = [0.65, 0.70, 0.75, 0.80]
```

Reversal threshold should generally be equal to or stricter than continuation threshold.

---

## 5. Optimization Objective

Do not optimize only `accepted_sample_accuracy`, because it can recreate the continuation/reversal seesaw issue.

Primary objective:

```python
score = utility \
        - 0.5 * max(0, 0.60 - continuation_accepted_accuracy) \
        - 0.7 * max(0, 0.55 - reversal_accepted_accuracy) \
        - 0.3 * downside_risk
```

Also compute an accuracy-focused score:

```python
balanced_score = accepted_sample_accuracy \
                 - 0.4 * abs(continuation_accepted_accuracy - reversal_accepted_accuracy) \
                 - 0.3 * downside_risk
```

Hard constraints:

```python
coverage >= 0.50
accepted_count >= minimum_accepted_count
continuation_accepted_count > 0
reversal_accepted_count > 0
```

Optional stricter versions:

```python
coverage >= 0.60
continuation_accepted_accuracy >= 0.60
reversal_accepted_accuracy >= 0.55
downside_risk <= 0.50
```

Report both constrained and unconstrained best variants.

---

## 6. Required Metrics

For every tested variant, output the following metrics:

```text
accepted_count
coverage
accepted_sample_accuracy
all_sample_accuracy
utility
selection_score
balanced_score
downside_risk

continuation_accepted_count
continuation_accepted_accuracy
continuation_coverage
continuation_sample_count

reversal_accepted_count
reversal_accepted_accuracy
reversal_coverage
reversal_sample_count

cont_model_used_count
rev_model_used_count
abstain_count
conflict_abstain_count
both_abstain_count

conflict_cont_win_count
conflict_rev_win_count
cont_only_count
rev_only_count
```

Also output prediction direction diagnostics:

```text
up_prediction_count
down_prediction_count
share_up_predictions
share_down_predictions
precision_up
precision_down
```

---

## 7. Required Per-Sample Audit Columns

For every validation sample, save an audit CSV with:

```text
timestamp
day
session
first_minute_side
actual_direction
actual_regime              # continuation / reversal

continuation_p_up
continuation_decision
continuation_accept
continuation_confidence

reversal_p_up
reversal_decision
reversal_accept
reversal_confidence

p_continuation             # only for gate experiment
p_reversal                 # only for gate experiment

final_decision
final_accept
final_correct
used_expert                # continuation / reversal / abstain
routing_reason             # cont_only / rev_only / conflict_cont_win / conflict_rev_win / conflict_abstain / both_abstain / gate_cont / gate_rev / gate_abstain
```

---

## 8. Output Files

Create output directory:

```text
artifacts\data_v2\reports\reversal_hybrid\20260612_catboost_continuation_reversal_hybrid_router
```

Save:

```text
experiment3_conflict_margin_summary.csv
experiment3_conflict_margin_best_metrics.json
experiment3_conflict_margin_audit.csv

experiment4_gate_hybrid_summary.csv
experiment4_gate_hybrid_best_metrics.json
experiment4_gate_hybrid_audit.csv

hybrid_experiment_metadata.json
```

The metadata file must include:

```json
{
  "continuation_expert_source": "artifacts\\data_v2\\reports\\reversal_hybrid\\20260611_catboost_continuation_side_coordinate_search",
  "continuation_expert_reason": "highest continuation_accepted_accuracy among current experiments",
  "reversal_expert_source": "artifacts\\data_v2\\reports\\reversal_hybrid\\20260611_catboost_reversal_sample_weight_search\\reversal_weight_8",
  "reversal_expert_reason": "highest reversal_accepted_accuracy among current reversal experiments",
  "hybrid_goal": "combine continuation expert and reversal expert using routing instead of probability averaging"
}
```

---

## 9. Success Criteria

The hybrid is considered successful if it improves over the standalone continuation expert on at least one of the following without destroying regime balance:

```text
higher utility
lower downside_risk
higher balanced_score
meaningful reversal_accepted_count with reversal_accepted_accuracy >= 0.55
continuation_accepted_accuracy remains acceptable
```

Do not select a variant only because it has the highest `accepted_sample_accuracy` if it has near-zero reversal coverage or near-zero continuation coverage.

---

## 10. Implementation Notes

* Do not average probabilities from the two experts.
* Use routing, not blending.
* Preserve expert specialization:

  * continuation expert = same-side only
  * reversal expert = opposite-side only
* If both experts fire and confidence is close, abstain.
* For gate hybrid, the gate decides which expert to consult, but the expert must still pass its own threshold and direction constraint.
* Always save per-sample audit output so errors can be inspected by regime and routing reason.
