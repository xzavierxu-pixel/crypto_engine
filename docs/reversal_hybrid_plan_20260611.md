# Reversal Hybrid Plan - 2026-06-11

## 1. Current Conclusion

The `first_minute_follow` experiments show a real reversal signal, but the follow model is not accurate enough to replace the baseline or to control a large share of decisions.

Current validation comparison:

```text
baseline / new feature experiment:
  training_target:                  polymarket_direction
  feature_count:                    1805
  validation utility:               0.2999464381
  selection_score:                  0.5382641920
  coverage:                         0.9209962507
  accepted_sample_accuracy:         0.6628380343
  continuation_accepted_accuracy:   0.9781118727
  reversal_accepted_accuracy:       0.0573248408

follow + reversal weighting:
  training_target:                  first_minute_follow
  feature_count:                    1805
  validation utility:               0.2366095340
  selection_score:                  0.3935070457
  coverage:                         0.9596946974
  accepted_sample_accuracy:         0.6232733361
  continuation_accepted_accuracy:   0.7695605573
  reversal_accepted_accuracy:       0.3505195843
```

The key trade-off is clear:

```text
follow model improves reversal recognition,
but creates too many false reversal calls on true continuation samples.
```

Therefore, the next step should not be "use the follow model more". The next step should be:

```text
keep baseline as the main decision model,
use follow only as a conservative risk gate,
and separately improve follow model precision before allowing override.
```

## 2. Why Follow Accuracy Is Too Low

The follow model predicts:

```text
p_follow = P(final resolved side == first_minute_side)
```

This is useful because it directly targets the continuation/reversal question. However, this target is also harder and noisier than final UP/DOWN in this dataset.

The baseline has learned a very strong default:

```text
first-minute continuation is usually right.
```

That creates excellent continuation performance:

```text
baseline continuation_accepted_accuracy: 0.9781118727
```

The follow model is trained to challenge that default. It detects some true reversal cases, but it also overreacts to features that look like reversal pressure:

```text
large first-minute impulse
high short-term volatility
micro realized variance spikes
HTF volatility deltas
flow / pressure instability
```

Many of these conditions are not pure reversal signals. They can also be strong continuation signals. This is the main reason continuation accuracy collapses under follow control.

In other words:

```text
the follow model's recall for reversal improved,
but its precision for "this is really a reversal" is still too low.
```

That is exactly why direct override is dangerous.

## 3. Diagnostic Evidence

On baseline-accepted validation samples:

```text
baseline accepted all:
  count:        6878
  baseline acc: 0.662838
  follow acc:   0.611079

baseline accepted reversal:
  count:        2355
  baseline acc: 0.057325
  follow acc:   0.284926

baseline accepted continuation:
  count:        4523
  baseline acc: 0.978112
  follow acc:   0.780898
```

Disagreement analysis:

```text
baseline wrong, follow right:
  count:          601
  reversal_share: 0.953411

baseline right, follow wrong:
  count:          820
  reversal_share: 0.035366
```

This is the central problem:

```text
follow fixes many reversal errors,
but it creates even more continuation errors.
```

So the goal is not to maximize follow coverage. The goal is to find the subset where follow is precise enough to be useful.

## 4. Four-Bucket Backtest

Definitions:

```text
fm_side     = first_minute_side
base_side   = baseline accepted side
follow_side = fm_side if p_follow >= 0.5 else opposite(fm_side)
```

Validation bucket results:

```text
bucket 1: base_side == fm_side, follow_side == fm_side
  count:          5176
  accuracy:       0.689722
  reversal_share: 0.310278
  avg_p_base:     0.534621
  avg_p_follow:   0.728907

bucket 2: base_side == fm_side, follow_side != fm_side
  count:          1468
  accuracy:       0.581744
  reversal_share: 0.418256
  avg_p_base:     0.493812
  avg_p_follow:   0.431642

bucket 3: base_side != fm_side, follow_side == base_side
  count:          197
  accuracy:       0.583756
  reversal_share: 0.583756
  avg_p_base:     0.492263
  avg_p_follow:   0.411022

bucket 4: base_side != fm_side, follow_side != base_side
  count:          37
  accuracy:       0.540541
  reversal_share: 0.540541
  avg_p_base:     0.483366
  avg_p_follow:   0.582513
```

Interpretation:

```text
bucket 1:
  best bucket; keep trading.

bucket 2:
  baseline says continuation but follow suspects reversal.
  accuracy is much worse than bucket 1.
  this is a good abstain candidate.

bucket 3:
  baseline already says reversal and follow confirms reversal.
  this is the best candidate for true reversal alpha.
  however count is only 197, so it is not enough by itself.

bucket 4:
  disagreement bucket.
  small count and weak accuracy; abstain.
```

Important limitation:

`validation_predictions.parquet` does not contain Polymarket entry price. The four-bucket analysis above reports classification accuracy and reversal share only. Average entry price and true PnL require joining the relevant market price / order book snapshot at decision time.

## 5. Recommended Decision Rule

First implementation should be abstain-only, not override.

Recommended initial rule:

```python
fm_side = first_minute_side
base_side = direction_from_p_base(p_base)
follow_side = fm_side if p_follow >= 0.5 else opposite(fm_side)

if base_side is None:
    signal = None

elif base_side == fm_side:
    # baseline says continuation
    if follow_side != fm_side and p_follow <= p_follow_cutoff and abs(p_base - 0.5) <= base_band:
        signal = None
    else:
        signal = base_side

else:
    # baseline already says reversal
    if follow_side == base_side and p_follow <= p_follow_cutoff:
        signal = base_side
    else:
        signal = None
```

Initial grid:

```text
p_follow_cutoff: 0.35, 0.40, 0.45, 0.50
base_band:       0.03, 0.05, 0.07, 0.10
mode:            abstain only
```

Why abstain first:

```text
The follow model is not accurate enough to flip many baseline continuation decisions.
But it is useful as a warning that the continuation trade may be lower quality.
```

Direct override should only be allowed after the follow model's reversal precision improves materially.

## 6. How To Improve Follow Model Accuracy

The follow model problem should be treated as a precision problem, not only a recall problem.

### 6.1 Train follow as a high-precision reversal detector

Current follow target is symmetric:

```text
y_follow = 1 if final side == first_minute_side else 0
```

For overlay usage, the more important event is:

```text
y_reversal = 1 if final side != first_minute_side else 0
```

Recommended experiment:

```text
train a reversal-risk auxiliary model,
optimize for high precision at low coverage,
use it only as a gate or overlay feature.
```

Do not optimize this auxiliary model for generic accuracy. Generic accuracy will reward predicting continuation too often. The useful objective is:

```text
high precision among accepted reversal-risk alerts
low false-reversal rate on continuation samples
positive contribution to final validation utility
```

Suggested reporting:

```text
reversal_alert_count
reversal_alert_precision
reversal_alert_recall
continuation_false_alert_rate
alert_reversal_share
alert_bucket_utility_delta
```

### 6.2 Use asymmetric weights

The previous reversal weighting improved reversal accuracy but damaged continuation too much.

The next weighting should penalize false reversal alerts more directly:

```text
false reversal on true continuation: high penalty
missed reversal: medium penalty
correct continuation: normal penalty
correct reversal: normal or moderately boosted
```

The model should learn:

```text
only call reversal when the evidence is strong.
```

This is different from simply increasing reversal class weight, which encourages more reversal calls and can destroy continuation accuracy.

### 6.3 Calibrate p_follow by bucket, not globally

Global calibration can hide the fact that `p_follow` has different meaning in different regimes.

Recommended calibration slices:

```text
base_side == fm_side
base_side != fm_side
abs(p_base - 0.5) buckets
first_minute_return magnitude buckets
volatility regime buckets
market session / time buckets
```

The useful question is not:

```text
is p_follow calibrated overall?
```

The useful question is:

```text
when p_follow is low inside bucket 2 or bucket 3,
does it actually identify reversal with enough precision?
```

### 6.4 Feature ablation for false reversal sources

The follow model appears likely to overuse broad volatility / realized variance features.

Potential false-reversal drivers:

```text
htf_rv_15m_delta_1
micro_rv_5s
micro_rv_10s
rv_30_delta_1
fm_abs_ret
```

These features can mean:

```text
reversal risk,
or simply strong continuation pressure.
```

Recommended ablation:

```text
Experiment F1:
  remove or downweight broad volatility features.

Experiment F2:
  keep impulse-quality, rejection, and flow-divergence features.

Experiment F3:
  train with feature selection focused on bucket 2 and bucket 3 precision.
```

The target metric is not overall follow accuracy. The target metric is:

```text
lower false reversal rate on continuation samples,
while preserving enough true reversal alerts to improve final utility.
```

### 6.5 Add a small meta-gate instead of trusting raw follow

The follow model should produce an input to a gate, not the final decision.

Recommended meta-gate inputs:

```text
p_base
abs(p_base - 0.5)
p_follow
abs(p_follow - 0.5)
base_side == fm_side
follow_side == fm_side
base_side == follow_side
first_minute_return
abs(first_minute_return)
fm_reversal_pressure_score
fm_continuation_pressure_score
micro / flow divergence features
volatility regime features
```

Gate output:

```text
KEEP_BASE
ABSTAIN
ALLOW_REVERSAL_OVERRIDE
```

Training rule:

```text
use chronological out-of-fold predictions for p_base and p_follow.
validation is only for final threshold selection and acceptance reporting.
```

The gate should be intentionally small:

```text
logistic regression
shallow LightGBM
or a monotonic score rule
```

Do not add a large second-stage model until a small gate proves the effect.

## 7. Candidate Experiment Order

### Experiment A: Four-Bucket Abstain Gate

Purpose:

```text
use follow only to remove bad baseline trades,
not to flip direction.
```

Primary metric:

```text
validation utility
```

Guardrails:

```text
coverage >= 0.70
continuation_accepted_accuracy >= 0.958
accepted_sample_accuracy > 0.50
positive utility
```

Required report:

```text
count by bucket
accuracy by bucket
reversal_share by bucket
utility by bucket
overall utility
overall selection_score
coverage
accepted_count
continuation_accepted_accuracy
reversal_accepted_accuracy
```

### Experiment B: Reversal-Precision Auxiliary Model

Purpose:

```text
improve follow model precision before any override.
```

Model target:

```text
y_reversal = final side != first_minute_side
```

Optimization target:

```text
high precision reversal alerts at low alert coverage
```

Reject if:

```text
continuation false-alert rate remains high,
or final utility does not improve after applying the gate.
```

### Experiment C: Follow Feature Ablation

Purpose:

```text
reduce false reversal alerts caused by volatility-only features.
```

Compare:

```text
current 1805 features
minus broad volatility features
only first-minute impulse / rejection / flow-divergence features
selected features from bucket-level importance
```

Required slice metrics:

```text
bucket 2 accuracy
bucket 3 accuracy
continuation false reversal rate
reversal alert precision
final utility under gate
```

### Experiment D: OOF Meta-Gate

Purpose:

```text
learn when to keep baseline, abstain, or allow reversal override.
```

Constraint:

```text
all base and follow probabilities must be out-of-fold on development rows.
validation must not be used to train the gate.
```

Acceptance standard:

```text
utility > baseline utility
coverage >= 0.70
continuation_accepted_accuracy does not drop more than about 0.02
reversal_accepted_accuracy improves materially
```

## 8. Acceptance Criteria

The next accepted experiment should beat the current validation baseline:

```text
baseline utility:                    0.2999464381
baseline selection_score:            0.5382641920
baseline coverage:                   0.9209962507
baseline accepted_sample_accuracy:   0.6628380343
baseline continuation accuracy:      0.9781118727
baseline reversal accuracy:          0.0573248408
```

Minimum acceptance:

```text
validation utility > 0.2999464381
coverage >= 0.70
accepted_sample_accuracy > 0.50
continuation_accepted_accuracy >= 0.958
reversal_accepted_accuracy > 0.08
```

Preferred target:

```text
validation utility >= 0.305
selection_score >= 0.54
continuation_accepted_accuracy >= 0.965
reversal_accepted_accuracy >= 0.10
```

Reject any result where:

```text
reversal accuracy improves only because continuation accuracy collapses,
follow model controls a large share of decisions without precision proof,
or validation utility does not improve.
```

## 9. Practical Recommendation

The current direction is correct, but the follow model should be demoted from "decision model" to "risk evidence".

Best near-term path:

```text
1. baseline remains the only primary direction model.
2. follow model is used to bucket and abstain.
3. no broad override until follow reversal precision improves.
4. train a reversal-precision auxiliary model.
5. add a small OOF meta-gate only after the manual bucket gate is validated.
```

The main technical challenge is not discovering reversals. The current follow model already finds some reversal signal.

The challenge is:

```text
separating true reversal pressure from noisy high-volatility continuation.
```

That is why the next work should focus on false-reversal reduction, bucket-level calibration, and low-coverage high-precision reversal alerts.
