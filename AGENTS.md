# AGENTS.md

## Context

This is an existing, complete BTC/USDT 5-minute direction prediction project. Do not treat it as a greenfield build.

Codex should improve the current system with small, measurable, low-risk changes. Do not rewrite the architecture unless clearly necessary.

Core architecture:

- shared core in `src/` for features, labels, schemas, training, inference
- thin Freqtrade / FreqAI adapter
- plugin-friendly model layer
- separate Polymarket execution layer
- shared core is the single source of truth

---

## Primary objective

Optimize:

```text
Score = Utility / Downside Risk
      = coverage * (2 * accepted_sample_accuracy - 1)
        / sqrt(coverage * (1 - accepted_sample_accuracy))
```

Subject to:

```text
coverage >= 0.40
```

Where:

```text
coverage                 = accepted_count / total_available_samples
accepted_sample_accuracy = correct accepted predictions / accepted_count
Utility                  = coverage * (2 * accepted_sample_accuracy - 1)
Downside Risk            = sqrt(coverage * (1 - accepted_sample_accuracy))
```

YES/NO balance, AUC, logloss, Brier, F1, and generic accuracy are diagnostics only. Do not optimize primarily for them.

Reject any result where coverage falls below 0.40, even if score improves.

YES/NO balance should be recorded for diagnosis, but it is not part of the objective.

---

## Required metrics

Every training, validation, threshold search, or optimization run must report:

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

Every experiment `report.json` must include top-level train and validation sections with these fields:

```json
{
  "train_metrics": {
    "sample_count": 0.0,
    "coverage": 0.0,
    "precision_up": 0.0,
    "precision_down": 0.0,
    "balanced_precision": 0.0,
    "all_sample_accuracy": 0.0,
    "accepted_sample_accuracy": 0.0,
    "share_up_predictions": 0.0,
    "share_down_predictions": 0.0,
    "selected_t_up": 0.0,
    "selected_t_down": 0.0,
    "accepted_count": 0.0,
    "up_prediction_count": 0.0,
    "down_prediction_count": 0.0,
    "roc_auc": 0.0,
    "brier_score": 0.0,
    "log_loss": 0.0,
    "utility": 0.0,
    "downside_risk": 0.0,
    "selection_score": 0.0
  },
  "train_window": {
    "row_count": 0,
    "start": "ISO-8601 timestamp",
    "end": "ISO-8601 timestamp"
  },
  "validation_metrics": {
    "sample_count": 0.0,
    "coverage": 0.0,
    "precision_up": 0.0,
    "precision_down": 0.0,
    "balanced_precision": 0.0,
    "all_sample_accuracy": 0.0,
    "accepted_sample_accuracy": 0.0,
    "share_up_predictions": 0.0,
    "share_down_predictions": 0.0,
    "selected_t_up": 0.0,
    "selected_t_down": 0.0,
    "accepted_count": 0.0,
    "up_prediction_count": 0.0,
    "down_prediction_count": 0.0,
    "roc_auc": 0.0,
    "brier_score": 0.0,
    "log_loss": 0.0,
    "utility": 0.0,
    "downside_risk": 0.0,
    "selection_score": 0.0
  },
  "validation_window": {
    "row_count": 0,
    "start": "ISO-8601 timestamp",
    "end": "ISO-8601 timestamp"
  }
}
```

The legacy aliases `up_signal_count`, `down_signal_count`, `total_signal_count`, `signal_coverage`, and `overall_signal_accuracy` may also be reported for compatibility, but they must not replace the explicit `*_prediction_count`, `accepted_count`, `coverage`, and `accepted_sample_accuracy` fields above.

Minimum valid result:

```yaml
objective:
  min_coverage: 0.40

threshold_search:
  hard_constraint: coverage_only
```

Optimization ranking:

1. validation selection_score with coverage >= objective.min_coverage
2. positive utility and accepted_sample_accuracy > 0.50
3. coverage and accepted_count
4. stability across time splits
5. leakage risk and implementation simplicity
6. YES/NO balance, AUC / logloss / Brier as diagnostics only

---

## Non-negotiable rules

- Keep offline and online logic consistent.
- Keep all business parameters in the unified config.
- Do not duplicate feature logic.
- Do not duplicate label logic.
- Keep Freqtrade strategy thin.
- Execution must not recompute BTC features.
- Do not silently change label, horizon, timestamp alignment, or feature semantics.
- Do not introduce future-looking features.
- Prefer small, local, testable changes.
- Do not add advanced models before the LightGBM baseline is clean, tested, and evaluated.

---

## Existing baseline

The project already supports:

- BTC/USDT
- 1m data
- 1s data
- 5m horizon
- shared feature builders
- shared label builders
- LightGBM baseline
- unified settings file
- model artifacts
- inference path
- derivatives feature inputs
- tests and training reports

Codex should focus on controlled optimization, validation, leakage checks, threshold tuning, reporting, and ablation.

Current validation baseline:

```yaml
experiment_id: 20260502_balanced_precision_holdout
config_path: experiments/configs/20260502_balanced_precision_holdout.yaml
report_path: artifacts/data_v2/experiments/20260502_balanced_precision_holdout/report.json
split_used_for_baseline: validation
t_up: 0.56
t_down: 0.50
precision_up: 0.6060606061
precision_down: 0.5193929174
balanced_precision: 0.5627267617
up_signal_count: 165
down_signal_count: 2372
total_signal_count: 2537
signal_coverage: 0.6108836985
overall_signal_accuracy: 0.5250295625
coverage_constraint_satisfied: true
```

---

## Label rule

Do not change the current label unless explicitly instructed:

```text
y = 1{close[t0 + 4m] >= open[t0]}
```

Do not change it to `close[t0 + 5m]`, `open[t0 + 5m]`, log return, thresholded return, or any other label automatically.

---

## Signal rule

For a binary model outputting `p_up`:

```text
p_down = 1 - p_up

UP signal     if p_up >= up_threshold
DOWN signal   if p_up <= down_threshold
NO-SIGNAL     otherwise
```

Thresholds must come from config or the trained artifact. Do not hardcode `0.5`.

Example:

```yaml
signal:
  policies:
    selective_binary_policy:
      t_up: null
      t_down: null
```

If null, load thresholds from the artifact.

---

## Threshold tuning

Threshold search must maximize selection_score subject to the coverage constraint:

```text
coverage >= objective.min_coverage
```

For each candidate, report:

```text
coverage
accepted_sample_accuracy
utility
downside_risk
selection_score
accepted_count
up_prediction_count
down_prediction_count
share_up_predictions
share_down_predictions
```

Tie-breakers:

1. higher utility
2. higher coverage
3. higher accepted_count
4. simpler thresholds
5. better time-split stability

---

## Validation protocol

Use time-based splits only.

Preferred structure:

```text
train / development
validation / threshold tuning and final acceptance
```

Rules:

- Tune thresholds on validation only.
- Use validation as the acceptance set for the current project workflow.
- Do not keep or add a separate holdout unless explicitly requested.
- Clearly mark validation metrics as threshold-tuned and optimistic.
- Compare future experiments against the current validation baseline recorded in this file.

---

## Leakage and consistency checks

Before claiming improvement, verify:

- no future OHLCV or future return columns are features
- no label-derived columns are features
- no `target`, `future_close`, `abs_return`, `signed_return`, `stage1_target`, or `stage2_target` in features
- rolling features do not include future bars
- joins use only data available at prediction time
- scalers, imputers, encoders, selectors are not fitted on validation
- offline, validation, inference, and signal generation use the same shared feature builder

If a feature cannot be built online at decision time, it must not be used offline.

---

## Allowed improvements

Prefer these before adding complex models:

- threshold tuning for selection_score with coverage >= 0.40
- validation discipline
- feature ablation
- feature importance review
- leakage removal
- class weight and sample weight experiments
- probability bucket analysis
- time-regime analysis
- conservative evaluation/reporting refactors

For any new feature pack, include loader/source, builder, registry entry, timestamp alignment, tests, online availability check, and ablation result.

---

## Experiment protocol

Before changing code, state:

```text
metric being improved
files affected
reason it may improve selection_score
how coverage >= 0.40 is preserved
tests or reports to verify it
```

After changing code, report:

```text
before / after selection_score
before / after utility and accepted_sample_accuracy
before / after signal_count
before / after coverage
coverage constraint satisfied: yes/no
```

Do not claim improvement without running or producing the relevant evaluation.

If metrics cannot be computed, say so clearly.


---

## Experiment reproducibility

Every completed experiment must be reproducible.

After each experiment run, Codex must:

1. save the exact config file used for the run
2. save or reference the training report / evaluation output
3. record the feature set, thresholds, split dates, and model settings used
4. create a git commit for the completed experiment
5. include the commit hash in the experiment summary

Do not overwrite the only copy of an experiment config.

If the main config is modified for an experiment, copy it to an experiment-specific path first, for example:

```text
experiments/configs/<timestamp>_<short_description>.yaml
```

The experiment summary must include:

```text
git_commit
config_path
report_path
primary_metric
signal_coverage
coverage_constraint_satisfied
```

A result is not considered complete unless the corresponding config and commit can reproduce it.

Use an experiment-specific commit for each completed experiment. Do not mix unrelated user edits into that commit.

---

## Config rules

All business parameters must live in the unified settings file.

Example:

```yaml
objective:
  label: settlement_direction
  optimize_metric: selection_score
  min_coverage: 0.40
  tie_breaker_metric: coverage
  balanced_precision_tie_tolerance: 0.002

threshold_search:
  enabled: true
  t_up_min: 0.50
  t_up_max: 0.60
  t_down_min: 0.40
  t_down_max: 0.50
  step: 0.005
  enforce_min_side_share: false
  min_side_share: 0.20
  min_up_signals: 50
  min_down_signals: 50
  min_total_signals: 150

validation:
  mode: chronological_validation
  train_days: 30
  validation_days: 30

signal:
  policies:
    selective_binary_policy:
      t_up: null
      t_down: null

model:
  active_plugin: lightgbm
```

Do not hardcode thresholds, horizons, feature lists, coverage limits, or label rules.

---

## Testing requirements

Tests should cover:

- label calculation and time grid alignment
- feature exclusion of target / future columns
- offline-online feature consistency
- threshold search
- UP / DOWN / NO-SIGNAL decisions
- selection_score, utility, downside_risk, and coverage calculation
- invalid result when coverage < 0.40
- artifact save/load and artifact thresholds
- execution layer does not recompute features

---

## Shell tooling

Prefer `rtk` for verbose shell commands.

Before declaring it unavailable, verify:

```powershell
where.exe rtk
rtk --version
```

Only fall back to raw PowerShell after both checks fail.

See `@RTK.md`.

---

## Definition of done

A change is complete only when:

1. offline and online logic remain consistent
2. labels and features remain centralized
3. thresholds come from config or artifact
4. selection_score, utility, accepted_sample_accuracy, signal counts, and coverage are reported
5. coverage >= 0.40
6. tests pass
7. no leakage columns are used
8. result is compared against the previous baseline
9. Codex states whether selection_score truly improved under the coverage constraint
