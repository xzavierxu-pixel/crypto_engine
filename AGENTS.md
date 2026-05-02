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
balanced_precision = (precision_up + precision_down) / 2
```

Subject to:

```text
signal_coverage >= 0.60
```

Where:

```text
precision_up     = correct UP signals / all UP signals
precision_down   = correct DOWN signals / all DOWN signals
signal_coverage  = emitted_signal_count / total_available_samples
```

AUC, logloss, Brier, F1, and generic accuracy are diagnostics only. Do not optimize primarily for them.

Reject any result where coverage falls below 0.60, even if precision improves.

Reject any result where one side nearly disappears.

---

## Required metrics

Every training, validation, threshold search, or optimization run must report:

```text
precision_up
precision_down
balanced_precision
up_signal_count
down_signal_count
total_signal_count
signal_coverage
overall_signal_accuracy
```

Minimum valid result:

```yaml
evaluation:
  min_up_signals: 50
  min_down_signals: 50
  min_total_signals: 150
  min_signal_coverage: 0.60
```

Optimization ranking:

1. holdout balanced_precision with coverage >= 0.60
2. validation balanced_precision with coverage >= 0.60
3. balance between precision_up and precision_down
4. signal_count and coverage
5. stability across time splits
6. leakage risk and implementation simplicity
7. AUC / logloss / Brier as diagnostics only

---

## Non-negotiable rules

- Keep offline and online logic consistent.
- Keep all business parameters in the unified config.
- Do not duplicate feature logic.
- Do not duplicate label logic.
- Keep Freqtrade strategy thin.
- Execution must not recompute BTC features.
- Do not silently change label, horizon, timestamp alignment, or feature semantics.
- Do not optimize on holdout data.
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
  up_threshold: null
  down_threshold: null
```

If null, load thresholds from the artifact.

---

## Threshold tuning

Threshold search must maximize balanced_precision subject to all validity constraints:

```text
up_signal_count >= min_up_signals
down_signal_count >= min_down_signals
total_signal_count >= min_total_signals
signal_coverage >= 0.60
```

For each candidate, report:

```text
precision_up
precision_down
balanced_precision
up_signal_count
down_signal_count
total_signal_count
signal_coverage
```

Tie-breakers:

1. higher signal_coverage
2. higher total_signal_count
3. smaller precision_up / precision_down gap
4. simpler thresholds
5. better time-split stability

---

## Validation protocol

Use time-based splits only.

Preferred structure:

```text
train / development
validation / threshold tuning
holdout / final evaluation
```

Rules:

- Tune thresholds on validation only.
- Report final performance on holdout.
- Never tune on holdout.
- Clearly mark validation metrics as threshold-tuned and optimistic.
- If no holdout exists, add one only when the change is simple, local, and tested.

---

## Leakage and consistency checks

Before claiming improvement, verify:

- no future OHLCV or future return columns are features
- no label-derived columns are features
- no `target`, `future_close`, `abs_return`, `signed_return`, `stage1_target`, or `stage2_target` in features
- rolling features do not include future bars
- joins use only data available at prediction time
- scalers, imputers, encoders, selectors are not fitted on validation or holdout
- offline, validation, holdout, inference, and signal generation use the same shared feature builder

If a feature cannot be built online at decision time, it must not be used offline.

---

## Allowed improvements

Prefer these before adding complex models:

- threshold tuning for balanced_precision with coverage >= 0.60
- validation / holdout discipline
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
reason it may improve balanced_precision
how coverage >= 0.60 is preserved
tests or reports to verify it
```

After changing code, report:

```text
before / after balanced_precision
before / after precision_up and precision_down
before / after signal_count
before / after signal_coverage
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

---

## Config rules

All business parameters must live in the unified settings file.

Example:

```yaml
label:
  horizon_minutes: 5
  rule: "close_t0_plus_4m_gte_open_t0"

evaluation:
  primary_metric: "balanced_precision"
  min_up_signals: 50
  min_down_signals: 50
  min_total_signals: 150
  min_signal_coverage: 0.60

signal:
  up_threshold: null
  down_threshold: null

model:
  type: "lightgbm"
  class_weight: null
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
- balanced_precision and signal_coverage calculation
- invalid result when coverage < 0.60
- invalid result when one side has too few signals
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
4. balanced_precision, precision_up, precision_down, signal counts, and coverage are reported
5. signal_coverage >= 0.60
6. tests pass
7. no leakage columns are used
8. result is compared against the previous baseline
9. Codex states whether balanced_precision truly improved under the coverage constraint
