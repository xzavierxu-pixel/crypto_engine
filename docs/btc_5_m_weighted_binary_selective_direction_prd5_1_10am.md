# Polymarket BTC 5m Selective Direction Requirements

## 1. Objective

Build a selective BTC 5-minute Polymarket direction model that predicts:

- UP;
- DOWN;
- ABSTAIN.

The system should keep coverage as high as possible, but coverage must not fall below 60%.

Primary optimization target:

```text
balanced_precision = (precision_up + precision_down) / 2
subject to coverage >= 0.60
```

If multiple threshold sets have similar balanced precision, prefer the one with higher coverage.

---

## 2. Business Setup

The target market is BTC 5-minute UP / DOWN settlement on Polymarket.

Assumptions:

- position is held to expiry;
- entry is around 0.5;
- payoff is effectively 1:1;
- payoff depends on final direction, not move size.

Therefore, the supervised target must remain the true settlement direction.

---

## 3. Main Architecture

Use one weighted binary model first. Probability calibration is not required in the first baseline.

```text
features
  ↓
weighted binary model
  ↓
raw p_up score
  ↓
asymmetric threshold policy
  ↓
UP / DOWN / ABSTAIN
```

The model outputs:

```text
p_up = P(settlement UP | features)
p_down = 1 - p_up
```

This avoids the inconsistency of two independent UP / DOWN probability models.

Decision policy:

```text
predict UP   if p_up >= t_up
predict DOWN if p_up <= t_down
abstain      otherwise
```

`t_up` and `t_down` are learned from validation data. They do not need to be symmetric.

---

## 4. Label Definition

Define:

```text
r = close_expiry / open_decision - 1
```

Primary label:

```text
y = 1 if r > 0 else 0
```

where:

```text
y = 1 → settlement UP
y = 0 → settlement DOWN
```

If Polymarket has a specific tie rule, use the actual market rule.

Do not replace the primary label with a 5bp strong-move label such as:

```text
UP   if close_expiry >= 1.0005 * open_decision
DOWN if close_expiry <= 0.9995 * open_decision
```

That would change the task from settlement prediction to strong-move prediction.

---

## 5. Sample Weighting

Use return magnitude only as a training weight, not as the final label.

Purpose:

- keep the true settlement target;
- reduce the influence of near-zero noisy samples;
- give stronger directional moves more training importance.

Initial weighting function:

```text
if abs(r) < 0.0001:
    weight = 0.20
elif abs(r) < 0.0005:
    weight = 0.20 + 0.80 * abs(r) / 0.0005
else:
    weight = 1.00
```

Config:

```yaml
sample_weighting:
  enabled: true
  mode: linear_ramp
  min_abs_return: 0.0001
  full_weight_abs_return: 0.0005
  min_weight: 0.20
  max_weight: 1.00
```

Boundary samples must remain in validation. Final metrics must be calculated on the full validation set.

---

## 6. Feature Requirements

The current issue is not only model choice. The feature layer must better capture short-horizon directional pressure.

### 6.1 Taker flow pressure

Initial features:

- taker_buy_ratio;
- taker_sell_ratio;
- taker_imbalance;
- rolling taker_imbalance mean;
- rolling taker_imbalance z-score;
- taker_imbalance_slope;
- signed_dollar_flow.

### 6.2 Book pressure

When best bid / ask data is available, add:

- spread_bps;
- mid_price;
- microprice;
- bid_ask_qty_imbalance;
- spread_change;
- imbalance_change;
- short-horizon mid drift.

### 6.3 Second-level microstructure

Minute-level bars may be too coarse for a 5-minute market. Add second-level aggregated features when data is available.

Do not train on raw tick data directly. Summarize sub-minute behavior into fixed decision-time features.

Initial features:

- last 5s / 10s / 30s / 60s return;
- last 5s / 10s / 30s / 60s realized volatility;
- second-level price slope into decision time;
- second-level taker buy / sell imbalance;
- second-level signed dollar flow;
- second-level volume burst;
- order book imbalance change over last 5s / 10s / 30s;
- spread widening / tightening before decision time;
- microprice drift before decision time;
- price direction flips in last 30s / 60s;
- last-second reversal flag;
- late-window acceleration / deceleration flag.

Required alignment rule:

```text
Only use information available at or before the decision timestamp.
```

### 6.4 Regime interactions

Initial interactions:

- taker_imbalance × volatility regime;
- taker_imbalance × trend strength;
- book imbalance × volatility regime;
- basis × trend strength;
- funding z-score × high-volatility flag.

### 6.5 Event-window burst

Use recent 3 to 5 bar summaries:

- consecutive bullish bar count;
- consecutive bearish bar count;
- burst volume flag;
- wick rejection count;
- compression-to-expansion flag;
- max positive single-bar return;
- max negative single-bar return;
- directional persistence score.

### 6.6 Side-specific transforms inside one model

Keep one binary model, but add directional feature transforms:

- positive_taker_imbalance;
- negative_taker_imbalance;
- bullish_burst_score;
- bearish_burst_score;
- upward_mid_drift;
- downward_mid_drift;
- positive_basis_pressure;
- negative_basis_pressure.

This lets one model learn asymmetric UP / DOWN behavior while preserving:

```text
p_up + p_down = 1
```

---

## 7. Calibration

Calibration is not required in the first baseline.

First run threshold search on raw model scores. Calibration can be tested later only if raw thresholds are unstable or probability scale becomes important for production.

Do not keep calibration only because Brier score or log loss improves. Keep it only if it improves at least one of:

- balanced_precision under coverage >= 60%;
- threshold stability;
- side-specific precision stability.

---

## 8. Threshold Search

Search over:

```text
t_up
t_down
```

Initial grid:

```text
t_up:   0.500 to 0.600, step 0.005
t_down: 0.400 to 0.500, step 0.005
```

Candidate rule:

```text
valid only if coverage >= 0.60
```

Selection rule:

1. maximize balanced_precision;
2. if tied within 0.2 percentage points, choose higher coverage;
3. monitor UP / DOWN prediction shares.

Optional side-balance guardrail:

```text
share_up_predictions >= 20% of accepted predictions
share_down_predictions >= 20% of accepted predictions
```

This guardrail can be disabled during early research.

---

## 9. Required Metrics

Every experiment must report:

- coverage;
- precision_up;
- precision_down;
- balanced_precision;
- all-sample accuracy;
- accepted-sample accuracy;
- share_up_predictions;
- share_down_predictions;
- selected t_up;
- selected t_down;
- precision versus coverage frontier.

Secondary diagnostics:

- ROC AUC;
- Brier score;
- log loss;
- calibration curve.

These diagnostics must not override the primary selection metric.

---

## 10. Required Slices and Reports

### 10.1 Validation report

Report on the selected validation window:

- train window;
- validation window;
- coverage;
- precision_up;
- precision_down;
- balanced_precision;
- selected thresholds.

### 10.2 Boundary slices

Report by realized absolute return bucket:

```text
abs(r) < 1bp
1bp <= abs(r) < 5bp
abs(r) >= 5bp
```

These are diagnostics only. Final validation still uses the full sample.

### 10.3 Regime slices

Report by:

- volatility regime;
- trend regime;
- session / time-of-day;
- spread regime, if book data is available;
- volume regime.

---

## 11. Validation Requirement

Use a single chronological train / validation split for the first baseline.

Minimum requirement:

- train only on past data;
- validate only on later data;
- no random shuffle;
- no future leakage;
- threshold search performed only on the validation window.

Example structure:

```text
Train: earlier historical period
Validate: later held-out period
```

Walk-forward validation is a later robustness check, not part of the first implementation.

---

## 12. Minimal Configuration

```yaml
objective:
  label: settlement_direction
  optimize_metric: balanced_precision
  min_coverage: 0.60
  tie_breaker_metric: coverage
  balanced_precision_tie_tolerance: 0.002

sample_weighting:
  enabled: true
  mode: linear_ramp
  min_abs_return: 0.0001
  full_weight_abs_return: 0.0005
  min_weight: 0.20
  max_weight: 1.00

model:
  type: lightgbm_binary
  calibrate_probability: false
  calibration_method: none

threshold_search:
  enabled: true
  t_up_min: 0.50
  t_up_max: 0.60
  t_down_min: 0.40
  t_down_max: 0.50
  step: 0.005
  enforce_min_side_share: false
  min_side_share: 0.20

validation:
  mode: chronological_holdout
  min_train_days: 60
  validation_days: 14
  report_worst_fold: false

features:
  enabled_packs:
    - momentum
    - volatility
    - regime
    - volume
    - derivatives_funding
    - derivatives_basis
    - derivatives_book_ticker
    - flow_pressure
    - book_pressure
    - second_level_microstructure
    - regime_interactions
    - event_window_burst
    - side_specific_transforms

reporting:
  include_precision_coverage_frontier: true
  include_boundary_slices: true
  include_regime_slices: true
  include_calibration_metrics: false
```

---

## 13. Implementation Plan

### Phase 1: Weighted binary baseline

- implement true settlement label;
- implement sample weights;
- train unweighted and weighted binary baselines;
- compare precision / coverage frontier.

Gate:

```text
weighted binary should improve frontier or worst-fold side precision versus unweighted binary
```

### Phase 2: Threshold search

- implement `t_up` / `t_down` search;
- optimize balanced precision under coverage >= 60%;
- report selected thresholds and frontier.

Gate:

```text
selected threshold set must satisfy coverage >= 60%
```

### Phase 3: Feature upgrade

Add:

- flow pressure;
- book pressure;
- second-level microstructure;
- regime interactions;
- event-window burst;
- side-specific transforms.

Gate:

```text
features should improve ranking quality, precision / coverage frontier, or worst-fold metrics
```

### Phase 4: Final research gate

Continue only if:

```text
coverage >= 60%
precision_up > random baseline
precision_down > random baseline
balanced_precision improves over unweighted binary baseline
results are not only driven by one narrow validation window
```

If not, treat it as evidence that the current data / feature boundary may not contain enough stable 5-minute directional signal.

---

## 14. Repository Constraints

- business parameters stay in `config/settings.yaml`;
- labels stay in shared label modules;
- features stay in shared feature modules;
- execution layer stays thin;
- execution must not recompute a separate BTC feature stack;
- threshold policy must be reusable, not notebook-only;
- experiment artifacts must include metrics, thresholds, slices, and frontiers.

---

## 15. Non-Goals

This iteration should not focus on:

- three-class UP / FLAT / DOWN classification;
- active / non-active first-stage modeling;
- median or quantile regression as the primary target;
- two independent UP and DOWN probability models;
- maximizing all-sample accuracy at the expense of selective precision;
- removing boundary samples from final validation;
- complex execution-side features;
- single-window validation.

---

## 16. Final Baseline Summary

```text
1. Use true settlement direction as label.
2. Train one weighted binary model to predict p_up.
3. Use return magnitude only for sample weighting.
4. Use raw p_up score for first threshold search.
5. Set p_down = 1 - p_up.
6. Learn asymmetric thresholds t_up and t_down.
7. Predict UP if p_up >= t_up.
8. Predict DOWN if p_up <= t_down.
9. Abstain otherwise.
10. Optimize balanced precision subject to coverage >= 60%.
11. Prefer higher coverage when precision is materially tied.
12. Add second-level microstructure features where available.
13. Validate on a chronological holdout window and report precision, coverage, frontier, boundary slices, and regime slices.
14. Add calibration and walk-forward validation later only if the first baseline shows usable signal.
```

