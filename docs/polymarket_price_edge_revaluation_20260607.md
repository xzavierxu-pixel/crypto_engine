# Polymarket Price-Edge Revaluation - 2026-06-07

## Scope

This report re-evaluates prior validation experiments and mandatory replay windows with a price-aware trading utility metric.

No model, label, feature, or execution semantics were changed. Polymarket prices are used only at evaluation time.

Parameter note:

```text
This historical revaluation used min_ev_threshold = 0.05.
The current default in scripts/analysis/evaluate_polymarket_price_edge.py is 0.10.
```

Artifacts:

```text
price_store:
  artifacts/data_v2/polymarket_prices/btc_updown_5m_first_price.parquet

evaluation_report:
  artifacts/data_v2/reports/price_edge/price_edge_evaluation_20260607.json

row_outputs:
  artifacts/data_v2/reports/price_edge/rows/
```

Scripts:

```text
scripts/analysis/build_polymarket_decision_price_store.py
scripts/analysis/evaluate_polymarket_price_edge.py
```

## Metric Definition

For each opportunity:

```text
q_up = model p_up
p_up = Polymarket YES token price
p_down = 1 - p_up

edge_up = q_up - p_up
edge_down = (1 - q_up) - p_down = p_up - q_up

side = UP if edge_up >= edge_down else DOWN
accept = max(edge_up, edge_down) > min_ev_threshold

if side == UP:
  profit = resolved_up - p_up

if side == DOWN:
  profit = resolved_down - p_down

trading_utility = sum(profit for accepted trades) / total_evaluable_opportunities
```

The fixed threshold used here is:

```text
min_ev_threshold: 0.05
```

Because `p_down = 1 - p_up`, the acceptance rule is equivalent to:

```text
abs(model_p_up - polymarket_yes_price) > 0.05
```

## Price Rule

For each `polymarket_slug`, the script fetches the YES token id from Gamma:

```text
https://gamma-api.polymarket.com/markets/{market_id}
https://gamma-api.polymarket.com/markets/slug/{slug}
```

Then it fetches CLOB price history:

```text
https://clob.polymarket.com/prices-history?market=<YES_TOKEN_ID>&startTs=<market_t0>&endTs=<market_t0+300>&fidelity=1
```

The evaluation price is the earliest returned point in that 5-minute market window.

Coverage of the price store:

```text
requested slugs: 7677
ok prices:       7651
missing_history: 26
```

The 26 missing prices are excluded from the denominator and reported as `missing_price_samples`.

## Validation Results

| run_id | trading_utility | realized_profit_sum | coverage | accepted_accuracy | accepted_count | avg_entry_price | original_selection_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260520_polymarket_resolved_extended_history_baseline | 0.130210 | 969.020 | 0.8515 | 0.6557 | 6337 | 0.5028 | 0.574851 |
| 20260521_baseline_coverage_090 | 0.130210 | 969.020 | 0.8515 | 0.6557 | 6337 | 0.5028 | 0.514459 |
| 20260521_regime_reversal_second_agg_features | 0.136954 | 1019.215 | 0.8538 | 0.6638 | 6354 | 0.5034 | 0.538264 |
| 20260605_regime_reversal_reversal_weight_boost | 0.134976 | 1004.495 | 0.8426 | 0.6619 | 6271 | 0.5018 | 0.535758 |
| 20260605_regime_reversal_catboost_regime_fm_reversal | 0.136549 | 1016.195 | 0.8636 | 0.6617 | 6427 | 0.5036 | 0.531090 |
| 20260605_regime_reversal_rank_blend | 0.136711 | 1017.405 | 0.8308 | 0.6667 | 6183 | 0.5021 | 0.538186 |

Validation ranking by `trading_utility`:

```text
1. 20260521_regime_reversal_second_agg_features:       0.136954
2. 20260605_regime_reversal_rank_blend:                0.136711
3. 20260605_regime_reversal_catboost_regime_fm_reversal: 0.136549
4. 20260605_regime_reversal_reversal_weight_boost:     0.134976
5. 20260520 current baseline / 20260521 coverage090:   0.130210
```

Validation interpretation:

```text
best price-edge validation improvement vs current baseline:
  trading_utility: +0.006745
  realized_profit_sum: +50.195
  accepted_accuracy: +0.008161
  coverage: +0.002284
```

The regime/reversal second-level feature artifact remains the best validation result under this fixed EV rule.

## Replay Results

### 2026-05-15/16 Profitable Window

| replay combo | trading_utility | realized_profit_sum | coverage | accepted_accuracy | accepted_count |
|---|---:|---:|---:|---:|---:|
| old_model_old_thresholds | 0.215784 | 22.010 | 0.7647 | 0.7692 | 78 |
| current_model_current_thresholds | 0.239167 | 24.395 | 0.7549 | 0.8052 | 77 |
| 20260521_regime_reversal_second_agg_features | 0.201275 | 20.530 | 0.7745 | 0.7468 | 79 |
| 20260605_reversal_weight_boost | 0.228235 | 23.280 | 0.7745 | 0.7848 | 79 |
| 20260605_catboost_regime_fm_reversal | 0.193284 | 19.715 | 0.8039 | 0.7317 | 82 |
| 20260605_rank_blend | 0.215735 | 22.005 | 0.7451 | 0.7763 | 76 |

Best replay result on this window:

```text
current accepted baseline model under EV rule:
  trading_utility: 0.239167
  realized_profit_sum: 24.395
```

The old/current threshold labels are not meaningful for this EV replay because the EV rule ignores `t_up/t_down` and uses only `model_p_up - market_price`.

### 2026-05-20/21 Loss Window

| replay combo | trading_utility | realized_profit_sum | coverage | accepted_accuracy | accepted_count |
|---|---:|---:|---:|---:|---:|
| old_model_old_thresholds | 0.091168 | 9.755 | 0.8598 | 0.6087 | 92 |
| current_model_current_thresholds | 0.094720 | 10.135 | 0.8692 | 0.6129 | 93 |
| 20260521_regime_reversal_second_agg_features | 0.086963 | 9.305 | 0.8598 | 0.6087 | 92 |
| 20260605_reversal_weight_boost | 0.088505 | 9.470 | 0.8598 | 0.6087 | 92 |
| 20260605_catboost_regime_fm_reversal | 0.086682 | 9.275 | 0.8598 | 0.6087 | 92 |
| 20260605_rank_blend | 0.103785 | 11.105 | 0.8505 | 0.6264 | 91 |

Best replay result on this window:

```text
20260605_regime_reversal_rank_blend:
  trading_utility: 0.103785
  realized_profit_sum: 11.105
```

## Interpretation

The price-edge objective changes the acceptance surface materially:

```text
old selective thresholds:
  trade when model confidence crosses t_up/t_down

price-edge policy:
  trade when model probability differs from Polymarket's first market price by more than 0.05
```

This makes the objective closer to the actual trade question:

```text
Is the model's probability sufficiently different from market price?
```

On validation, all tested artifacts are positive under this simplified price model, and the best result is the 20260521 regime/reversal second-level feature artifact.

On the mandatory replay windows, the result is mixed:

```text
2026-05-15/16:
  current accepted baseline is best among the tested replay combos.

2026-05-20/21:
  20260605 rank blend is best among the tested replay combos.
```

So the price-edge metric does not produce a single artifact that dominates both replay windows.

## Important Caveat

This is not live matched PnL.

The replay uses the earliest CLOB `prices-history` point in the market window, with `p_down = 1 - p_up`. It does not model:

```text
actual bid/ask spread
order queue position
maker fill probability
taker crossing
partial fills
second-leg optionality
live avg matched price
```

This explains why the 2026-05-20/21 replay is positive here even though the live matched-trade report was negative. The live report paid an average matched price around `0.6299`; this simplified price-edge replay has average entry prices around `0.50`.

The result is therefore best interpreted as:

```text
price-aware signal utility using market first price
```

not as deployable execution PnL.

## Recommendation

Use `trading_utility` as the next primary offline selection metric, but keep it explicitly scoped to the price source:

```text
primary_metric: trading_utility
price_source: clob_prices_history_first_point_in_market_window
min_ev_threshold: 0.05
p_down_rule: 1 - p_up
```

Keep `selection_score`, accepted accuracy, coverage, AUC, Brier, and logloss as diagnostics.

Before using this for deployment, add one more evaluation layer using realistic executable prices:

```text
entry_price = configured order price or observed best bid/ask at decision time
```

That layer is required to reconcile the positive simplified replay with the negative live matched-trade outcome.
