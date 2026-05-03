# MRC-LSTM for BTC 5-Minute Direction Prediction — Requirements Document

## 1. Project Objective

This document defines the requirements for implementing an **MRC-LSTM model** for BTC short-horizon direction prediction.

The target use case is:

> Use second-level BTC market data to predict whether the reference BTC price at the end of the next 5-minute window will be higher or lower than the start reference price.

The model is designed for the user's Polymarket-style 5-minute BTC Up/Down prediction workflow, where the primary objective is not full-sample AUC alone, but **high signal accuracy under acceptable coverage**.

---

## 2. Core Prediction Task

### 2.1 Target

Binary classification:

```text
y = 1 if future_reference_price > current_reference_price
y = 0 otherwise
```

Where:

```text
current_reference_price = BTC reference price at market/window start
future_reference_price  = BTC reference price at market/window end
```

For approximation using Binance data:

```text
current_reference_price ≈ nearest available BTC price at t0
future_reference_price  ≈ nearest available BTC price at t0 + 5 minutes
```

### 2.2 Prediction Horizon

Default horizon:

```text
5 minutes
```

### 2.3 Feature Lookback Window

Default sequence input:

```text
past 300 seconds
```

Optional windows for experiments:

```text
120 seconds
300 seconds
600 seconds
900 seconds
```

Recommended initial setting:

```text
seq_len = 300
```

---

## 3. Model Overview

### 3.1 Model Name

```text
MRC-LSTM
```

Full conceptual name:

```text
Multi-scale Residual Convolutional Neural Network + LSTM
```

### 3.2 Model Intuition

The model has two major components:

1. **Multi-scale Residual CNN block**
   - Extracts short-term and medium-term local patterns from second-level data.
   - Uses multiple Conv1D branches with different kernel sizes.
   - Captures different microstructure windows such as 3s, 5s, 10s, 30s, and 60s.

2. **LSTM block**
   - Learns sequential dependency after the CNN has extracted local patterns.
   - Uses the final hidden state for binary direction classification.

### 3.3 Why MRC-LSTM Is Relevant

BTC 5-minute direction prediction depends on short-lived market microstructure patterns, such as:

```text
sudden volume burst
short-term momentum continuation
buy/sell trade imbalance
volatility compression or expansion
temporary order-flow dominance
local path shape
```

A plain LSTM may struggle to extract these local patterns directly from noisy second-level data.

MRC-LSTM improves this by first using CNN branches to detect multi-scale local structures, then using LSTM to model the resulting sequence.

---

## 4. Input Data Requirements

### 4.1 Required Data Granularity

Minimum required granularity:

```text
1-second aggregated data
```

The raw source can be:

```text
Binance aggTrades
Binance trades
Binance bookTicker
Binance futures klines
Binance futures taker buy/sell volume
```

### 4.2 Recommended Raw Data Sources

| Data Source | Purpose |
|---|---|
| aggTrades | Trade flow, trade count, buy/sell pressure |
| bookTicker | Best bid/ask, spread, top-of-book imbalance |
| futures kline | Price, volume, volatility features |
| spot kline | Cross-market price reference |
| funding / open interest | Derivatives regime features |
| taker buy/sell volume | Directional flow pressure |

### 4.3 Input Tensor Shape

The model input should be:

```text
[batch_size, seq_len, feature_dim]
```

Example:

```text
[128, 300, 50]
```

Meaning:

```text
128 samples
300 seconds of history per sample
50 features per second
```

---

## 5. Feature Requirements

### 5.1 Required Feature Groups

The implementation should support the following feature groups.

### A. Price and Return Features

Examples:

```text
price
mid_price
log_price
ret_1s
log_ret_1s
ret_3s
ret_5s
ret_10s
ret_30s
ret_60s
ret_300s
```

### B. Volatility Features

Examples:

```text
rolling_std_5s
rolling_std_10s
rolling_std_30s
rolling_std_60s
realized_vol_30s
realized_vol_60s
realized_vol_300s
abs_ret_1s
max_abs_ret_30s
volatility_zscore_300s
```

### C. Volume and Trade Intensity Features

Examples:

```text
volume_1s
quote_volume_1s
trade_count_1s
avg_trade_size_1s
relative_volume_30s
relative_volume_60s
volume_zscore_300s
trade_intensity_10s
trade_intensity_30s
```

### D. AggTrade Flow Features

Examples:

```text
buy_volume_1s
sell_volume_1s
buy_volume_ratio_1s
sell_volume_ratio_1s
trade_imbalance_1s
trade_imbalance_5s
trade_imbalance_30s
large_trade_count_30s
large_buy_volume_30s
large_sell_volume_30s
```

### E. Order Book / bookTicker Features

Examples:

```text
best_bid
best_ask
bid_size
ask_size
spread
spread_bps
mid_price
microprice
top_order_imbalance
bid_ask_size_ratio
microprice_return_1s
spread_zscore_300s
```

### F. Path Shape Features

Examples:

```text
path_high_60s
path_low_60s
drawdown_60s
runup_60s
range_position_60s
num_price_flips_60s
trend_consistency_60s
v_shape_score_300s
```

### G. Regime Features

Examples:

```text
vol_regime
volume_regime
trend_regime
range_regime
high_vol_flag
low_vol_flag
liquidity_thin_flag
```

---

## 6. Model Architecture Requirements

### 6.1 Multi-scale Residual CNN Block

The CNN block should include multiple Conv1D branches.

Recommended initial kernel sizes:

```text
kernel_sizes = [3, 5, 10, 30, 60]
```

Interpretation:

| Kernel Size | Meaning |
|---:|---|
| 3 | 3-second micro movement |
| 5 | 5-second trade burst |
| 10 | 10-second short momentum |
| 30 | 30-second local structure |
| 60 | 1-minute structure |

Each branch should include:

```text
Conv1D
BatchNorm1D
ReLU
Dropout
```

Outputs from all branches should be concatenated and fused with a 1x1 Conv1D layer.

### 6.2 Residual Connection

The model should include a residual projection from the original input to the CNN hidden dimension.

Purpose:

```text
Preserve raw input information
Improve gradient flow
Reduce degradation risk from deeper CNN blocks
```

### 6.3 LSTM Block

Recommended initial configuration:

```text
lstm_hidden_dim = 64
lstm_layers = 1
bidirectional = False
dropout = 0.2
```

Use unidirectional LSTM by default to avoid any confusion around future-looking behavior.

A bidirectional LSTM can be tested later, but only because the entire input window is historical. It must not include any future information beyond the decision time.

### 6.4 Classification Head

Recommended structure:

```text
LayerNorm
Dropout
Linear
ReLU
Dropout
Linear
```

Output:

```text
single logit
```

The final probability is:

```text
p_up = sigmoid(logit)
```

---

## 7. Training Requirements

### 7.1 Loss Function

Default:

```text
BCEWithLogitsLoss
```

Optional:

```text
weighted BCE loss
focal loss
return-weighted BCE loss
```

### 7.2 Sample Weighting

Because the user cares about direction accuracy and signal quality, the system should support optional sample weights.

Recommended weighting logic:

```text
higher absolute future return → higher sample weight
near-zero future return → lower sample weight
```

Example:

```text
sample_weight = clip(abs(future_return) / rolling_median_abs_return, lower, upper)
```

Purpose:

```text
Do not fully remove noisy samples
But make meaningful movement samples more influential
```

### 7.3 Optimizer

Recommended:

```text
AdamW
```

Initial parameters:

```text
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 128
epochs = 20
early_stopping_patience = 5
gradient_clip_norm = 1.0
```

### 7.4 Regularization

Required regularization options:

```text
dropout
weight decay
early stopping
gradient clipping
small hidden dimensions at first
```

Recommended first-run model size:

```text
cnn_hidden_dim = 64
lstm_hidden_dim = 64
lstm_layers = 1
```

Avoid starting with a large model because the signal is weak and validation overfitting is likely.

---

## 8. Validation Requirements

### 8.1 Time-aware Split

Do not use random split.

Required:

```text
train period < validation period < test period
```

Recommended:

```text
walk-forward validation
```

### 8.2 No Leakage Rules

The feature window for a sample must end at or before decision time.

For each sample:

```text
features use data from [t0 - seq_len, t0]
label uses price from t0 to t0 + 5 minutes
```

No feature may include data after `t0`.

### 8.3 Evaluation Metrics

Full-sample metrics:

```text
accuracy
AUC
logloss
Brier score
```

Trading/signal metrics:

```text
signal_accuracy
signal_coverage
precision_up
precision_down
number_of_signals
expected_pnl
accuracy_by_price_bucket
accuracy_by_volatility_regime
```

The most important metrics for this project are:

```text
signal_accuracy
signal_coverage
precision_up
precision_down
expected_pnl
```

Not AUC alone.

### 8.4 Signal Definition

The implementation should support threshold-based signals.

Example:

```text
if p_up >= upper_threshold:
    signal = "UP"
elif p_up <= lower_threshold:
    signal = "DOWN"
else:
    signal = "NO_TRADE"
```

Initial thresholds:

```text
upper_threshold = 0.55
lower_threshold = 0.45
```

Thresholds should be tuned on validation only.

---

## 9. Integration with LightGBM

### 9.1 Direct Classifier Mode

MRC-LSTM can be trained as a standalone classifier:

```text
second-level sequence → MRC-LSTM → p_up
```

### 9.2 Encoder Mode

Preferred production-style integration:

```text
second-level sequence
→ MRC-LSTM encoder
→ deep embeddings
→ concatenate with existing tabular features
→ LightGBM final classifier
```

Recommended exported features:

```text
mrc_lstm_embedding_1 ... mrc_lstm_embedding_64
mrc_lstm_p_up
mrc_lstm_logit
mrc_lstm_confidence = abs(p_up - 0.5)
```

### 9.3 Why Encoder Mode Is Preferred

The user's existing pipeline already relies on strong tabular features and LightGBM.

MRC-LSTM should initially be used to add sequence representation rather than fully replace the current system.

Benefits:

```text
Less overfitting risk than pure end-to-end deep learning
Easier to compare incremental value
Easier to debug
Can reuse current LightGBM evaluation and signal-selection framework
```

---

## 10. Experiment Plan

### Experiment 1 — Baseline LightGBM

Purpose:

```text
Confirm current baseline using existing tabular features.
```

Metrics:

```text
valid AUC
valid accuracy
signal accuracy
signal coverage
precision_up
precision_down
```

### Experiment 2 — MRC-LSTM Standalone

Purpose:

```text
Test whether MRC-LSTM sequence model can predict direction directly.
```

Input:

```text
past 300 seconds second-level features
```

Output:

```text
p_up
```

### Experiment 3 — MRC-LSTM Encoder + LightGBM

Purpose:

```text
Test whether MRC-LSTM embeddings improve existing LightGBM.
```

Process:

```text
train MRC-LSTM
extract embeddings
join embeddings to sample-level tabular dataset
train LightGBM
compare against baseline
```

### Experiment 4 — Kernel Size Ablation

Test:

```text
[3, 5, 10]
[3, 5, 10, 30]
[3, 5, 10, 30, 60]
[5, 15, 30, 60]
```

Purpose:

```text
Identify useful temporal scales.
```

### Experiment 5 — Lookback Window Ablation

Test:

```text
120s
300s
600s
900s
```

Purpose:

```text
Find best historical window for 5-minute direction.
```

### Experiment 6 — Feature Group Ablation

Test removing/adding:

```text
price only
price + volume
price + volume + trade flow
price + volume + trade flow + bookTicker
all features
```

Purpose:

```text
Identify whether aggTrade and bookTicker actually improve validation signal.
```

---

## 11. Acceptance Criteria

### 11.1 Technical Acceptance

The implementation is acceptable if:

```text
model trains without shape errors
no data leakage
supports train/valid/test split by time
supports GPU if available
exports validation predictions
exports optional embeddings
saves model checkpoint
saves config and metrics
```

### 11.2 Modeling Acceptance

The model is useful only if it improves at least one of the following on validation or test:

```text
signal_accuracy
signal_coverage at same accuracy
precision_up
precision_down
expected_pnl
LightGBM + MRC embeddings vs LightGBM baseline
```

A higher full-sample AUC alone is not sufficient.

### 11.3 Minimum Target Improvement

Initial target:

```text
MRC-LSTM encoder + LightGBM should improve signal accuracy or coverage versus LightGBM baseline.
```

Example acceptable improvement:

```text
same signal accuracy with higher coverage
or
same coverage with higher signal accuracy
or
better precision_up and precision_down balance
```

---

## 12. Recommended Initial Configuration

```yaml
task:
  prediction_horizon_seconds: 300
  sequence_length_seconds: 300
  label_type: binary_direction

model:
  name: MRC_LSTM
  input_dim: auto
  cnn_hidden_dim: 64
  lstm_hidden_dim: 64
  lstm_layers: 1
  kernel_sizes: [3, 5, 10, 30, 60]
  dropout: 0.2
  bidirectional: false

training:
  optimizer: AdamW
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 128
  epochs: 20
  early_stopping_patience: 5
  gradient_clip_norm: 1.0
  loss: BCEWithLogitsLoss

validation:
  split_type: time_based
  metrics:
    - accuracy
    - auc
    - logloss
    - brier
    - signal_accuracy
    - signal_coverage
    - precision_up
    - precision_down
    - expected_pnl

signal:
  upper_threshold: 0.55
  lower_threshold: 0.45
```

---

## 13. Implementation Deliverables

The implementation should produce:

```text
1. model_mrc_lstm.py
2. dataset_sequence.py
3. train_mrc_lstm.py
4. evaluate_signals.py
5. extract_mrc_embeddings.py
6. config_mrc_lstm.yaml
7. validation_predictions.csv
8. model_checkpoint.pt
9. metrics_summary.json
```

### 13.1 Required Output Columns for Predictions

```text
timestamp
market_id or sample_id
y_true
p_up
logit
pred_label
signal
future_return
signal_correct
```

### 13.2 Required Output Columns for Embeddings

```text
timestamp
sample_id
mrc_lstm_embedding_1
mrc_lstm_embedding_2
...
mrc_lstm_embedding_64
mrc_lstm_p_up
mrc_lstm_logit
mrc_lstm_confidence
```

---

## 14. Key Risks

### 14.1 Overfitting

Risk:

```text
Train performance improves but validation/live performance does not.
```

Mitigation:

```text
small model first
dropout
weight decay
early stopping
walk-forward validation
feature group ablation
```

### 14.2 Label Noise

Risk:

```text
Many 5-minute BTC windows are near-zero movement and direction is almost random.
```

Mitigation:

```text
sample weighting
gray-zone analysis
evaluate by future-return magnitude bucket
do not over-optimize full-sample accuracy
```

### 14.3 Data Leakage

Risk:

```text
second-level features accidentally include future information.
```

Mitigation:

```text
strict timestamp alignment
feature window must end at t0
label window starts after t0
unit tests for sample construction
```

### 14.4 Poor Live Transfer

Risk:

```text
Validation result does not survive live trading due to regime shift.
```

Mitigation:

```text
walk-forward validation
regime-specific metrics
rolling retraining
monitor live signal accuracy
```

---

## 15. Recommended Development Order

```text
Step 1: Build clean sequence dataset
Step 2: Train small MRC-LSTM standalone model
Step 3: Validate no leakage and correct tensor shapes
Step 4: Evaluate signal metrics, not only AUC
Step 5: Export embeddings
Step 6: Add embeddings into LightGBM
Step 7: Compare against current baseline
Step 8: Run ablations on kernel size, lookback length, and feature groups
Step 9: Only scale model size if validation improves
```

---

## 16. Final Recommendation

The recommended first implementation should not try to replace the full current LightGBM system.

The best first version is:

```text
MRC-LSTM as sequence encoder
+
LightGBM as final decision model
```

This allows the system to benefit from second-level sequence information while keeping the existing tabular modeling and signal-selection framework.

The standalone MRC-LSTM classifier should still be implemented for comparison, but the encoder mode is the more practical path for the user's BTC 5-minute Polymarket prediction workflow.
