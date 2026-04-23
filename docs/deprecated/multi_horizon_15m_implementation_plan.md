# 15m Horizon Implementation Plan

## 1. Goal

Extend the existing BTC/USDT prediction system so it supports `15m` direction prediction while preserving the current `5m` pipeline.

This must keep the current architecture intact:

- one shared feature pipeline
- one shared label pipeline
- one shared time-grid implementation
- thin strategy adapters
- execution isolated from feature recomputation

The target is not to create a second parallel system for `15m`. The target is to make the existing shared pipeline horizon-aware enough to support both `5m` and `15m`.

## 2. Scope

### In scope

- training support for `15m`
- dataset building support for `15m`
- offline evaluation support for `15m`
- online signal generation support for `15m`
- shared label generation for `15m`
- shared feature generation for `15m`
- model artifact separation by horizon
- tests for `5m` and `15m` parity

### Out of scope for the first implementation

- replacing the current `5m` execution mapper
- converting the Polymarket execution layer into a generic multi-horizon router
- assuming a `15m` Polymarket market exists
- rewriting the strategy into a multi-product trading bot

## 3. Current State

The codebase is already partially horizon-aware:

- [config/settings.yaml](C:\Users\ROG\Desktop\crypto_engine\config\settings.yaml) already has a `horizons` section
- [src/core/config.py](C:\Users\ROG\Desktop\crypto_engine\src\core\config.py) already parses horizon specs
- [src/horizons/base.py](C:\Users\ROG\Desktop\crypto_engine\src\horizons\base.py) defines a generic `HorizonSpec`
- [src/horizons/registry.py](C:\Users\ROG\Desktop\crypto_engine\src\horizons\registry.py) already resolves horizon config dynamically
- [src/labels/grid_direction.py](C:\Users\ROG\Desktop\crypto_engine\src\labels\grid_direction.py) already computes labels from `horizon.minutes`
- [src/features/builder.py](C:\Users\ROG\Desktop\crypto_engine\src\features\builder.py) already builds features using `horizon.grid_minutes`
- [src/data/dataset_builder.py](C:\Users\ROG\Desktop\crypto_engine\src\data\dataset_builder.py) already accepts `horizon_name`
- [src/services/feature_service.py](C:\Users\ROG\Desktop\crypto_engine\src\services\feature_service.py) already accepts `horizon_name`
- [src/services/signal_service.py](C:\Users\ROG\Desktop\crypto_engine\src\services\signal_service.py) already accepts `horizon_name`

The system is not fully multi-horizon yet because:

- config only declares `5m`
- feature profiles are only defined for `5m`
- some strategy and execution paths still hardcode `5m`
- execution mapping is implemented only for BTC 5-minute Polymarket markets

## 4. Core Design Decision

The correct implementation is:

- keep `5m` and `15m` in the same shared pipeline
- make horizon a config-driven runtime parameter
- preserve `5m` execution as-is unless and until a real `15m` market mapping is confirmed

This means:

- `15m` should share the same raw 1m BTC spot input
- `15m` should share the same derivatives raw input pipeline
- `15m` should share the same feature builder
- `15m` should share the same label builder
- `15m` should not introduce duplicated logic in scripts or services

## 5. Label Definition

The `15m` label should follow the same rule family as the current `5m` label:

- `y = 1{close[t0+15m] > open[t0]}`

This should continue to be implemented only through:

- [src/labels/grid_direction.py](C:\Users\ROG\Desktop\crypto_engine\src\labels\grid_direction.py)

No script, strategy, or signal service should manually recreate this label.

## 6. Time Grid Rules

### 5m

- base data frequency: `1m`
- sample rows only at 5-minute aligned timestamps
- prediction target uses the completed close `15` minutes? No. For `5m`, it remains `close[t0+5m]`

### 15m

- base data frequency remains `1m`
- sample rows only at 15-minute aligned timestamps
- prediction target uses `close[t0+15m]`

The horizon-specific grid behavior must continue to live only in:

- [src/core/timegrid.py](C:\Users\ROG\Desktop\crypto_engine\src\core\timegrid.py)
- [src/features/builder.py](C:\Users\ROG\Desktop\crypto_engine\src\features\builder.py)
- [src/labels/grid_direction.py](C:\Users\ROG\Desktop\crypto_engine\src\labels\grid_direction.py)

## 7. Configuration Changes

### 7.1 Horizons

Extend [config/settings.yaml](C:\Users\ROG\Desktop\crypto_engine\config\settings.yaml) to include both horizons:

```yaml
horizons:
  active: ["5m", "15m"]
  specs:
    "5m":
      minutes: 5
      grid_minutes: 5
      label_builder: grid_direction
      feature_profile: core_5m
      signal_policy: default_edge_policy
      sizing_plugin: fixed_fraction
    "15m":
      minutes: 15
      grid_minutes: 15
      label_builder: grid_direction
      feature_profile: core_15m
      signal_policy: default_edge_policy_15m
      sizing_plugin: fixed_fraction
```

### 7.2 Feature Profiles

Add a separate `core_15m` profile.

This should not reuse the `core_5m` profile name. It may initially reuse most parameter values, but it must be stored separately so later tuning can diverge cleanly.

### 7.3 Signal Policies

Add a separate `default_edge_policy_15m` in the signal config, even if initial thresholds match `5m`.

This avoids coupling two horizons to one execution threshold policy forever.

## 8. Shared Pipeline Changes

### 8.1 No duplicate feature logic

Do not create:

- `build_15m_feature_frame`
- `build_15m_labels`
- `signal_service_15m`

Instead, continue to use the existing parameterized interfaces:

- [src/features/builder.py](C:\Users\ROG\Desktop\crypto_engine\src\features\builder.py)
- [src/data/dataset_builder.py](C:\Users\ROG\Desktop\crypto_engine\src\data\dataset_builder.py)
- [src/services/feature_service.py](C:\Users\ROG\Desktop\crypto_engine\src\services\feature_service.py)
- [src/services/signal_service.py](C:\Users\ROG\Desktop\crypto_engine\src\services\signal_service.py)

### 8.2 Model artifacts must become horizon-aware

Artifacts must be separated by horizon so `5m` and `15m` models cannot be confused at inference time.

Recommended layout:

```text
artifacts/models/
  5m/
    lightgbm/
    catboost/
  15m/
    lightgbm/
    catboost/
```

At minimum, training reports and experiment reports must always record:

- `horizon`
- `feature_profile`
- `feature_columns`
- `config_hash`

## 9. Script-Level Changes

The following scripts should remain single implementations, but each must cleanly accept `--horizon 15m`:

- [scripts/build_dataset.py](C:\Users\ROG\Desktop\crypto_engine\scripts\build_dataset.py)
- [scripts/train_model.py](C:\Users\ROG\Desktop\crypto_engine\scripts\train_model.py)
- [scripts/run_model_experiments.py](C:\Users\ROG\Desktop\crypto_engine\scripts\run_model_experiments.py)
- [scripts/run_shadow.py](C:\Users\ROG\Desktop\crypto_engine\scripts\run_shadow.py)
- [scripts/run_live_signal.py](C:\Users\ROG\Desktop\crypto_engine\scripts\run_live_signal.py)

Implementation rules:

- keep `--horizon` explicit
- do not add separate `*_15m.py` scripts
- ensure reports and output directories encode horizon

## 10. Strategy Changes

The strategy should remain thin.

The current adapter at [BTCGridFreqAIStrategy.py](C:\Users\ROG\Desktop\crypto_engine\src\strategies\BTCGridFreqAIStrategy.py) already carries `horizon_name`, but it still contains `5m` assumptions in user-facing tags and naming.

Required changes:

- make entry tags horizon-aware
- keep `custom_exit()` using `self.horizon.minutes`
- avoid introducing any horizon-specific feature logic into the strategy

The strategy should continue to act as an adapter only.

## 11. Execution Layer Boundary

Execution must be treated separately from shared prediction support.

### First version

Support `15m` in:

- dataset building
- training
- experiments
- online signal generation

Do not automatically support `15m` in:

- Polymarket market mapping
- order routing
- live execution

### Why

Current execution mapping is explicitly `5m`-specific:

- [src/execution/mappers/btc_5m_polymarket.py](C:\Users\ROG\Desktop\crypto_engine\src\execution\mappers\btc_5m_polymarket.py)

That file should remain correct for `5m`. It should not be turned into a generic multi-horizon file full of conditional branches.

If `15m` market execution is needed later, add a separate mapper such as:

- `src/execution/mappers/btc_15m_polymarket.py`

Only do that after confirming the target market actually exists and is stable.

## 12. Train/Live Parity Rules

The most important implementation rule is:

`15m` must be another configuration instance of the same pipeline, not a second implementation.

That means:

- training and live inference both start from the same 1m raw data
- training and live inference both call the same feature builder
- training and live inference both use the same derivatives alignment logic
- training and live inference both select rows based on the same 15-minute grid rule

Forbidden:

- hand-built `15m` labels in scripts
- script-side filtering for 15-minute rows that does not go through shared time-grid logic
- manual 15-minute aggregation logic outside shared modules

## 13. Implementation Steps

### Phase 1: Config and Horizon Registration

- add `15m` to `horizons.specs`
- add `core_15m` feature profile
- add `default_edge_policy_15m`

### Phase 2: Shared Pipeline Validation

- verify `grid_direction` works correctly for `15m`
- verify feature builder selects 15-minute grid rows correctly
- verify dataset builder produces a valid 15m training frame
- verify signal service can infer the latest 15m row

### Phase 3: Script Integration

- ensure all core scripts accept and correctly record `--horizon 15m`
- separate model outputs by horizon
- ensure reports include horizon metadata

### Phase 4: Thin Strategy Cleanup

- remove remaining `5m` string assumptions from strategy tags and metadata
- keep the adapter thin

### Phase 5: Testing and Parity

- add horizon-specific unit tests
- add train/live parity tests for `15m`
- add regression tests proving `5m` behavior remains unchanged

### Phase 6: Optional Execution Expansion

Only after the above is stable:

- evaluate whether 15-minute execution mapping is actually required
- if yes, implement a separate `15m` execution mapper

## 14. File-Level Implementation Plan

### Config and core

- [config/settings.yaml](C:\Users\ROG\Desktop\crypto_engine\config\settings.yaml)
- [src/core/config.py](C:\Users\ROG\Desktop\crypto_engine\src\core\config.py)
- [src/horizons/registry.py](C:\Users\ROG\Desktop\crypto_engine\src\horizons\registry.py)

### Shared pipeline

- [src/labels/grid_direction.py](C:\Users\ROG\Desktop\crypto_engine\src\labels\grid_direction.py)
- [src/features/builder.py](C:\Users\ROG\Desktop\crypto_engine\src\features\builder.py)
- [src/data/dataset_builder.py](C:\Users\ROG\Desktop\crypto_engine\src\data\dataset_builder.py)
- [src/services/feature_service.py](C:\Users\ROG\Desktop\crypto_engine\src\services\feature_service.py)
- [src/services/signal_service.py](C:\Users\ROG\Desktop\crypto_engine\src\services\signal_service.py)

### Strategy and execution boundary

- [src/strategies/BTCGridFreqAIStrategy.py](C:\Users\ROG\Desktop\crypto_engine\src\strategies\BTCGridFreqAIStrategy.py)
- [src/execution/mappers/btc_5m_polymarket.py](C:\Users\ROG\Desktop\crypto_engine\src\execution\mappers\btc_5m_polymarket.py)

### Scripts

- [scripts/build_dataset.py](C:\Users\ROG\Desktop\crypto_engine\scripts\build_dataset.py)
- [scripts/train_model.py](C:\Users\ROG\Desktop\crypto_engine\scripts\train_model.py)
- [scripts/run_model_experiments.py](C:\Users\ROG\Desktop\crypto_engine\scripts\run_model_experiments.py)
- [scripts/run_shadow.py](C:\Users\ROG\Desktop\crypto_engine\scripts\run_shadow.py)
- [scripts/run_live_signal.py](C:\Users\ROG\Desktop\crypto_engine\scripts\run_live_signal.py)

## 15. Testing Plan

### Unit tests

- confirm `15m` horizon config parses correctly
- confirm 15-minute grid selection works
- confirm the label offset is correct for `15m`

### Integration tests

- build a 15m training frame from 1m raw data
- train a 15m model
- generate a 15m live signal from the latest raw frame

### Parity tests

- given the same raw frame and derivatives raw inputs, compare:
  - the final 15m row from the training path
  - the final 15m row from the live signal path
- enforce exact column parity and stable feature ordering

### Regression tests

- prove existing `5m` outputs and tests still pass unchanged

## 16. Acceptance Criteria

The `15m` implementation is complete when:

- the repo supports both `5m` and `15m` from the same shared pipeline
- no duplicate label or feature logic is introduced
- model artifacts are horizon-aware
- `5m` remains backward compatible
- `15m` train/live parity passes
- execution remains isolated unless a dedicated `15m` mapper is intentionally added

## 17. Recommended First Delivery

The first delivery should include:

1. `15m` config support
2. `15m` training and dataset support
3. `15m` online signal support
4. `15m` parity tests
5. no execution-layer expansion yet

This is the smallest correct implementation that preserves the current architecture and avoids turning the system into two divergent codepaths.
