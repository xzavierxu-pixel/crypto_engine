# BTC 5-Minute Polymarket × Freqtrade/FreqAI V1 实施方案（工程化增强版）

## 1. 目标

本方案在原有“训练与 Polymarket 解耦”的基础上，进一步强化以下工程目标：

- 在线与离线逻辑最大化复用
- 所有核心参数统一配置
- 代码高可读性、可维护性
- 高内聚、低耦合
- 避免“改了训练没改线上”或“改了执行没改特征”的问题
- 保持 Freqtrade / FreqAI 的原生最佳实践使用方式

---

## 2. 总体设计原则

### 2.1 单一事实来源（Single Source of Truth）
所有以下内容都只能有一个定义来源：

- 交易所 / 交易对
- 时间框架
- 样本网格（5 分钟整点）
- 标签定义
- 特征窗口长度
- 模型参数
- 交易阈值
- 风控限制
- 路径与 artifact 位置

这些都统一放在一个配置文件中。

---

### 2.2 训练、推理、执行三层分离
系统分为三层：

#### A. Data + Signal Layer（Freqtrade / FreqAI）
负责：
- BTC/USD 数据拉取
- 特征构造
- 标签构造
- 模型训练
- 在线推理

#### B. Execution Layer（Polymarket Adapter）
负责：
- 获取 Polymarket 市场
- 获取 CLOB 价格
- 订单提交与状态跟踪

#### C. Shared Core Layer
负责：
- 公共配置
- 时间网格
- 特征函数
- 标签函数
- 数据 schema
- 日志结构
- 通用校验逻辑

> 原则：训练和线上都依赖 Shared Core，而不是各自复制逻辑。

---

## 3. 推荐目录结构

```text
project/
├─ config/
│  ├─ settings.yaml
│  ├─ freqtrade_config.json
│  └─ logging.yaml
│
├─ src/
│  ├─ core/
│  │  ├─ config.py
│  │  ├─ constants.py
│  │  ├─ timegrid.py
│  │  ├─ schemas.py
│  │  ├─ logging_utils.py
│  │  └─ validation.py
│  │
│  ├─ features/
│  │  ├─ base_features.py
│  │  ├─ momentum.py
│  │  ├─ volatility.py
│  │  ├─ path_structure.py
│  │  ├─ regime.py
│  │  └─ builder.py
│  │
│  ├─ labels/
│  │  ├─ grid.py
│  │  └─ target_builder.py
│  │
│  ├─ data/
│  │  ├─ loaders.py
│  │  ├─ preprocess.py
│  │  └─ dataset_builder.py
│  │
│  ├─ model/
│  │  ├─ train.py
│  │  ├─ infer.py
│  │  ├─ calibration.py
│  │  └─ registry.py
│  │
│  ├─ execution/
│  │  ├─ polymarket_client.py
│  │  ├─ market_mapper.py
│  │  ├─ order_router.py
│  │  ├─ position_sizer.py
│  │  └─ decision_engine.py
│  │
│  ├─ services/
│  │  ├─ feature_service.py
│  │  ├─ signal_service.py
│  │  └─ audit_service.py
│  │
│  └─ strategies/
│     └─ BTCGridFreqAIStrategy.py
│
├─ notebooks/
├─ artifacts/
├─ tests/
│  ├─ test_timegrid.py
│  ├─ test_labels.py
│  ├─ test_features.py
│  ├─ test_dataset_builder.py
│  └─ test_decision_engine.py
│
└─ scripts/
   ├─ build_dataset.py
   ├─ train_model.py
   ├─ run_shadow.py
   └─ run_live_signal.py
```

---

## 4. 参数统一管理

### 4.1 原则
所有参数放到一个主配置文件中，例如：

`config/settings.yaml`

Freqtrade 原生配置仍保留，但它只负责 bot/exchange/freqai 所需配置；业务参数统一从 `settings.yaml` 注入，避免分散在：
- strategy 文件
- notebook
- execution 脚本
- 临时常量

### 4.2 建议配置项

```yaml
project:
  name: btc-polymarket-v1
  timezone: UTC

market:
  exchange: binance
  pair: BTC/USD
  timeframe: 1m
  grid_minutes: 5

dataset:
  train_start: "2024-01-01"
  train_end: "2025-12-31"
  drop_incomplete_candles: true
  strict_grid_only: true

labels:
  horizon_minutes: 5
  use_grid_open_close: true
  positive_rule: "future_close_gt_current_open"

features:
  momentum_windows: [1, 3, 5, 10, 15]
  vol_windows: [3, 5, 10, 30]
  slope_windows: [3, 5]
  range_windows: [3, 5, 10]
  use_vwap_distance: true
  use_regime_features: true
  use_time_features: true

model:
  type: lightgbm
  random_state: 42
  class_weight: balanced
  calibration: isotonic

execution:
  yes_price_min: 0.40
  yes_price_max: 0.60
  edge_threshold: 0.03
  single_position_cap: 0.02
  max_total_exposure: 0.20

paths:
  artifacts_dir: "./artifacts"
  model_dir: "./artifacts/models"
  logs_dir: "./artifacts/logs"
```

---

## 5. 单一通用逻辑如何抽取

这是本方案最关键的工程要求。

### 5.1 时间网格逻辑只写一次
新建：

- `src/core/timegrid.py`

只在这里定义：
- 什么叫 5 分钟整点
- 如何判断某条 1m candle 是否属于合法样本时点
- 如何从 1m 数据抽取 5m grid 样本

训练、回测、线上推理全部调用同一个函数。

---

### 5.2 标签逻辑只写一次
新建：

- `src/labels/target_builder.py`

统一定义：

```python
y = 1{close[t0+5m] > open[t0]}
```

不要在：
- notebook 再写一遍
- strategy 再写一遍
- live execution 脚本再写一遍

标签逻辑必须只有一个实现。

---

### 5.3 特征逻辑只写一次
新建：

- `src/features/builder.py`

所有特征都由统一入口生成，例如：

```python
build_feature_frame(df, cfg)
```

训练时用它。
线上推理时也用它。
影子运行时也用它。

不要出现：
- 训练脚本一套 rolling 逻辑
- strategy 里面又重写一套指标
- execution service 里再拼一套轻量版特征

---

### 5.4 信号决策逻辑只写一次
新建：

- `src/execution/decision_engine.py`

统一封装：

```python
should_place_order(p_yes, q_yes, cfg, risk_state)
```

这样：
- 回测
- shadow run
- live
都共用同一个入场判断器。

---

## 6. Freqtrade / FreqAI 的 best practice 结合方式

### 6.1 让 Freqtrade 做它擅长的事
Freqtrade / FreqAI 最适合承担：
- 交易所数据接入
- 历史数据下载
- 统一策略结构
- 特征工程
- 模型训练/回测
- 在线推理

第一版不要强行让 Freqtrade 承担 Polymarket 执行。

---

### 6.2 Strategy 里只保留薄层
`BTCGridFreqAIStrategy.py` 不应塞满业务逻辑。

它只应该负责：
- 接 Freqtrade 生命周期
- 调用 shared feature builder
- 调用 shared target builder
- 把 FreqAI 需要的列正确挂上去

不要把复杂业务逻辑全写在 Strategy 类里。

---

### 6.3 FreqAI 只做模型，不做业务编排
不要把：
- CLOB 路由
- Polymarket market discovery
- position sizing
- order state machine

写进 FreqAI strategy 逻辑里。

这些应放到独立 execution layer。

---

### 6.4 使用统一 config 注入 strategy
Strategy 中只读取配置对象，不硬编码窗口长度、阈值、pair 等参数。

坏例子：
```python
ret_5 = df["close"].pct_change(5)
```

好例子：
```python
for w in cfg.features.momentum_windows:
    df[f"ret_{w}"] = df["close"].pct_change(w)
```

---

### 6.5 用 services 层桥接离线与在线
建议建立：
- `FeatureService`
- `SignalService`

这样 notebook、训练脚本、shadow run、线上推理都不直接碰底层细节。

例如：
```python
signal = signal_service.predict_from_latest_frame(df_latest)
```

---

## 7. 标签与数据构造 best practice

### 7.1 样本必须只取固定 5 分钟 grid
只在：
- 00
- 05
- 10
- ...

这些分钟建样本。

不要用每分钟滚动训练 future 5m，否则训练目标和真实交易时点错位。

---

### 7.2 标签必须以当前窗口 open 为起点
统一定义：

```python
y = 1 if close[t0+5m] > open[t0] else 0
```

不能混用：
- close[t0]
- next open
- arbitrary mid price

---

### 7.3 特征严格只用 t0 前可见信息
如果你线上是在窗口开始时判断，那么训练也只能使用窗口开始前可见的数据。

这是避免 leakage 的底线。

---

## 8. 特征工程 best practice（兼顾准确率与可维护性）

### 8.1 采用“模块化特征包”
不要把所有特征堆在一个 500 行函数里。

建议分成：
- `momentum.py`
- `volatility.py`
- `path_structure.py`
- `regime.py`

每个模块只负责一类特征。

---

### 8.2 特征命名统一
建议统一命名规范：

- `ret_1`
- `ret_5`
- `rv_5`
- `slope_3`
- `range_10`
- `vwap_dist_5`
- `hour_sin`
- `hour_cos`

命名稳定后：
- 训练集列名稳定
- 模型输入稳定
- SHAP / importance 更易解释

---

### 8.3 第一版优先高价值特征
优先做：
1. 多窗口 momentum
2. realized volatility
3. path/range/slope
4. time-of-day
5. regime features

不要第一版就引入大量外部杂项源，先把主干做稳。

---

### 8.4 所有 rolling 特征统一通过 helper 生成
例如：

```python
rolling_std(df["ret_1"], window=5)
rolling_slope(df["close"], window=3)
```

把所有共性逻辑收进 helper，减少复制与 bug。

---

## 9. 训练与推理 best practice

### 9.1 训练脚本和线上推理都走同一 pipeline
统一入口：
- `dataset_builder.build_training_frame()`
- `feature_service.build_online_frame()`

它们内部应尽量复用相同的 feature/label/timegrid 模块。

---

### 9.2 严格时间切分
不要 random shuffle。
只做：
- chronological split
- walk-forward
- rolling validation

---

### 9.3 模型注册与版本化
新建：
- `src/model/registry.py`

记录：
- model_version
- feature_version
- config_hash
- training_period
- git_commit（如果可用）

保证以后知道线上到底跑的是哪一版。

---

### 9.4 概率校准单独模块化
因为你最终要拿 `p_up` 去和 Polymarket 价格比，概率质量很重要。

建议：
- calibration 独立模块
- 训练后保存 calibrator
- 推理时统一经过 calibrator

---

## 10. 执行层 best practice

### 10.1 Execution 只消费信号，不生成信号
Execution layer 输入应该是：

- `p_up`
- 当前市场价格
- 风控状态

它不应该自己再算一套 BTC 特征或标签。

---

### 10.2 决策函数纯函数化
例如：

```python
decision = evaluate_entry(
    p_up=p_up,
    yes_price=q_yes,
    cfg=cfg,
    exposure_state=state,
)
```

这样最容易测、最容易复盘。

---

### 10.3 Position sizing 独立
不要把仓位逻辑混进 order_router。

拆开：
- `decision_engine.py`
- `position_sizer.py`
- `order_router.py`

---

## 11. 可测试性设计

### 11.1 关键函数必须可单测
必须单测的模块：

- time grid 对齐
- 标签构造
- feature builder
- dataset builder
- decision engine
- position sizing

---

### 11.2 避免不可测写法
避免：
- 到处读全局变量
- 函数内部直接访问磁盘/网络
- strategy 里硬编码阈值
- dataframe 修改副作用过多

---

## 12. 可维护性规范

### 12.1 每个模块只有一个职责
- features 只负责特征
- labels 只负责标签
- execution 只负责执行
- config 只负责配置
- services 负责编排

---

### 12.2 Strategy 文件保持薄
目标：
- 200 行左右，越薄越好
- 不做业务中心
- 不做系统总控

---

### 12.3 避免“神函数”
任何一个函数超过 80–120 行就要考虑拆分。

---

### 12.4 配置对象类型化
建议用 Pydantic / dataclass 封装配置，避免到处写：

```python
cfg["features"]["momentum_windows"]
```

改为：

```python
cfg.features.momentum_windows
```

更可读，也更安全。

---

## 13. 推荐开发顺序

### Phase 1：Shared Core
先写：
- config loader
- timegrid
- label builder
- feature builder

### Phase 2：Dataset Builder
把离线训练集构造跑通。

### Phase 3：FreqAI Strategy 薄封装
让 strategy 只接入 shared modules。

### Phase 4：Model + Calibration
训练 baseline，保存模型与 calibrator。

### Phase 5：Signal Service
做统一线上推理接口。

### Phase 6：Polymarket Execution
接 CLOB，接 decision engine，做 shadow run。

---

## 14. 第一版推荐的工程结论

如果你的实现要求是：

- 在线离线统一
- 参数单点管理
- 可读性强
- 可维护性强
- 高内聚低耦合

那么最优方案不是“把所有逻辑塞进 Freqtrade strategy”，而是：

### 最终推荐
- **Freqtrade/FreqAI：数据、特征、训练、推理**
- **Shared Core：时间、标签、特征、配置、schema**
- **Polymarket Adapter：执行**
- **一个统一 settings 文件管理全部业务参数**

---

## 15. 一句话架构原则

> **让 Freqtrade/FreqAI 成为信号引擎，而不是整个系统的唯一容器。**  
> **把训练与在线共用逻辑沉到 shared core，把执行独立出来。**  
> **所有参数只在一个配置文件里定义一次。**
