# BTC 5-Minute Polymarket × Freqtrade/FreqAI V1 补充文档
## 插件化扩展架构与实现准则

## 1. 文档目的

本补充文档服务于以下目标：

- 作为主实施文档的工程补充说明
- 强化“插件化思维”的实现边界
- 为后续从 5 分钟市场扩展到 15 分钟、30 分钟、1 小时甚至更多市场提供统一架构
- 让 agent 在实现时明确哪些部分必须抽象、哪些部分必须保持稳定
- 降低后续重构成本，避免“为了扩展周期而推翻第一版”

本文件应与以下文档一起使用：

1. 主实施方案文档
2. Freqtrade / FreqAI best practice 工程架构文档
3. 本补充文档（插件化扩展架构）

---

## 2. 核心原则

### 2.1 先稳定主干，再扩展插件
第一版只做：
- 单一资产：BTC/USD
- 单一 horizon：5 分钟
- 单一执行市场：Polymarket BTC up/down 5m

但从架构设计上，必须提前保证以下扩展不会破坏主干：

- 5m → 15m / 30m / 1h
- BTC → ETH 或其他标的
- 单模型 → 多模型并行
- 单执行市场 → 多执行 venue
- 单阈值 → 分 horizon / 分 regime 阈值
- 单策略 → 多策略组合

---

### 2.2 插件化不是为了“好看”，而是为了减少未来改动面
本项目的插件化目标不是形式上的模块拆分，而是满足：

- 新增一个 horizon，不需要复制整套代码
- 修改一个特征，不需要同时改训练、推理、执行三处
- 更换一个执行市场，不影响模型训练部分
- 增加一个新模型，不影响 signal routing 逻辑
- 改阈值和风控，不影响特征与标签逻辑

---

### 2.3 主干与插件分层
架构上必须区分：

#### 主干（Core）
长期稳定、必须单一来源、各模块共享：
- 配置对象
- 时间网格定义
- 特征构建入口
- 标签构建入口
- 样本 schema
- 模型 registry
- 日志 schema
- 信号 schema
- 风控状态 schema

#### 插件（Plugins / Extensions）
允许未来替换或增加：
- horizon 插件
- 特征包插件
- 模型插件
- 执行适配器插件
- 阈值策略插件
- 仓位 sizing 插件
- 市场过滤插件
- 校准插件

---

## 3. 建议的插件化拆分

## 3.1 Horizon Plugin

### 作用
封装不同 horizon 的定义，例如：
- 5m
- 15m
- 30m
- 1h

### 统一职责
每个 horizon 插件负责定义：
- 样本网格频率
- 预测窗口长度
- 标签定义
- 推荐特征窗口范围
- 默认阈值配置
- 默认风险参数（可选）

### 统一接口建议
```python
class HorizonSpec:
    name: str
    minutes: int
    grid_minutes: int
    label_builder: Callable
    feature_profile: dict
```

### 5m 例子
- `name = "5m"`
- `minutes = 5`
- `grid_minutes = 5`
- 标签：`y = 1{close[t0+5m] > open[t0]}`

### 为什么必须单独抽象
因为未来扩展时：
- 15m 的样本网格不同
- 特征窗口优先级不同
- 交易阈值和概率分布也可能不同

如果 horizon 不抽象，后续每加一个周期都会复制一套代码。

---

## 3.2 Feature Package Plugin

### 作用
把特征按主题做成可组合的插件包。

### 建议插件包
- `MomentumFeaturePack`
- `VolatilityFeaturePack`
- `PathStructureFeaturePack`
- `RegimeFeaturePack`
- `TimeFeaturePack`
- `VolumeFeaturePack`（可选）
- `CrossTimeframeFeaturePack`（后续）

### 统一接口建议
```python
class FeaturePack:
    name: str
    def transform(self, df, cfg) -> pd.DataFrame:
        ...
```

### 为什么这样设计
这样可以做到：
- 第一版先启用最稳的 4 个 pack
- 后续实验时按 pack 开关
- 不同 horizon 用不同 feature profile
- 做 ablation 更容易

### 实现要求
- 每个 feature pack 只负责自己类别的列
- 不允许互相隐式依赖未声明字段
- 所有新增列命名规范统一

---

## 3.3 Label Builder Plugin

### 作用
统一封装标签定义逻辑。

### 第一版
- `GridDirectionLabelBuilder`

定义：
```python
y = 1{close[t0+h] > open[t0]}
```

### 未来扩展可能
- `BufferedDirectionLabelBuilder`
  - `y = 1{future_return > threshold}`
- `TernaryLabelBuilder`
  - up / flat / down
- `MetaLabelBuilder`
  - 先由主模型给方向，再做交易过滤标签
- `ContractMappedLabelBuilder`
  - 第二阶段如需和 Polymarket 更强耦合时再加

### 为什么必须插件化
标签是最敏感的核心逻辑之一。  
如果标签逻辑散落在多个文件里，未来切换标签定义时极易出错。

---

## 3.4 Model Plugin

### 作用
统一封装模型训练与推理后端。

### 第一版建议支持
- `LightGBMClassifierPlugin`
- `CatBoostClassifierPlugin`

### 未来可加
- XGBoost
- Logistic Regression baseline
- MLP
- LSTM/Transformer wrapper
- Ensemble / stacking

### 统一接口建议
```python
class ModelPlugin:
    name: str
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        ...
    def predict_proba(self, X):
        ...
    def save(self, path):
        ...
    def load(self, path):
        ...
```

### 为什么这样设计
这样 execution / signal service 永远只消费统一的 `predict_proba()` 输出，而不关心底层模型细节。

---

## 3.5 Calibration Plugin

### 作用
对模型输出概率进行统一校准。

### 第一版建议
- `NoCalibration`
- `IsotonicCalibration`
- `PlattScalingCalibration`

### 原则
校准必须独立于主模型插件，不能写死在模型类内部。

原因：
- 便于对比
- 便于替换
- 便于单测
- 便于保存和加载

---

## 3.6 Signal Policy Plugin

### 作用
定义如何从 `model output` 变成 `可交易信号`。

### 第一版
建议做成纯规则型插件：

输入：
- `p_up`
- 当前 CLOB yes 价格
- 风控状态
- horizon 配置
- execution 配置

输出：
- `should_trade`
- `side`
- `reason`
- `edge`
- `target_size`

### 未来扩展
- horizon-specific signal policy
- regime-aware threshold policy
- multi-model fusion policy
- portfolio-aware signal policy

### 为什么必须插件化
因为“预测”和“交易决策”不是一回事。  
模型插件负责预测；signal policy 插件负责决定是否出手。

---

## 3.7 Position Sizing Plugin

### 作用
定义仓位分配方法。

### 第一版
- `FixedFractionSizer`
  - 单合约固定上限，比如 0.02

### 未来扩展
- volatility-adjusted sizing
- confidence-adjusted sizing
- portfolio-cap-aware sizing
- drawdown-aware sizing

### 原则
Sizer 不直接访问模型和特征；只吃：
- signal
- current exposure
- risk config

---

## 3.8 Execution Adapter Plugin

### 作用
对不同执行市场做统一抽象。

### 第一版
- `PolymarketExecutionAdapter`

### 未来扩展
- 其他 prediction market
- Kalshi adapter
- paper execution adapter
- simulation adapter

### 统一接口建议
```python
class ExecutionAdapter:
    name: str
    def list_active_markets(self):
        ...
    def get_orderbook(self, market_id):
        ...
    def place_limit_order(self, market_id, side, price, size):
        ...
    def cancel_order(self, order_id):
        ...
```

### 为什么必须单独抽象
因为执行系统最可能随平台改变。  
把执行层抽象掉，才能保证训练和 signal 主干长期稳定。

---

## 3.9 Market Mapper Plugin

### 作用
把模型预测对象映射为实际可交易合约。

### 第一版
- `BTC5mPolymarketMapper`

它负责：
- 找到当前时间对应的 5m 合约
- 校验该合约起止时刻
- 提取必要 market metadata
- 生成标准化 market object

### 未来扩展
- 15m 合约映射器
- 多资产映射器
- 多市场映射器

### 为什么不放进 execution adapter
因为：
- execution adapter 解决“怎么下”
- market mapper 解决“下哪张”

两者职责不同。

---

## 4. 关键共享抽象（必须统一）

## 4.1 Sample Schema
训练样本与线上样本必须使用同一套 schema 观念。

建议字段包括：
- `asset`
- `horizon`
- `t0`
- `grid_id`
- 特征列
- 标签列（训练时）
- feature_version
- label_version

---

## 4.2 Signal Schema
模型输出后必须立即标准化成统一信号对象，例如：

```python
@dataclass
class Signal:
    asset: str
    horizon: str
    t0: datetime
    p_up: float
    model_version: str
    feature_version: str
    decision_context: dict
```

执行层只能吃 `Signal`，不能去碰原始 dataframe。

---

## 4.3 Decision Schema
signal policy 输出统一 decision 对象：

```python
@dataclass
class Decision:
    should_trade: bool
    side: str | None
    edge: float | None
    reason: str
    target_size: float
```

这样：
- shadow run
- live execution
- replay
都能统一处理。

---

## 4.4 Audit/Event Schema
所有事件统一日志化：
- signal generated
- market mapped
- order submitted
- order filled
- order expired
- contract resolved

后续想做 replay/backtest/shadow 对比时，这层非常关键。

---

## 5. 配置体系如何支持插件化

所有插件参数必须从统一配置对象读取。

建议配置分层：

```yaml
project:
  name: btc-polymarket-v1

market:
  asset: BTC/USD
  exchange: binance
  timeframe: 1m

horizons:
  active: ["5m"]
  specs:
    "5m":
      minutes: 5
      grid_minutes: 5
      label_builder: grid_direction
      feature_profile: core_5m
      signal_policy: default_edge_policy

features:
  profiles:
    core_5m:
      packs:
        - momentum
        - volatility
        - path_structure
        - regime
        - time
      momentum_windows: [1, 3, 5, 10, 15]
      vol_windows: [3, 5, 10, 30]

model:
  active_plugin: lightgbm
  plugins:
    lightgbm:
      random_state: 42
      class_weight: balanced

calibration:
  active_plugin: isotonic

signal:
  policies:
    default_edge_policy:
      yes_price_min: 0.40
      yes_price_max: 0.60
      edge_threshold: 0.03

sizing:
  active_plugin: fixed_fraction
  plugins:
    fixed_fraction:
      single_position_cap: 0.02

execution:
  active_adapter: polymarket
```

### 关键好处
- 增加 15m 时，只需新增一个 horizon spec
- 切换模型时，只改 active_plugin
- 开关某类特征时，只改 feature profile
- 训练和线上都读同一份配置

---

## 6. 对 agent 的明确实现要求

为了避免 agent 实现时走偏，建议明确以下规则：

### 6.1 禁止事项
- 不允许在多个地方重复实现标签逻辑
- 不允许在 strategy 内硬编码业务阈值
- 不允许在 execution 层自行构造 BTC 特征
- 不允许训练和线上各写一套 feature 逻辑
- 不允许 notebook 成为唯一真实逻辑来源

### 6.2 必须事项
- 所有业务参数从统一配置对象读取
- 所有 feature 通过统一 builder 入口生成
- 所有 label 通过统一 label plugin 生成
- 所有 signal 必须标准化输出
- 所有 decision 必须经过 signal policy
- 所有 execution 必须通过 adapter
- 所有关键模块必须可单元测试

---

## 7. 扩展路线图（架构视角）

## Stage 1：当前 V1
- BTC/USD
- 1m 数据
- 5m horizon
- 单模型
- 单 signal policy
- 单 execution adapter

## Stage 2：多 horizon
新增：
- 15m HorizonSpec
- 30m HorizonSpec
- 不同 horizon 的 feature profile
- horizon-specific threshold

## Stage 3：多模型 / 多策略
新增：
- second model plugin
- calibration comparison
- ensemble signal policy

## Stage 4：更复杂执行
新增：
- 多 market mapper
- 多 execution adapter
- 组合级别风控

---

## 8. 推荐给 agent 的实现顺序

### Step 1
先实现 Shared Core：
- config loader
- timegrid
- label plugin
- feature pack system
- schema definitions

### Step 2
实现 5m horizon plugin

### Step 3
实现 dataset builder + model plugin + calibration plugin

### Step 4
实现 signal policy + position sizing

### Step 5
实现 Polymarket execution adapter + market mapper

### Step 6
实现 shadow run / replay / audit

---

## 9. 最终架构判断标准

如果实现是正确的，那么未来新增一个 15m 市场时，应当只需要：

1. 新增一个 horizon spec
2. 选择或调整一个 feature profile
3. 训练一个对应模型
4. 在 signal/execution 配置中启用

而不应该需要：
- 重写 strategy
- 重写标签逻辑
- 重写 execution 主干
- 重写配置体系

如果新增一个 horizon 要改十几个文件，说明插件化设计失败。

---

## 10. 一句话总结

> **第一版虽然只交易 5 分钟市场，但实现时必须把 horizon、feature、model、signal、sizing、execution 都设计成可插拔组件。**
>
> **这样未来扩展不是“重写系统”，而是“新增插件 + 改配置”。**
