# BTC/USDT 5m 方向预测：衍生品数据接入架构修订版（详细可执行）

## 1. 文档目的

本文件用于修订当前“衍生品数据接入”方案，给出一套：

- 与现有 spot `BTC/USDT` 主模型兼容
- 保持在线/离线逻辑一致
- 可逐步落地
- 便于 Claude Code / Codex 实现与审查
- 不破坏现有 shared builder / thin strategy / separate execution 原则

的详细方案。

本文默认前提：

- 当前主模型任务不变：
  - 主标的：`BTC/USDT`
  - 底层频率：`1m`
  - 标签：`y = 1{close[t0+5m] > open[t0]}`
  - 样本只取 5 分钟整点网格
- 训练和推理继续由 shared core 统一处理
- Polymarket 执行层继续解耦，不参与特征计算

---

## 2. 核心结论（必须先统一）

## 2.1 不采用“纯方案 A”
不建议将整个系统切成“Freqtrade futures bot 模式”来承载衍生品数据。

原因：
- 当前主模型以 spot `BTC/USDT` 为中心
- 现有 shared builder / dataset builder / signal service 已围绕 spot 主流程建立
- 若直接切成 futures-first，会显著增加耦合和迁移成本
- futures 模式下的数据结构和 pair 命名（如 `BTC/USDT:USDT`）与当前主流程并不一致

---

## 2.2 采用“方案 A-lite / 混合方案”
最优做法是：

> **用 Freqtrade 负责“拿原始衍生品数据”**
>
> **用 shared builder 负责“统一构造衍生品特征”**
>
> **Execution 层只消费 signal，不重算特征**

也就是说：

### Freqtrade / 原始数据层
负责：
- 下载 spot `BTC/USDT`
- 下载 futures `mark`
- 下载 futures `funding_rate`
- （可选）辅助 futures 原始序列

### Shared core / feature 层
负责：
- 对齐 spot 与 futures 原始时间序列
- 生成 derivatives feature packs
- 保证训练和线上使用相同逻辑

### Signal layer
负责：
- 从最新原始 spot + derivatives 序列现场重算特征
- 调模型推理

### Execution layer
负责：
- 读取 signal
- 读取 Polymarket 市场价格
- 决策与下单
- 不重算 BTC 或衍生品特征

---

## 3. 为什么这是正确的方案

## 3.1 保持在线/离线一致
你当前线上流程本质是：

- 输入最新原始 OHLCV
- 通过 shared feature builder 现场重算特征
- 再喂模型预测

这本身是合理的，优点是：
- 逻辑与训练一致
- 不依赖单独的特征落库服务
- 更容易保证 feature parity

需要修正的只是架构描述：
- **Signal layer 可以重算特征**
- **Execution layer 不可以重算特征**

---

## 3.2 不把 futures informative pair 强塞进 spot strategy
当前不建议依赖“在 spot strategy 中直接混 futures informative candles”作为主方案。

原因：
- 这会增加交易模式边界复杂度
- 容易在策略层写出隐式耦合逻辑
- 不利于保持 strategy 薄层设计
- 不利于 future derivatives 数据源扩展（如 OI / options）

---

## 3.3 衍生品原始数据和 spot 主线解耦
一旦 derivatives 数据层独立，你未来可以：
- 替换交易所
- 新增 OI
- 新增 options IV
- 加入额外 perp 市场

而无需推翻主模型管线。

---

## 4. 衍生品数据接入的总分层

建议新增以下目录：

```text
src/
  data/
    derivatives/
      funding_loader.py
      basis_loader.py
      oi_loader.py
      options_loader.py
      aligner.py
      feature_store.py

  features/
    derivatives_funding.py
    derivatives_basis.py
    derivatives_oi.py
    derivatives_options.py
```

---

## 5. 三阶段接入计划

## Phase 1：Funding + Basis（必须先做）

这是当前最现实、最有希望提供正交 alpha 的部分。

### 5.1 目标
先接入以下原始数据：

- funding rate
- mark price
- index price
- premium index
- 现货 spot close（已有）

再由 builder 构造：
- funding 特征
- basis 特征

---

### 5.2 原始数据来源

#### 5.2.1 Spot
- 来源：现有 `BTC/USDT` 原始 OHLCV

#### 5.2.2 Futures mark / funding
优先方式：
- 使用 Freqtrade 数据下载能力获取 futures `mark` / `funding_rate` 原始数据
- 或使用单独脚本从 Binance 官方 API 抓取并落地
- 训练和线上统一读取本地标准化文件

> 无论下载来源是什么，最终都必须进入统一的原始数据表，再由 shared builder 处理。

---

### 5.3 原始数据 schema

建议至少落成：

```text
artifacts/data/derivatives/binance_btcusdt_perp_raw.parquet
```

字段建议：

- `timestamp`
- `symbol`
- `exchange`
- `mark_price`
- `index_price`
- `premium_index`
- `funding_rate`
- `funding_effective_time`
- `source_version`

说明：
- `funding_rate` 可按最近已知值 forward-fill
- `funding_effective_time` 要保留，防止未来穿越
- `mark_price`、`index_price`、`premium_index` 统一按 1m 对齐

---

### 5.4 derivatives feature packs（Phase 1）

#### 新增文件
- `src/features/derivatives_funding.py`
- `src/features/derivatives_basis.py`

#### Funding 特征建议
- `funding_rate`
- `funding_rate_lag1`
- `funding_rate_change_1`
- `funding_rate_zscore_30d`
- `funding_is_pos`
- `funding_abs`

#### Basis 特征建议
- `basis_mark_spot = mark_price / spot_close - 1`
- `basis_index_spot = index_price / spot_close - 1`
- `premium_index`
- `basis_mark_spot_lag1`
- `basis_mark_spot_change_1`
- `basis_mark_spot_zscore`
- `premium_index_zscore`
- `basis_sign`

---

### 5.5 时间对齐规则（极重要）

#### funding
funding 通常不是 1m 频率，必须按“最近一次已知值”使用。

规则：
- 对每个 `t0`，只能使用 `timestamp <= t0` 的最近 funding
- 允许在下一个 funding 生效前 forward-fill
- 不允许对未来 funding 做线性插值
- 不允许提前看到未来 funding

#### mark / index / premium
规则：
- 统一 resample 到 1m
- 只允许用已完成的 1m bar
- 统一在 builder 中做 asof merge

---

### 5.6 Phase 1 builder 接入方式

#### Data layer
`feature_store.py` 输出统一 derivatives frame：

- index = timestamp
- columns = 原始衍生品字段 + 预清洗字段

#### Feature builder
主 `build_feature_frame()` 扩展：
1. 先构造现有 spot 特征
2. 再 merge derivatives raw frame
3. 再执行 derivatives feature packs
4. 返回完整 feature frame

---

## Phase 2：Open Interest（第二优先级）

## 6.1 目标
新增 OI 相关原始数据与特征。

### 6.2 原始数据来源
通过单独 loader 获取：
- 当前 OI
- OI 历史统计

建议不要把 OI 获取逻辑塞进 strategy 或 signal service。

新增：
- `src/data/derivatives/oi_loader.py`

输出：
```text
artifacts/data/derivatives/binance_btcusdt_oi_raw.parquet
```

字段建议：
- `timestamp`
- `open_interest`
- `oi_notional`（如可得）
- `source_version`

---

### 6.3 OI 特征建议
新增：
- `src/features/derivatives_oi.py`

建议特征：
- `oi_level`
- `oi_change_5m`
- `oi_change_1h`
- `oi_zscore`
- `oi_slope`
- `oi_x_basis`
- `oi_x_funding`

其中：
- `oi_x_basis = oi_change_5m * basis_mark_spot`
- `oi_x_funding = oi_change_5m * funding_rate`

---

### 6.4 OI 时间对齐规则
- 若原始 OI 频率为 5m 或更低，使用 asof merge
- 不插值未来
- 可在下一个已知点前 forward-fill，但需要记录原始更新时间

---

## Phase 3：Options IV（第三优先级）

## 7.1 目标
只有在 funding + basis + OI 都落地且验证有意义之后，再考虑接 IV。

### 7.2 原则
当前不建议先接全期权链 minute-by-minute。
建议只做低频 IV regime 层。

新增：
- `src/data/options/options_loader.py`
- `src/features/derivatives_options.py`

建议字段：
- `atm_iv_near`
- `iv_term_slope`
- `iv_change_1h`
- `iv_regime`

---

## 8. 推荐配置方案

所有衍生品参数必须进入统一配置。

建议在 `settings.yaml` 增加：

```yaml
derivatives:
  enabled: true
  exchange: binance
  symbol_spot: BTC/USDT
  symbol_perp: BTC/USDT:USDT

  funding:
    enabled: true
    source: freqtrade_or_binance
    ffill_until_next: true
    zscore_window: 720

  basis:
    enabled: true
    use_mark_price: true
    use_index_price: true
    use_premium_index: true
    zscore_window: 720

  oi:
    enabled: false
    frequency: 5m
    zscore_window: 288

  options:
    enabled: false

features:
  profiles:
    core_5m:
      packs:
        - momentum
        - volatility
        - path_structure
        - regime
        - volume
        - candle_structure
        - market_quality
        - time
        - derivatives_funding
        - derivatives_basis
```

---

## 9. 代码模块职责（必须严格遵守）

## 9.1 funding_loader.py
职责：
- 下载或读取 funding 原始数据
- 标准化字段
- 落盘 / 返回 DataFrame

禁止：
- 在这里直接构造模型特征

---

## 9.2 basis_loader.py
职责：
- 下载或读取 mark/index/premium 原始数据
- 标准化字段
- 落盘 / 返回 DataFrame

禁止：
- 在这里做 feature engineering

---

## 9.3 aligner.py
职责：
- 统一 spot / derivatives 时间戳
- asof merge
- 统一缺失值处理
- 防未来穿越

禁止：
- 直接生成最终模型特征

---

## 9.4 feature_store.py
职责：
- 提供统一的 derivatives raw frame 给 builder
- 训练与线上都通过这一层获取衍生品原始数据

---

## 9.5 derivatives_funding.py / derivatives_basis.py / derivatives_oi.py
职责：
- 纯特征工程
- 输入：已对齐的 raw frame
- 输出：新增衍生品特征列

禁止：
- 访问网络
- 读取配置之外的硬编码常量
- 直接做落盘

---

## 9.6 SignalService
职责：
- 读取最新 spot 原始行情
- 读取最新 derivatives raw frame
- 统一调用 shared builder 生成最终特征
- 模型推理

禁止：
- 在这里单独写 funding/basis 特征逻辑
- 在这里绕过 shared builder 手工拼列

---

## 9.7 Execution layer
职责：
- 使用 `Signal`
- 使用 Polymarket CLOB 价格
- 做 decision / sizing / routing

禁止：
- 重新读取 derivatives raw 数据
- 重新计算衍生品特征
- 依赖 builder

---

## 10. 训练与线上流程（修订版）

## 10.1 线下训练

### Step 1
准备主 spot 原始数据：
- BTC/USDT 1m

### Step 2
准备 derivatives 原始数据：
- funding
- mark
- index
- premium

### Step 3
通过统一 aligner 生成训练可用 raw dataset

### Step 4
通过 shared feature builder 构造完整训练特征

### Step 5
训练模型并保存：
- model artifact
- feature version
- config hash
- derivatives schema version

---

## 10.2 线上推理

### Step 1
输入最新 spot OHLCV

### Step 2
读取最新已知 derivatives raw frame

### Step 3
通过 shared feature builder 现场重算全部特征

### Step 4
输出 signal

### Step 5
execution 使用 signal + Polymarket 价格进行决策

---

## 11. 需要修正的项目原则表述

当前原则：

> Execution must not recompute BTC features

建议修正为：

> **Execution layer must not recompute BTC or derivatives features.**
>
> **Signal layer may recompute features using the shared builder from the latest raw inputs.**

这样更准确，也更符合你当前设计。

---

## 12. 单元测试与验证要求

新增测试：

### 数据层
- `test_derivatives_aligner.py`
- `test_funding_loader_schema.py`
- `test_basis_loader_schema.py`

### 特征层
- `test_derivatives_funding.py`
- `test_derivatives_basis.py`
- `test_derivatives_oi.py`

### 一致性
- `test_train_live_feature_parity_with_derivatives.py`

---

## 13. 对 Claude Code / Codex 的明确实施要求

## 必须遵守
1. 不允许在 strategy 中直接抓 derivatives 数据
2. 不允许在 SignalService 之外重复实现衍生品特征
3. 不允许在 Execution 层读取 derivatives raw frame
4. 所有 derivatives 参数必须来自统一配置
5. 所有时间对齐必须通过统一 aligner
6. 所有 derivatives 特征必须是 leak-free
7. 训练和线上必须共用同一 builder

## 首期只做
- funding
- basis

不要首期就上：
- OI
- options IV
- 复杂多交易所融合
- execution 优化

---

## 14. 推荐实施顺序（详细）

## Phase 1：目录与配置
- 新增 `src/data/derivatives/`
- 新增配置项
- 新增 schema / type definitions

## Phase 2：原始数据层
- 实现 `funding_loader.py`
- 实现 `basis_loader.py`
- 实现 `aligner.py`
- 实现 `feature_store.py`

## Phase 3：特征层
- 实现 `derivatives_funding.py`
- 实现 `derivatives_basis.py`
- builder 接入

## Phase 4：训练验证
- 跑完整训练
- 比较：
  - baseline
  - baseline + funding
  - baseline + funding + basis

## Phase 5：线上推理接入
- SignalService 读取 derivatives raw frame
- 复用 shared builder
- 验证 live / train parity

## Phase 6：再决定是否进入 OI
只有在 funding+basis 确认有边际价值后，才进入 OI。

---

## 15. 最终建议

当前最优方案不是：

- 把整套系统改成 futures 模式

而是：

> **保留现有 spot 主模型主线**
>
> **用 Freqtrade / Binance 数据能力获取原始衍生品数据**
>
> **通过 shared builder 把 derivatives 作为统一特征层接入**
>
> **让 SignalService 继续现场重算特征**
>
> **让 Execution layer 严格只消费 signal**

一句话总结：

> **衍生品数据应作为“额外原始输入层”并入 shared feature pipeline，而不是改写当前系统的主运行形态。**