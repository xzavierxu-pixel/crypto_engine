# BTC/USDT 5m 方向预测：Binance 公开数据全量下载与分阶段接入实施文档（2026-01-01 起）

## 0. 当前实施状态（2026-04-11）

### 0.1 已完成

- 已完成 `data.binance.vision` 回填下载器、checksum 校验、manifest 落盘、normalized parquet 输出与 schema manifest。
- 已完成 `spot / futures_um / futures_cm / option` 的基础目录结构、按数据类型生成请求、按 `monthly + daily` 规则下载。
- 已完成 `fundingRate`、`markPriceKlines`、`indexPriceKlines`、`premiumIndexKlines`、`metrics`、`BVOLIndex`、`bookTicker` 的标准化读写链路。
- 已完成 normalized 输出接入 shared builder / dataset builder / live feature 路径，训练侧和在线侧共用同一套 shared feature 逻辑。
- 已完成 `funding`、`basis`、`OI(metrics -> OI)`、`options(BVOLIndex -> options)`、`book_ticker` 的 archive loader 与特征包接入。
- 已完成 public normalized 真实链路验证：`funding_basis_oi_options` 级别的 downstream 训练链路已经能从 normalized 输出跑通。
- 已完成对 daily-only 数据型的 period planning 修正：`metrics` 与 `BVOLIndex` 现在会从 `start_date` 连续回补到最新可得日期，而不是只补开放月尾巴。
- 已完成对衍生品 helper metadata 泄漏的修正，避免 `source_file_oi`、`checksum_status_oi` 等列进入训练集导致样本被整体清空。
- 已完成对 public `BVOLIndex` archive 输入的分钟级降采样与百分数口径缩放，避免秒级原始序列直接拖垮 1m builder。

### 0.2 已确认的现实约束

- `spot trades/aggTrades` 这种重事件流现在功能上已支持，单元测试也已通过，但拿真实整日文件做 full normalization 仍然很慢。
- 当前重事件流 normalization 已不再 OOM，但在 10 分钟超时内不一定能跑完。
- `futures_um.bookTicker` 在更早历史窗口存在，但在当前目标窗口（`2026-01-01` 之后）官方公开桶对对应月/日文件存在明显缺档，当前不应把它视为稳定可用输入。

### 0.3 未完成

- 尚未把 `spot trades/aggTrades` 的真实整日 full normalization 优化到可稳定进入常规流程。
- 尚未把 `um trades/aggTrades` 的真实整日 full normalization 优化到可稳定进入常规流程。
- 尚未对 `bookDepth`、`liquidationSnapshot`、`EOHSummary` 做同等程度的真实大窗口性能验证与下游接入评估。
- 尚未形成“重事件流常规回填 + 常规 normalized + 常规实验/数据构建”的稳定生产级吞吐基线。

### 0.4 下一阶段目标

- 当前功能侧已经没有主要缺口，下一阶段直接进入性能工作。
- 优先优化 `spot trades/aggTrades` 的 full normalization 吞吐。
- 然后优化 `futures_um trades/aggTrades` 的 full normalization 吞吐。
- 目标不是继续补功能，而是把重事件流从“能跑”推进到“能在工程上稳定常规运行”。

## 1. 文档目标

本文件用于给当前 BTC/USDT 5 分钟方向预测系统制定一个新的实施方案，核心目标只有两个：

1. **先尽量把 Binance 公开可用的相关历史数据从 2026-01-01 起下载齐、校验齐、归档齐**
2. **在不改当前 label、不改当前 shared pipeline 原则的前提下，先完成数据层验证，再分阶段接入 feature**

这份文档明确采用如下固定前提：

- **label 不改**
- 当前主任务仍为 `BTC/USDT`
- 底层频率仍为 `1m`
- 样本仍只取 5 分钟整点
- 标签仍为：

```text
y = 1{close[t0+5m] > open[t0]}
```

- 离线训练历史数据：**统一来自 `data.binance.vision`**
- 在线预测：**统一通过 Binance API / WebSocket 获取最新原始数据**
- 训练、离线评估、线上推理：**继续共用同一 shared builder，不允许复制 label / feature 逻辑**

本次实现的工作重点不是立刻追求更高的模型分数，而是：

- 先把离线与在线的数据流程跑通
- 先把可以下载的 Binance 公共数据尽量下载并归档完整
- 先把标准化、QA、manifest、lineage 做扎实
- 再用更多数据逐步提升模型表现

---

## 2. 当前策略与边界（本次不改）

### 2.1 绝对不改的内容

本轮不做以下事情：

- 不改 label 定义
- 不改 5m grid 采样方式
- 不改 execution layer 边界
- 不新增第二套 feature pipeline
- 不为了衍生品数据去改成 futures-first 架构
- 不把 Polymarket 执行逻辑重新耦合回训练层

### 2.2 本轮真正要做的事

本轮的重点是：

1. 先把 **Binance 官方公开历史数据尽量多地拉全**
2. 先把这些数据变成**可复现、可校验、可重复回补**的本地归档
3. 先验证：
   - 文件是否齐
   - 时间戳是否齐
   - schema 是否稳定
   - 离线与线上能否共用同一 raw schema
4. 在这个基础上，再按优先级分阶段做 feature 接入

---

## 3. 官方数据来源与时间窗口规则

### 3.1 离线历史的唯一来源

离线历史数据统一来自：

- **Binance Data Collection / `data.binance.vision`**

这是一手官方公开历史归档入口，按 `daily` 和 `monthly` 文件提供历史公共市场数据。

### 3.2 时间窗口

本次回补起点固定为：

- **开始日期：`2026-01-01`**

该起点在第一版实现中视为**已确认冻结**。当前目标不是回看更长历史，而是先把 `2026-01-01` 以来的离线归档、标准化、QA、在线映射与 shared builder 流程完整跑通；待离线/在线一致性确认后，再考虑把时间窗口继续向前扩展，或引入更多数据提升模型表现。

按官方归档规则，下载逻辑固定为：

- **最近完整月份之前：优先下载 `monthly`**
- **当前未完结月份：补 `daily`**

在当前日期（2026-04-11）下，本轮建议窗口为：

- `monthly`：`2026-01`、`2026-02`、`2026-03`
- `daily`：`2026-04-01` 到 **当前官方可得的最后一个日文件**

注意：

- 官方 `daily` 文件通常是**次日可用**
- 官方 `monthly` 文件通常在**次月第一个星期一**可用
- Spot 自 `2025-01-01` 起时间戳为**微秒**
- 因此本项目必须在标准化阶段统一时间单位，避免 spot / futures 对齐出错

---

## 4. 总体策略：先“尽量拿齐”，再“按价值接入”

为了满足“先尽量把 Binance 公开数据拿齐”的目标，本方案把数据层分成两步：

### Step A：原始归档尽量全

先下载并归档 **BTC 相关、且对 5m 预测可能有价值的 Binance 公共数据类型**，尽量覆盖：

- spot
- USDⓈ-M futures
- COIN-M futures
- option

这里的“尽量全”是明确的实现原则，不是可选项：

- 第一版不是只下载即将接 feature 的数据
- 第一版是先把当前可下载的数据尽量下载齐
- feature 接入顺序和下载归档顺序是两个不同层次的问题

### Step B：特征接入分层

下载齐后，不是一口气全塞进模型，而是按下列原则分层：

1. 先接入**最稳、最容易对齐、最可能有增量的信息**
2. 再接入**更重但更可能带来短周期 alpha 的数据**
3. 最后再接入**成本高、收益不确定、但值得研究的扩展数据**

这意味着：

- **下载与归档层尽量全**
- **特征接入层分阶段推进**
- **建模层仍然保持 BTC/USDT spot-centered，不改成 futures-first**

---

## 5. 本轮建议下载的数据覆盖清单

下面这张清单区分的是：

- **要不要先下载归档**
- **要不要马上接 feature**
- 以及优先级

需要明确的是：

- 只要是当前已确认可下载、且后续可能有研究价值的数据，第一版都应优先纳入下载与归档体系
- 是否进入第一轮 shared feature pipeline，则由后续实验阶段决定

---

## 6. Layer 1：必须先下载的核心数据（第一优先级）

这一层的原则是：

> **即使暂时不立刻做 feature，也应该先归档下来。**

### 6.1 Spot（BTCUSDT）

#### 必下目录

- `data/spot/monthly/klines/BTCUSDT/1m/`
- `data/spot/daily/klines/BTCUSDT/1m/`
- `data/spot/monthly/aggTrades/BTCUSDT/`
- `data/spot/daily/aggTrades/BTCUSDT/`
- `data/spot/monthly/trades/BTCUSDT/`
- `data/spot/daily/trades/BTCUSDT/`

#### 为什么必须下

这是主价格时间轴，也是所有对齐的基础。即使当前 builder 主要从 1m OHLCV 起步，后续一旦要增强 flow proxy 或研究逐笔成交，`aggTrades` 和 `trades` 会非常重要。

#### 当前接入建议

- **立即接入模型：**
  - `spot klines 1m`
- **先归档、后研究：**
  - `spot aggTrades`
  - `spot trades`

---

### 6.2 USDⓈ-M Futures（BTCUSDT）

#### 必下目录

- `data/futures/um/monthly/klines/BTCUSDT/1m/`
- `data/futures/um/daily/klines/BTCUSDT/1m/`
- `data/futures/um/monthly/markPriceKlines/BTCUSDT/1m/`
- `data/futures/um/daily/markPriceKlines/BTCUSDT/1m/`
- `data/futures/um/monthly/indexPriceKlines/BTCUSDT/1m/`
- `data/futures/um/daily/indexPriceKlines/BTCUSDT/1m/`
- `data/futures/um/monthly/premiumIndexKlines/BTCUSDT/1m/`
- `data/futures/um/daily/premiumIndexKlines/BTCUSDT/1m/`
- `data/futures/um/monthly/fundingRate/BTCUSDT/`
- `data/futures/um/daily/fundingRate/BTCUSDT/`
- `data/futures/um/monthly/bookTicker/BTCUSDT/`
- `data/futures/um/daily/bookTicker/BTCUSDT/`
- `data/futures/um/monthly/aggTrades/BTCUSDT/`
- `data/futures/um/daily/aggTrades/BTCUSDT/`
- `data/futures/um/monthly/trades/BTCUSDT/`
- `data/futures/um/daily/trades/BTCUSDT/`
- `data/futures/um/daily/metrics/BTCUSDT/`
- `data/futures/um/daily/bookDepth/BTCUSDT/`
- `data/futures/um/daily/liquidationSnapshot/BTCUSDT/`

#### 为什么必须下

这一层是 Binance 公共数据里最像“短周期 alpha 原料”的部分，原因包括：

- `mark/index/premium`：提供现货、永续、公允价之间的偏离信息
- `fundingRate`：提供杠杆拥挤与持仓成本信息
- `bookTicker`：提供最轻量级盘口结构
- `metrics`：可能包含 OI、long-short、taker flow、basis 等高价值衍生品统计
- `aggTrades/trades`：提供真实成交流
- `bookDepth`：提供更细颗粒度盘口信息
- `liquidationSnapshot`：提供爆仓、挤压类事件信息

#### 当前接入建议

- **第一批马上接 feature：**
  - `um klines`
  - `markPriceKlines`
  - `indexPriceKlines`
  - `premiumIndexKlines`
  - `fundingRate`
- **第二批优先接：**
  - `bookTicker`
  - `metrics`
- **第三批再接：**
  - `aggTrades`
  - `trades`
  - `bookDepth`
  - `liquidationSnapshot`

---

## 7. Layer 2：建议一并归档的 BTC 扩展数据（第二优先级）

### 7.1 COIN-M Futures（BTCUSD 系列）

如果目标是“尽量把 Binance 公开 BTC 数据拿齐”，建议同时归档 COIN-M 这条线，而不是只看 USDⓈ-M。

#### 建议目录

- `data/futures/cm/monthly/klines/`
- `data/futures/cm/daily/klines/`
- `data/futures/cm/monthly/markPriceKlines/`
- `data/futures/cm/daily/markPriceKlines/`
- `data/futures/cm/monthly/indexPriceKlines/`
- `data/futures/cm/daily/indexPriceKlines/`
- `data/futures/cm/monthly/premiumIndexKlines/`
- `data/futures/cm/daily/premiumIndexKlines/`
- `data/futures/cm/monthly/fundingRate/`
- `data/futures/cm/daily/fundingRate/`
- `data/futures/cm/monthly/bookTicker/`
- `data/futures/cm/daily/bookTicker/`
- `data/futures/cm/monthly/aggTrades/`
- `data/futures/cm/daily/aggTrades/`
- `data/futures/cm/monthly/trades/`
- `data/futures/cm/daily/trades/`
- `data/futures/cm/daily/metrics/`
- `data/futures/cm/daily/bookDepth/`
- `data/futures/cm/daily/liquidationSnapshot/`

#### 说明

这部分在第一版中**确认启用下载归档**，而不是只做空壳支持。原因是：

- 本轮目标之一就是把能下载的数据尽量下载齐
- COIN-M 虽然不一定立刻进入 feature，但后续作为 context 的研究价值明确存在
- 下载器、manifest、normalized、QA 如果一开始就不覆盖 COIN-M，后续会产生结构性返工

但同时要明确：

- COIN-M 第一版**启用下载**
- COIN-M 第一版**不进入第一轮 shared feature pipeline**
- COIN-M 第一版**不改变主任务仍为 BTCUSDT 的 spot-centered 建模结构**

#### 当前接入建议

- **第一版就启用下载归档**
- **第一版就完成 raw/normalized/manifest/QA 支持**
- **第一版仍不进入第一轮特征**
- 等 USDⓈ-M 跑稳后，再决定是否作为二级 context 接入

---

### 7.2 Options（BTC 相关）

#### 建议目录

- `data/option/daily/BVOLIndex/BTCBVOLUSDT/`
- `data/option/daily/EOHSummary/BTCUSDT/`

#### 为什么要下

即使现在不立刻做完整 options 链特征，这两类公开数据也值得先归档：

- `BVOLIndex`：可作为 BTC 期权隐含波动率 regime proxy
- `EOHSummary`：可作为期权层摘要信息的研究输入

#### 当前接入建议

- **先归档**
- `BVOLIndex` 可以作为**后期 regime 特征**
- `EOHSummary` 先不急着接 feature，先研究 schema

---

## 8. Layer 3：当前不作为主目标，但脚本应预留支持

如果下载器做成参数化，建议一开始就支持：

- spot 其他 BTC 现货对
- futures 其他 BTC 相关 symbol
- 后续 ETHUSDT / ETH perpetual 扩展
- 将来做 cross-symbol context 的下载能力

这不是因为现在就要用，而是因为：

> **下载器一旦做成通用的，以后就不会为了新 symbol 重写一遍。**

这里的“预留支持”指的是：

- 配置结构要支持
- 路径模板要支持
- manifest 结构要支持
- 标准化和 QA 入口要支持

但第一版仍然只围绕 BTC 相关数据落地，不主动把项目扩成多资产框架。

---

## 9. 下载范围的具体实施顺序

下面是建议执行顺序。

### Phase 0：确认核心窗口与落地原则

当前固定为：

- 起点：`2026-01-01`
- 完整月份：`2026-01` ~ `2026-03`
- 尾部日文件：`2026-04-01` ~ 当前官方可得最后一天

本阶段还固定以下实现原则：

- 第一版目标是**先把离线与在线流程打通**
- 第一版目标是**先把能下载的数据尽量下载并归档好**
- 第一版目标不是立刻把所有新增数据接进 feature
- 缺文件或 checksum 失败时：
  - **单文件不中断整批**
  - **继续处理其他文件**
  - **最终整体返回非零状态**
  - **问题完整写入 manifest**
- `raw` 层**不保留 zip 原包**
- `normalized` 层统一输出 **Parquet**

### Phase 1：先完成第一批可立即支撑流程打通的下载归档

先完成以下目录的下载、校验、解压、manifest、normalized 与 QA 支持：

1. `spot klines 1m`
2. `um klines 1m`
3. `um markPriceKlines 1m`
4. `um indexPriceKlines 1m`
5. `um premiumIndexKlines 1m`
6. `um fundingRate`
7. `um bookTicker`
8. `um metrics`
9. `option BVOLIndex`

本阶段结束后，应能支撑：

- baseline 主时间轴
- 最轻一批衍生品数据的 raw/normalized 打通
- 在线侧字段映射开始与离线统一
- 第一轮最轻 feature 接入实验具备前置条件

### Phase 2：补全其余已确认可下载的数据归档

然后继续补齐以下目录，同样要求下载、校验、解压、manifest、normalized 与 QA 全部到位：

10. `spot aggTrades`
11. `spot trades`
12. `um aggTrades`
13. `um trades`
14. `um bookDepth`
15. `um liquidationSnapshot`
16. `option EOHSummary`

本阶段核心目标是：

- 把当前已知高价值但更重的数据先归档齐
- 先把研究素材准备好，而不是急着接 feature
- 为后续微观结构实验、期权摘要实验预留标准化输入

### Phase 3：扩展 COIN-M，并保持主任务仍以 BTCUSDT 为中心

最后补：

17. `cm klines`
18. `cm mark/index/premium`
19. `cm fundingRate`
20. `cm bookTicker`
21. `cm metrics`
22. `cm aggTrades/trades`
23. `cm bookDepth`
24. `cm liquidationSnapshot`

这样能兼顾：

- 最快开始实验
- 不错过重要公开数据
- 避免一开始就被最重文件拖垮
- 同时落实“COIN-M 第一版也启用下载，但不直接改成 futures-first 建模架构”

---

## 10. 统一目录结构（建议直接采用）

建议离线归档统一落到：

```text
artifacts/data/binance_public/
  raw/
    spot/
      klines/
      aggTrades/
      trades/
    futures_um/
      klines/
      markPriceKlines/
      indexPriceKlines/
      premiumIndexKlines/
      fundingRate/
      bookTicker/
      metrics/
      aggTrades/
      trades/
      bookDepth/
      liquidationSnapshot/
    futures_cm/
      klines/
      markPriceKlines/
      indexPriceKlines/
      premiumIndexKlines/
      fundingRate/
      bookTicker/
      metrics/
      aggTrades/
      trades/
      bookDepth/
      liquidationSnapshot/
    option/
      BVOLIndex/
      EOHSummary/
  normalized/
    spot/
      *.parquet
    futures_um/
      *.parquet
    futures_cm/
      *.parquet
    option/
      *.parquet
  manifests/
    download_manifest.json
    file_checksums.json
    schema_manifest.json
```

### 原则

- `raw/` 保留官方 zip 解压后的原始结构语义
- `raw/` **不保留 zip 原包**；zip 下载后只用于校验与解压，完成后可删除
- `.CHECKSUM` 文件本身是否保留可实现时决定，但**checksum 结果必须写入 manifest**
- `normalized/` 只做统一 schema 转换，输出格式固定为 **Parquet**，不做 feature engineering
- `manifests/` 记录所有文件、日期范围、校验结果、缺口、重试记录、标准化输出路径与 lineage 关系

---

## 11. 下载脚本设计（本次应一次性做对）

建议新增：

```text
scripts/backfill_binance_public_history.py
```

### 职责

这个脚本只负责：

1. 根据配置枚举要下载的 `market/data-type/symbol/interval`
2. 按月份优先、按天补尾部
3. 下载 zip 文件
4. 下载对应 `.CHECKSUM`
5. 做 SHA256 校验
6. 解压
7. 删除 zip 原包
8. 记录 manifest
9. 遇到缺文件时明确报错或记录 missing，而不是静默跳过

进一步细化为：

1. 按配置生成完整下载计划
2. 区分 `monthly` 与 `daily` 的优先级与尾部补齐逻辑
3. 对每个目标尝试下载 zip 与 `.CHECKSUM`
4. 计算并验证 SHA256
5. 解压到 `raw/`
6. 删除 zip 原包，仅保留解压结果与校验记录
7. 把成功、失败、missing、checksum mismatch 全部写入 manifest
8. 批次结束时给出总状态：
   - 若全部成功，返回零状态
   - 若存在 missing、checksum 失败、解压失败，则返回非零状态

### 返回码策略

实现时应明确采用如下策略：

- **单文件失败不立即中断整批**
- **其余文件继续处理**
- **最终只要存在 missing、checksum mismatch、解压失败、schema 失败中的任一项，整批任务返回非零状态**
- **所有失败项必须完整写入 manifest**

### 禁止

- 不在下载脚本里直接做模型特征
- 不在下载脚本里直接生成训练集
- 不在下载脚本里直接拼接 live-only 逻辑

---

## 12. 配置化要求

所有下载目标必须配置化，避免硬编码。

建议在 `settings.yaml` 新增：

```yaml
data_backfill:
  provider: binance_public
  start_date: "2026-01-01"
  use_monthly_for_full_months: true
  use_daily_for_open_month_tail: true
  verify_checksum: true

  spot:
    enabled: true
    symbols: ["BTCUSDT"]
    data_types:
      klines:
        intervals: ["1m"]
      aggTrades: {}
      trades: {}

  futures_um:
    enabled: true
    symbols: ["BTCUSDT"]
    data_types:
      klines:
        intervals: ["1m"]
      markPriceKlines:
        intervals: ["1m"]
      indexPriceKlines:
        intervals: ["1m"]
      premiumIndexKlines:
        intervals: ["1m"]
      fundingRate: {}
      bookTicker: {}
      metrics: {}
      aggTrades: {}
      trades: {}
      bookDepth: {}
      liquidationSnapshot: {}

  futures_cm:
    enabled: true
    symbols: []
    data_types:
      klines:
        intervals: ["1m"]
      markPriceKlines:
        intervals: ["1m"]
      indexPriceKlines:
        intervals: ["1m"]
      premiumIndexKlines:
        intervals: ["1m"]
      fundingRate: {}
      bookTicker: {}
      metrics: {}
      aggTrades: {}
      trades: {}
      bookDepth: {}
      liquidationSnapshot: {}

  option:
    enabled: true
    symbols:
      BVOLIndex: ["BTCBVOLUSDT"]
      EOHSummary: ["BTCUSDT"]
```

### 说明

由于当前已确认第一版就启用 COIN-M 下载，这里不应再把 `futures_cm.symbols` 长期留空。更合理的落地方式是：

- 第一版先把脚本、目录、manifest、normalized、QA 全部支持好
- `futures_cm.symbols` 在实施时填入明确要回补的 BTC 相关合约集合
- 但在 feature 接入阶段，仍保持 COIN-M 只作为后续扩展上下文，不进入第一轮主干实验

换言之：

- **下载与归档层：COIN-M 第一版启用**
- **特征与建模层：COIN-M 第一版默认不启用**

---

## 13. 统一标准化 schema（非常关键）

下载之后，不要直接拿各目录原始 CSV 去喂训练。

应该统一进入 `normalized` 层，每个 family 做一个标准化表。

### 13.1 必备公共字段

每张 normalized 表至少应有：

- `timestamp`
- `symbol`
- `market_family`（`spot` / `futures_um` / `futures_cm` / `option`）
- `data_type`
- `interval`（如适用）
- `source_file`
- `source_date`
- `source_granularity`（`monthly` / `daily`）
- `source_version`
- `ingested_at`
- `checksum_status`
- `raw_timestamp`

如果原始文件存在多个时间字段，还应保留最直接、最原始的那个时间字段，避免后续排查困难。

### 13.2 时间标准

统一要求：

- 全部转成 `UTC`
- 内部统一到一个时间精度（建议 `datetime64[ns, UTC]`）
- 保留原始时间戳列，避免排查困难
- Spot 自 `2025-01-01` 起的微秒时间戳，在标准化阶段统一落到内部统一精度
- 不允许在不同 family 之间混用未经标准化的原始时间单位

### 13.3 原始数值不提前做交易意义假设

例如：

- `bookTicker` 保留 bid/ask 与 qty
- `fundingRate` 保留原始 funding 时间
- `metrics` 第一版以 **Binance 原始组织形式为准**
- `metrics` 先保留原始字段名
- `metrics` 允许附加解释性别名字段
- 但不要在第一版里强行抽象成统一衍生指标语义层
- 不要在 normalized 层就提前做“买卖方向标签”之类推断

这项原则的目的是：

- 先把数据下载与标准化做对
- 先让研究阶段可以对照 Binance 原始字段核查
- 避免过早抽象导致后续排错困难

---

## 14. 数据质量检查（下载后必须执行）

每一类数据都必须跑如下 QA。

### 14.1 文件级

- zip 是否成功下载过
- checksum 是否匹配
- 是否可解压
- 是否有空文件
- zip 是否按策略删除
- 解压后文件是否落到预期路径

### 14.2 表级

- schema 是否符合预期
- 时间戳是否单调
- 是否存在重复时间戳
- 是否存在非法负值
- 是否存在明显断档

但“明显断档”需要按数据类型分级处理，不能一刀切：

- `klines 1m`：第一版就做**严格分钟级连续性检查**
- `fundingRate`：检查事件时间是否合理、是否存在异常倒序或重复，不要求分钟连续
- `bookTicker` / `aggTrades` / `trades` / `bookDepth` / `liquidationSnapshot`：第一版先检查
  - 非空
  - 时间单调
  - 关键列存在
  - 是否可被正确聚合或重采样
  - 是否有异常重复或明显损坏
- `option` 系列：第一版优先保证 schema 稳定与时间列可解释，不先施加过强连续性假设

### 14.3 跨表级

- `spot klines` 与 `um klines` 是否能对齐
- `mark/index/premium` 是否能按 asof merge 到 1m 主时间轴
- `fundingRate` 是否能安全 forward-fill 到下一次 funding 前
- `bookTicker` / `aggTrades` / `trades` 是否能正确聚合成 1m/5m 特征输入
- `option` 数据是否能以合理频率并入主时间轴或 regime 逻辑

### 14.4 manifest 级

manifest 至少记录：

- 下载开始时间 / 结束时间
- 文件数
- 成功数 / 失败数
- 缺失文件清单
- checksum 结果
- 标准化输出路径
- 行数摘要
- 是否保留 zip（本方案固定为否）
- 解压后原始文件路径
- 最终任务返回码依据

---

## 15. 先测试数据，不先改特征：具体怎么做

这部分是本次文档的核心思想。

### 15.1 第一阶段测试目标

不是马上追求最好 AUC，而是先回答：

1. 数据是否真下齐了？
2. schema 是否稳定？
3. builder 是否能读？
4. train/live parity 是否还成立？
5. 哪些数据类型最值得先接 feature？

并且这里的“第一阶段”要明确理解为：

- **先验证数据层与流程层**
- **先验证下载、标准化、QA、shared builder、在线映射是否闭环**
- **不是先做大规模特征竞赛**

### 15.2 第一阶段不该做的事

- 不改 label
- 不做大规模 feature 重构
- 不同时改模型、阈值、执行
- 不一开始就把 trades / depth / options 全塞进模型

### 15.3 第一阶段应该做的实验

先做下面 4 组：

#### Exp 1：现有 baseline 复跑

只用当前 baseline 数据与现有 feature，确认新增数据归档没有破坏主干。

这一步的重点不是提分，而是确认：

- 现有离线训练仍可运行
- shared builder 未被破坏
- 新增数据目录与配置没有污染旧主流程

#### Exp 2：最轻衍生品接入

加入：

- `fundingRate`
- `markPriceKlines`
- `indexPriceKlines`
- `premiumIndexKlines`

这一步的重点是确认：

- 非 OHLCV 数据能否稳定 asof 对齐到主时间轴
- train/live parity 是否还能保持
- schema 与对齐逻辑是否值得固化到 shared builder

#### Exp 3：盘口统计接入

在 Exp 2 基础上加入：

- `bookTicker`
- `metrics`

这一步重点是：

- 验证轻量盘口与统计类衍生品数据能否在不复制逻辑的前提下进入 shared builder
- 验证 `metrics` 原样字段保留策略是否足够支撑实验

#### Exp 4：期权 regime 接入

在 Exp 3 基础上加入：

- `BVOLIndex`

这一步重点是：

- 验证 option 类摘要数据是否适合作为 regime/context 信号
- 验证低频摘要数据并入主 1m/5m 体系时的对齐方式是否稳定

### 15.4 第二阶段再碰的内容

等以上稳定之后，再做：

- `aggTrades`
- `trades`
- `bookDepth`
- `liquidationSnapshot`
- `EOHSummary`
- `COIN-M context`

---

## 16. 推荐的 feature 接入顺序（数据下载完成后）

本节只决定“接 feature 的顺序”，不决定“先不先下载”。

### 第一梯队：先接

1. `um fundingRate`
2. `um markPriceKlines`
3. `um indexPriceKlines`
4. `um premiumIndexKlines`

### 第二梯队：再接

5. `um bookTicker`
6. `um metrics`
7. `option BVOLIndex`

### 第三梯队：重数据后接

8. `um aggTrades`
9. `um trades`
10. `spot aggTrades`
11. `spot trades`
12. `um bookDepth`
13. `um liquidationSnapshot`

### 第四梯队：扩展接入

14. `option EOHSummary`
15. `cm futures` 系列

这部分要再次强调：

- **feature 接入顺序晚于下载归档顺序**
- **下载层尽量全，特征层再分层**

---

## 17. 在线预测对应规则

离线历史已经规定统一来自 `data.binance.vision`，线上则必须使用 Binance API / WebSocket。

### 在线侧原则

- 在线只拉**最新原始数据**
- 不在 execution 层重算特征
- 由 shared feature builder 统一重算
- 保持与离线同 schema、同字段名、同对齐逻辑

### 在线侧大致映射

- `spot klines` → Spot API / WebSocket kline
- `um klines` → Futures API / WebSocket kline
- `mark/index/premium/funding` → Futures REST / stream
- `bookTicker` → WebSocket
- `aggTrades` → WebSocket
- 更重的盘口流按需订阅

当前第一版实施要关注的，不是把所有在线订阅一次性做完，而是：

- 离线标准化字段命名要能映射到在线原始数据
- 在线只负责补最新原始输入
- 特征的重算与对齐逻辑必须仍然由 shared builder 统一承担

---

## 18. 为什么本次先“全量下载”，后“精细接入”

因为当前要解决的不是“没有下一个 feature 点子”，而是：

- 还没有把 Binance 公共数据资产系统化归档
- 很多高价值数据只在短窗口测试过，结论不稳
- 如果现在就继续改 feature，而数据层不稳，后面很难复现与比较

所以本次正确顺序是：

```text
先下齐
→ 先校验
→ 先做轻量接入测试
→ 再扩 feature
→ 最后再做更重微观结构与期权摘要
```

并且这里的“先下齐”包括：

- spot
- futures_um
- futures_cm
- option

只要当前能下载并对后续研究有价值，就应该优先进入归档与标准化体系。

---

## 19. 本轮实施清单（可以直接照着做）

### Phase A：下载器与下载清单

- [ ] 新增 `scripts/backfill_binance_public_history.py`
- [ ] 新增 `settings.yaml` 中的 `data_backfill` 段
- [ ] 支持 monthly + daily 混合下载
- [ ] 支持 checksum 校验
- [ ] 支持 manifest 输出
- [ ] 支持“单文件不中断整批，最终按整体状态返回非零”
- [ ] 下载后删除 zip，仅保留 checksum 结果与解压文件
- [ ] 支持 spot / futures_um / futures_cm / option 四大族
- [ ] 将可下载目标尽量下载齐，而不是只覆盖即将接 feature 的子集

### Phase B：raw 归档层

- [ ] Spot BTCUSDT：`klines / aggTrades / trades`
- [ ] Futures UM BTCUSDT：`klines / mark / index / premium / funding / bookTicker / metrics / aggTrades / trades / bookDepth / liquidationSnapshot`
- [ ] Futures CM：按确认的 BTC 相关合约集合下载同类数据
- [ ] Option：`BVOLIndex / EOHSummary`
- [ ] 所有目标均写入下载与缺失 manifest
- [ ] 所有目标均保留 checksum 结果

### Phase C：标准化层

- [ ] 统一 raw → normalized schema
- [ ] 输出统一为 Parquet
- [ ] 统一时间戳到 `UTC`
- [ ] 保留原始时间列
- [ ] `metrics` 先保留 Binance 原始字段组织
- [ ] 统一 manifest 与 lineage 记录
- [ ] 写 QA 检查

### Phase D：数据质量与流程验证

- [ ] 文件级 QA
- [ ] 表级 QA
- [ ] 跨表对齐 QA
- [ ] `klines 1m` 严格连续性检查
- [ ] 其他高频事件流的非空、单调、可聚合检查
- [ ] 离线输入 schema 与在线侧目标 schema 对照
- [ ] baseline 复跑，确认主干未破坏

### Phase E：第一轮特征接入实验

- [ ] 轻衍生品接入实验：`fundingRate / mark / index / premium`
- [ ] `bookTicker + metrics` 实验
- [ ] `BVOLIndex` 实验
- [ ] 确认哪些新增数据值得正式并入 shared feature pipeline

### Phase F：第二轮再做重数据接入

- [ ] aggTrades / trades 接入
- [ ] bookDepth 接入
- [ ] liquidationSnapshot 接入
- [ ] EOHSummary 研究
- [ ] COIN-M 扩展

这里再次强调：

- COIN-M 的**下载归档**在前面阶段就应完成
- 这里说的 “COIN-M 扩展” 指的是**是否进入特征层与模型层**

---

## 20. 成功标准

本轮算完成，至少满足：

1. 从 `2026-01-01` 起的 Binance 公共数据已经按计划下载归档
2. `monthly + daily` 混合回补逻辑可复现
3. 每类数据都有 checksum / manifest / QA
4. 当前 `5m` label 与 shared pipeline 完全不变
5. baseline 未被破坏
6. zip 原包不保留，但 checksum 结果、解压结果与 lineage 记录完整
7. normalized 层统一输出为 Parquet
8. 第一批衍生品与期权 regime 数据已能进入统一 feature pipeline
9. 重数据虽未全部接 feature，但已经归档完毕，后续可直接做实验

---

## 21. 一句话总结

这次实施不改 label，不先大改 feature。

这次实施的核心是：

> **从 2026-01-01 起，尽量把 Binance 官方公开 BTC 相关数据先下载齐、校验齐、归档齐；**
> **先把离线与在线流程打通，再做轻量接入测试，最后按价值逐层引入更重的数据。**

这是当前阶段最稳、最可复现、也最容易真正把后续实验做扎实的路线。
