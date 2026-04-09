# BTC/USDT 5m 方向预测：下一版 Feature Roadmap

## 文档目的

本文件用于指导下一版特征工程优化，目标是：

- 在不破坏现有工程主干的前提下，提高 5 分钟方向预测的可交易性
- 尽量优先引入正交信息，而不是继续堆叠同类指标
- 保持当前架构原则不变：
  - 在线与离线共用同一套 builder
  - 所有特征来自统一 feature pack 体系
  - 所有参数由统一配置文件管理
  - 新增特征以插件化方式实现，避免影响既有模块

---

## 当前判断

当前 `core_5m` 已经覆盖了以下 8 组特征：

- momentum
- volatility
- path_structure
- regime
- volume
- candle_structure
- market_quality
- time

总计约 63 个共享特征。

这说明当前问题不太像“特征数量不够”，而更像：

1. 特征类型还不够强
2. 缺少更高层级的上下文信息
3. 缺少状态变化信息
4. 缺少压缩→扩张和不对称结构这类更贴近 5m alpha 的信息
5. 训练目标可能过于噪音，需要配合质量过滤或样本加权

---

## 下一版优化原则

### 1. 不优先继续堆普通 TA 指标
不建议优先加入大量：
- RSI
- MACD
- Stoch
- KDJ
- CCI

原因：
- 这些大多与已有 momentum / volatility / path 特征高度相关
- 新增信息量可能有限
- 会增加复杂度而不一定显著提升可交易性

### 2. 优先加入“当前缺失的信息维度”
下一版优先新增的是：

- 高时间框架上下文
- 状态变化（lag / delta）
- 压缩—突破结构
- 波动不对称 / 路径不对称
- 更强的 flow proxy

### 3. 所有新增内容必须模块化
每类新特征都必须新建独立 feature pack 文件，不能直接塞进已有大函数。

---

## 优先级总览

### 第一梯队（最优先）
1. `htf_context.py`
2. 核心特征的选择性 lag
3. `compression_breakout.py`

### 第二梯队
4. `asymmetry.py`
5. `flow_proxy.py`

### 第三梯队
6. 相关市场上下文（如 ETH/USDT）
7. 样本加权 / 样本过滤策略联动

---

# Part A. 高时间框架上下文（最高优先级）

## 目标
当前大量特征仅基于 1m 数据快照。  
但 5m 方向往往依赖于更慢背景：

- 当前是否处于更大级别趋势中
- 当前是否处于高压缩状态
- 当前 1m 动量发生在什么环境里

因此下一版应优先引入 5m / 15m context。

## 新文件建议

`src/features/htf_context.py`

## 建议加入的特征

### 5m 级别
- `htf_ret_5m_1`
- `htf_rv_5m`
- `htf_range_pos_5m`
- `htf_close_z_5m`
- `htf_efficiency_5m`

### 15m 级别
- `htf_ret_15m_1`
- `htf_rv_15m`
- `htf_range_pos_15m`
- `htf_close_z_15m`
- `htf_efficiency_15m`
- `htf_regime_trend_strength_15m`

## 设计说明

这些特征的作用不是替代 1m 特征，而是为 1m 信号提供背景条件。

例如：
- 同样的 `ret_3 > 0`
- 在 15m 强趋势中，可能意味着顺势延续
- 在 15m 区间顶部，可能意味着短期反抽后回落

## 实现要求

- 必须通过统一 builder 注入
- 必须从统一配置中声明启用的 HTF
- 不允许单独在 strategy 或 inference 逻辑里额外拼接

---

# Part B. 核心特征的选择性 Lag（高优先级）

## 目标
当前特征更多是“当前状态快照”，但 5m alpha 很可能来自“状态变化”。

例如：
- 波动是否刚刚抬升
- volume 是否连续放大
- market_quality 是否刚从低质量切换到高质量

因此建议对少量高价值特征做滞后扩展，而不是对全部特征盲目扩维。

## 推荐做 lag 的特征

### 1. 动量类
- `ret_1`
- `ret_3`

### 2. 波动类
- `rv_5`

### 3. 流量/活跃度类
- `signed_volume_1`
- `nz_volume_share_5`
- `flat_share_5`

### 4. candle / path 类
- `body_pct_1`
- `close_location_1`

## 推荐 lag 范围
- `lag1`
- `lag2`
- `lag3`

示例：
- `ret_1_lag1`
- `ret_1_lag2`
- `ret_1_lag3`

## 为什么只做选择性 lag
因为：
- 全特征 lag 会大幅增加维度
- 很多 lag 只是重复同类信息
- 训练复杂度和过拟合风险会上升

## 实现建议

可以新增：

`src/features/lagged.py`

或者在 `builder.py` 中对指定 feature name 列表做统一 lag 扩展。

建议配置化：

```yaml
features:
  lagged_features:
    names:
      - ret_1
      - ret_3
      - rv_5
      - signed_volume_1
      - nz_volume_share_5
      - flat_share_5
      - body_pct_1
      - close_location_1
    lags: [1, 2, 3]
```

---

# Part C. Compression / Breakout Pack（高优先级）

## 目标
5m 方向经常不是来自平稳延续，而是来自：

- 压缩之后的扩张
- 方向选择临界点
- 波动收缩后突然释放

当前已有 range / efficiency / regime，但还缺少更直接的“压缩→突破”特征。

## 新文件建议

`src/features/compression_breakout.py`

## 建议加入的特征

### 压缩宽度类
- `bb_width_20`
- `bb_width_pct_rank_100`
- `donchian_width_20`
- `atr_ratio_5_20`

### 结构标记类
- `nr4_flag`
- `nr7_flag`

### 突破距离类
- `breakout_up_dist_20`
- `breakout_down_dist_20`
- `compression_score`

## 特征含义说明

### `bb_width_20`
衡量 Bollinger 带宽，越窄越说明压缩。

### `bb_width_pct_rank_100`
当前压缩程度在过去一段时间中的分位。

### `nr4_flag` / `nr7_flag`
判断当前是否为近 4 / 7 根中最窄区间。

### `breakout_up_dist_20`
当前价格距离近期上沿还有多远。

### `compression_score`
可组合以下元素：
- band width
- ATR ratio
- donchian width
- recent narrow range status

形成一个统一压缩评分。

## 为什么这类特征重要
它们更贴近“方向即将形成”的状态，而不只是描述当前 bar 发生了什么。

---

# Part D. Asymmetry Pack（第二梯队）

## 目标
当前有 realized volatility，但仍偏总量描述。  
下一版应加入波动和路径的不对称性。

## 新文件建议

`src/features/asymmetry.py`

## 建议加入的特征

### 波动不对称
- `upside_rv_5`
- `downside_rv_5`
- `upside_rv_20`
- `downside_rv_20`

### 偏度 / 分布不对称
- `realized_skew_10`
- `realized_skew_20`

### wick 不对称
- `wick_imbalance_3`
- `wick_imbalance_5`

### body 不对称
- `body_imbalance_3`
- `body_imbalance_5`

## 作用说明
很多时候：
- 下行波动大但价格跌不下去
- 上影线很重但 close 还站得住
- 这些都可能对应短期方向 edge

这类信息通常比单纯 `rv_5` 更有用。

---

# Part E. Flow Proxy Pack（第二梯队）

## 目标
当前 volume 特征偏统计量，缺少更接近 order flow 的代理。

由于当前系统先不引入逐笔或 orderbook 级别数据，所以需要基于 OHLCV 构造更强的 flow proxy。

## 新文件建议

`src/features/flow_proxy.py`

## 建议加入的特征

### 基础流向代理
- `clv_1`
- `clv_x_volume_1`
- `clv_x_dollar_volume_1`

### 压力代理
- `wick_pressure_1`
- `wick_pressure_x_volume_1`

### 有符号成交额
- `signed_dollar_volume_1`
- `signed_dollar_volume_3`

### 扩张配合类
- `range_expansion_x_volume_3`
- `body_x_volume_3`

## 说明
单独 volume 往往不够，必须结合：
- close 在 bar 中的位置
- body 大小
- wick 方向
- 成交额强弱

才更接近“真实的买卖压力”。

---

# Part F. 第三梯队：相关市场上下文

## 目标
用一个相关资产作为额外风险偏好 / 情绪代理。

## 建议方向
第一版若要加相关标的，只建议尝试：

- `ETH/USDT`

不要一口气引入很多 alt。

## 可加入的特征
只需少量：
- `eth_ret_1`
- `eth_ret_5`
- `eth_rv_5`
- `eth_regime_trend_strength`

## 注意
这部分优先级低于：
- HTF
- lag
- compression
- asymmetry

只有当前四类补充后仍无改善，再考虑引入相关标的。

---

# Part G. 与样本过滤 / 样本加权联动（非常重要）

## 目标
不要只把 `market_quality` 当普通特征，还应考虑把它用于：

- 样本过滤
- 样本权重
- 线上 gating

## 当前已有特征
- `nz_volume_share_*`
- `flat_share_*`
- `abs_ret_mean_*`
- `dollar_vol_mean_*`

这些已经非常适合做质量控制。

## 建议方向

### 训练阶段
尝试以下两种方案：

#### 方案 1：样本过滤
过滤掉低质量窗口，例如：
- `nz_volume_share_20 < threshold`
- `flat_share_20 > threshold`

#### 方案 2：样本加权
质量越高，训练权重越高。

例如：
- `sample_weight = f(nz_volume_share_20, abs_ret_mean_20, dollar_vol_mean_20)`

### 线上阶段
可作为额外 gating：
- 活跃度不足，不交易
- 平价状态太高，不交易

## 为什么这步很重要
你当前结果已经表明：
- 阈值放松 → 亏损
- 阈值收紧 → 0 交易

这通常意味着：
- 模型本身边际有限
- 但某些更高质量窗口也许仍有 edge

因此“只在高质量环境交易”很可能比“全样本继续堆特征”更有效。

---

# Part H. 不建议当前优先做的事

## 1. 不建议继续大量补传统 TA 指标
原因：
- 信息重叠大
- 可维护性变差
- 不一定提高边际信号强度

## 2. 不建议全量 lag 所有特征
原因：
- 维度暴涨
- 噪声增加
- 容易过拟合

## 3. 不建议先切换更重模型
原因：
- 若当前缺的是正交信息，更复杂模型只会更快拟合噪音

## 4. 不建议先优先优化 execution
当前结果更像是：
- 输入信息还不够强
- 或任务需要更强 abstention/filter

因此下一步重点仍应放在：
- feature
- sample quality
- signal policy

---

# Part I. 推荐实施顺序

## Phase 1（最先做）
1. 新增 `htf_context.py`
2. 新增选择性 lag
3. 新增 `compression_breakout.py`

## Phase 2
4. 新增 `asymmetry.py`
5. 新增 `flow_proxy.py`

## Phase 3
6. 引入样本过滤 / 样本加权实验
7. 视情况引入 ETH/USDT 上下文

---

# Part J. 对 Codex / Agent 的明确要求

## 必须遵守
- 所有新增特征必须通过统一 builder 接入
- 不允许在线和离线各写一套特征逻辑
- 所有窗口长度、lag、feature pack 开关必须来自统一配置
- 所有新增 pack 必须有单元测试
- 所有新特征都必须只使用 `t0` 前已完成 candle
- 继续保持 leak-free

## 必须新增的测试
建议增加：
- `test_htf_context.py`
- `test_lagged_features.py`
- `test_compression_breakout.py`
- `test_asymmetry.py`
- `test_flow_proxy.py`

---

# Part K. 最终建议

当前阶段的重点不是把特征数从 63 机械加到 120，而是：

- 加入更强的背景上下文
- 加入状态变化信息
- 加入压缩→扩张特征
- 加入不对称结构
- 更聪明地决定什么时候不交易

一句话总结：

> 下一版应优先补“正交信息”和“高质量窗口过滤能力”，而不是继续堆同类指标。