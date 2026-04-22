# BTC/USDT 方向模型准确率诊断与优化方案（基于现有 artifacts 深入分析）

日期：`2026-04-11`

---

## 1. 结论先行

当前结果差，不是单一原因，也不是“再多跑几组 LightGBM 参数”能解决的问题。

从 `artifacts/models` 里的现有实验看，问题主要来自 4 层：

1. **当前 5m 目标本身接近噪声上限**
2. **评估设计过弱，模型选择建立在单一验证尾窗上**
3. **特征数持续增加，但新增信息密度没有同步提高，导致树模型稳定过拟合**
4. **衍生品特征的有效性强依赖窗口与市场状态，短窗有效不代表长窗可泛化**

如果目标是**提高预测准确率**，优先级不应该继续放在“再追加更多相似特征”上，而应该先做：

- 重构评估基线
- 重构标签/样本定义
- 收缩并重建特征集
- 把衍生品信号改成“有条件启用”的 regime/context 信号，而不是默认全量堆入

---

## 2. 现有结果的核心事实

以下结论直接来自当前保留在仓库中的 artifacts。

### 2.1 长历史 5m 主基线几乎贴近随机

参考：

- [artifacts/models/manual/btc_usdt_5m_recent30d/training_report.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\manual\btc_usdt_5m_recent30d\training_report.json)
- [artifacts/models/experiments/public_archive_light_2026-01-01_2026-04-10/summary.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\public_archive_light_2026-01-01_2026-04-10\summary.json)

关键数字：

- `5m baseline / lightgbm / validation roc_auc = 0.531100`
- `validation log_loss = 0.691730`
- `raw_validation log_loss = 0.691390`
- `validation accuracy = 0.518257`
- `validation positive_rate = 0.502486`

解释：

- `log_loss` 已经非常接近随机二分类的 `0.693`
- `positive_rate` 接近 `0.50`
- `accuracy` 仅比 coin-flip 高很少

这说明当前任务在现有标签定义下，**可学习信号非常弱**。

### 2.2 树模型学得到训练集，但泛化明显不够

参考：

- [docs/model_experiment_log.md](C:\Users\ROG\Desktop\crypto_engine\docs\model_experiment_log.md)
- [artifacts/models/experiments/model_family_baseline](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\model_family_baseline)

当前保留排名：

- `catboost`: valid AUC `0.539404`
- `logistic`: valid AUC `0.535921`
- `lightgbm`: valid AUC `0.531100`

同时：

- `lightgbm train AUC = 0.612795`, `valid AUC = 0.531100`
- `catboost train AUC = 0.584590`, `valid AUC = 0.539404`
- `logistic train AUC = 0.530948`, `valid AUC = 0.535921`

解释：

- `LightGBM` 明显在记忆训练集结构
- `CatBoost` 泛化稍好，但提升也有限
- `Logistic` 虽然 ceiling 低，但说明**线性层面也只有很弱的稳定信号**

换句话说：当前不是“模型太弱”，而是**信号弱且不稳定**。

### 2.3 15m 比 5m 更容易预测

参考：

- [artifacts/models/horizon_compare/5m/training_report.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\horizon_compare\5m\training_report.json)
- [artifacts/models/horizon_compare/15m_train_report_v2/training_report.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\horizon_compare\15m_train_report_v2\training_report.json)

关键数字：

- `5m valid AUC = 0.531100`
- `15m valid AUC = 0.539170`

解释：

- 同样的建模框架下，`15m` 明显比 `5m` 更容易学到方向性
- 这非常强地暗示：**5m close[t+5m] > open[t] 这个目标过于接近短期噪声**

这不是说 `15m` 一定足够好，而是说明：

> 当前主要瓶颈更像是“目标定义的信息密度不足”，不是“模型族不够高级”。

### 2.4 衍生品在短窗里偶尔有帮助，但长窗上不能稳定迁移

短窗结果参考：

- [artifacts/models/experiments/derivatives_ablation_2026-03-10_2026-04-08/summary.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\derivatives_ablation_2026-03-10_2026-04-08\summary.json)

长窗 public archive 结果参考：

- [artifacts/models/experiments/public_archive_light_2026-01-01_2026-04-10/summary.json](C:\Users\ROG\Desktop\crypto_engine\artifacts\models\experiments\public_archive_light_2026-01-01_2026-04-10\summary.json)

短窗最佳：

- `funding_basis_oi_options / lightgbm / valid AUC = 0.547549`
- 但 `train AUC = 0.922087`
- `overfit_gap_roc_auc = 0.374538`

长窗 public archive：

- `baseline = 0.531100`
- `funding = 0.512095`
- `funding_basis = 0.515260`
- `funding_basis_oi = 0.516376`
- `funding_basis_oi_options = 0.518796`

解释：

- 衍生品特征**不是完全没用**
- 但它们现在更像是**短期 regime-specific 信号**
- 一旦放回更长时间窗，它们没有稳定泛化，反而引入更严重过拟合

这说明当前衍生品层存在问题：

- 信号窗口太短
- 对齐方式可能没问题，但**统计稳定性不够**
- 特征是“能解释某些局部阶段”，不是“默认适用于整段历史”

### 2.5 当前 walk-forward 实际上只有 1 个 fold

多个 training report 中都显示：

- `walk_forward_summary.fold_count = 1`

这意味着当前“walk-forward”在事实上并没有形成稳健的多折时序验证，而只是：

- 一个 development 段
- 一个验证尾窗

这会带来两个后果：

1. 模型选择很容易被单一市场阶段误导
2. 任何“提升 0.005 到 0.01 AUC”的结果都可能是窗口偶然性

所以当前实验多，不代表证据强。**样本数多，证据密度不够。**

---

## 3. 为什么结果差：分层原因分析

## 3.1 第一层原因：目标定义本身太难

当前标签：

```text
y = 1{close[t0+5m] > open[t0]}
```

这个标签的问题不是逻辑错误，而是**统计意义上太“薄”**：

- 只看 `5m`
- 只看方向
- 不设最小波动阈值
- 把几乎没有可交易意义的微小涨跌，也强制当成正负标签

结果就是：

- 大量样本其实是“接近 0 的小波动”
- 对模型来说，这些样本和真正有趋势的样本混在一起
- 方向标签被大量“低信噪比样本”稀释

这会直接拉低可达到的准确率上限。

### 影响

- AUC 容易被压在 `0.53x`
- logloss 很难明显拉开
- 模型只能学到少量条件性 edge

---

## 3.2 第二层原因：当前实验主要在优化模型，而不是优化问题定义

从现有实验看，投入最多的是：

- model family 对比
- derivatives ablation
- 不同特征组合

但这些实验大都默认了：

- 同一个标签
- 同一个单尾窗验证方案
- 同一套特征工程结构

这会导致一个典型问题：

> 在一个可能本身噪声很高的问题上，反复调模型和堆特征，收益会非常有限。

也就是说，目前优化方向更多是：

- “怎么拟合这个问题”

而不是：

- “这个问题是否值得这样定义”

---

## 3.3 第三层原因：特征数量增长快于独立信息增长

当前 baseline 已经有 `126` 个特征。

引入 derivatives 后：

- `funding`: `136`
- `funding_basis`: `144`
- `funding_basis_oi`: `152`
- `funding_basis_oi_options`: `157`

问题不在于 157 个特征很多，而在于：

- 这些特征很多是同源变换
- 同一底层价格序列被多次窗口化、滞后化、归一化
- derivatives 信号在长窗内并不稳定

结果就是：

- 模型容量上升
- 训练集可分性上升
- 验证集不跟着上升
- overfit gap 急剧扩大

最典型例子就是：

- `funding_basis_oi_options` 短窗 train AUC `0.922087`
- valid AUC 只到 `0.547549`

这是明显的“模型学到了大量局部结构，但这些结构不能稳健迁移”。

---

## 3.4 第四层原因：评估方案不足以支持可靠模型选择

当前 `fold_count = 1` 是一个核心问题。

这意味着：

- 你看到的最佳模型，很可能只是某一个尾窗最合适
- 你看到的衍生品增益，也可能只是某一个 regime 的偶然收益

这会直接导致两个偏差：

1. **过早保留无效特征**
2. **过早淘汰潜在有效特征**

尤其在 crypto 这种 regime change 很快的环境里，单一尾窗验证非常脆弱。

---

## 3.5 第五层原因：Calibration 和 ranking 目标混在了一起

从现有 report 看，`raw_validation_metrics` 在多个实验里并不比 calibration 后更差，甚至更好：

- 5m baseline:
  - raw valid AUC `0.532295`
  - calibrated valid AUC `0.531100`
  - raw logloss `0.691390`
  - calibrated logloss `0.691730`
- 15m:
  - raw valid AUC `0.539129`
  - calibrated valid AUC `0.539170`
  - raw logloss `0.692409`
  - calibrated logloss `0.691566`

说明：

- calibration 不是当前主要增益来源
- 有时它改善概率形状，但并不提升排序能力
- 如果目标是先提高“预测准确率”或“方向排序能力”，calibration 不应成为主优化轴

---

## 3.6 第六层原因：样本质量过滤和加权策略缺少反证实验

当前配置里默认启用了：

- `sample_quality_filter.enabled = true`
- `sample_weighting.enabled = true`

但现有 artifacts 里，没有形成清晰证据表明：

- 这些过滤确实提高了 out-of-sample AUC
- 这些权重确实优于不加权

而 training report 里样本权重大致集中在：

- min `1.51`
- mean `1.89`
- max `2.15`

这说明：

- 权重变化范围不算特别大
- 但它仍然在持续改变训练分布

如果没有系统对照实验，就有可能出现：

- 过滤掉了最难但最重要的样本
- 权重把模型推向某些容易学但泛化差的结构

---

## 4. 优化方向：按优先级排序

下面分成两类：

- **A 类：不改主标签前提下的优化**
- **B 类：为了显著提高准确率，建议评估的更深层改动**

如果坚持当前 V1 约束，先做 A 类。
如果目标明确是“显著提高预测准确率”，B 类最终很可能不可回避。

---

## 5. A 类优化：不改当前主标签的前提下

## 5.1 把评估从“单尾窗”改成“多时段 walk-forward”

### 当前问题

- `fold_count = 1`
- 任何结论都太依赖 `2026-03-09` 到 `2026-04-08` 这种单段尾窗

### 建议

建立固定 benchmark：

1. 使用至少 `4` 到 `8` 个时间连续的 walk-forward folds
2. 每个 fold 保持严格时间顺序和 purge
3. 排名时优先看：
   - `roc_auc_mean`
   - `roc_auc_std`
   - `log_loss_mean`
   - `delta_vs_baseline_mean`

### 预期收益

- 先提升“结论可信度”
- 再提升“模型选择质量”

### 优先级

- `P0`

---

## 5.2 先把 CatBoost 作为主 baseline，不要默认以 LightGBM 为中心

### 事实

当前 retained baseline 里：

- `catboost valid AUC = 0.539404`
- `logistic valid AUC = 0.535921`
- `lightgbm valid AUC = 0.531100`

### 结论

目前最强的树模型 baseline 不是 `lightgbm`，而是 `catboost`。

### 建议

- 后续实验的默认对照基线改成：
  - `logistic`
  - `catboost`
- `lightgbm` 保留，但不再默认作为唯一主力

### 原因

- `CatBoost` 当前在这个任务上更稳
- `LightGBM` 更容易把特征堆叠变成训练集记忆

### 优先级

- `P0`

---

## 5.3 先做“减法实验”，不要再直接加 pack

### 当前问题

现有路线主要是：

- baseline
- 加 funding
- 加 basis
- 加 oi
- 加 options

但没有系统回答：

- 哪些 core feature pack 本身在拖后腿

### 建议

对 baseline 先做核心 pack ablation：

1. `baseline - time`
2. `baseline - lagged`
3. `baseline - asymmetry`
4. `baseline - compression_breakout`
5. `baseline - htf_context`
6. `baseline - flow_proxy`
7. `baseline - regime`

目标不是一次删光，而是找出：

- 稳定贡献 pack
- 不稳定但偶尔有效 pack
- 长期有害 pack

### 预期收益

- 降低特征冗余
- 降低树模型过拟合自由度
- 为后续 derivatives 留出“真实增量空间”

### 优先级

- `P0`

---

## 5.4 对 sample quality filter 和 sample weighting 做严格对照

### 当前问题

过滤和加权已默认启用，但缺少充分证据证明它们提高泛化。

### 建议

至少做 4 组对照：

1. `no_filter + no_weight`
2. `filter + no_weight`
3. `no_filter + weight`
4. `filter + weight`

记录：

- mean AUC
- std AUC
- valid logloss
- 训练集行数变化
- 各月验证表现

### 预期收益

- 判断当前“质量样本偏好”是否真的有利
- 避免模型过度偏向高波动、高成交量 regime

### 优先级

- `P0`

---

## 5.5 把 calibration 从主排名中剥离

### 当前问题

现在 calibration 和模型质量有时混在一起解读。

### 建议

实验汇总中同时输出两套：

- `raw_validation_metrics`
- `calibrated_validation_metrics`

排序优先建议：

1. `raw roc_auc`
2. `raw log_loss`
3. `calibrated log_loss`

### 原因

- 提高准确率首先是 ranking 问题
- calibration 更多是概率形状问题

### 优先级

- `P1`

---

## 5.6 衍生品特征改成“条件启用”，而不是默认累加

### 当前事实

短窗里：

- `funding_basis_oi_options` 有提升

长窗里：

- 所有 derivatives 组合都输给 baseline

### 结论

衍生品信号现在更像：

- **有条件有效**
- 而不是**全局稳定 alpha**

### 建议

对 derivatives 做 regime gating：

1. 只在高波动状态启用某些 derivatives pack
2. 只在 funding/basis 偏离超过阈值时启用对应子模型
3. options 只作为 regime 开关，不直接全量进入主模型

### 预期收益

- 降低全局过拟合
- 保留局部 regime 下的真实增量

### 优先级

- `P1`

---

## 6. B 类优化：如果目标是显著提升准确率，建议重新定义样本问题

这部分超出“当前主标签不改”的范围，但如果目标是明显提升准确率，基本绕不开。

## 6.1 对 5m 标签引入最小收益阈值，去掉模糊样本

### 当前问题

`close[t+5m] > open[t]` 把极小涨跌也作为硬标签。

### 建议

改成三段式样本处理：

- `future_return > +thr` → 正样本
- `future_return < -thr` → 负样本
- `abs(future_return) <= thr` → 丢弃或设为中性类

### 预期收益

- 提高标签信噪比
- 显著提升 AUC 上限

### 风险

- 样本数减少
- 与当前 V1 固定标签规则不一致

### 优先级

- `P1`

---

## 6.2 把 15m 作为主 benchmark，而不是只作为对照

### 事实

- `15m valid AUC = 0.539170`
- `5m valid AUC = 0.531100`

### 结论

15m 已经显示出更强可预测性。

### 建议

- 保留 5m 作为原目标
- 同时把 15m 升级为正式 benchmark
- 后续优先在 15m 验证新特征、新评估方案是否真的有价值

### 原因

- 如果一个改动在 15m 上都没有提升，通常很难期待它在更噪声的 5m 上带来稳定提升

### 优先级

- `P0`

---

## 6.3 把“是否交易”与“方向”拆成两阶段问题

### 当前问题

目前模型被迫对所有 grid 点输出方向判断。

### 建议

改为两阶段：

1. 第一阶段判断：这一个 `t0` 是否值得预测/交易
2. 第二阶段判断：如果值得，再判断方向

即典型的：

- `meta-label / tradeability filter`

### 预期收益

- 把最难、最模糊的样本挡在模型外
- 提高留下样本的方向准确率

### 优先级

- `P1`

---

## 7. 不建议继续优先做的方向

下面这些方向当前不应作为主轴：

### 7.1 继续只调 LightGBM 超参数

原因：

- 现有问题不是简单的 bias 太大
- 当前更像 variance 和任务定义问题

### 7.2 不做评估重构，继续堆新的 derivatives pack

原因：

- 现有证据已经显示长窗泛化不稳定

### 7.3 把 calibration 当成提准确率主手段

原因：

- calibration 主要影响概率质量，不解决可分性不足

### 7.4 只看单次尾窗 AUC 排名

原因：

- 会持续放大 regime 偶然性

---

## 8. 建议的实验路线图

## Phase 1：先修“证据质量”

目标：把实验结论从“单窗口偶然结果”提升为“多窗口稳健证据”。

建议顺序：

1. 实现多 fold walk-forward
2. 输出 mean/std/max drawdown-style 的验证汇总
3. 固定三条 benchmark：
   - `5m + logistic`
   - `5m + catboost`
   - `15m + catboost`

成功标准：

- 同一模型在多 fold 上结果稳定可解释

## Phase 2：做 baseline 减法

目标：找出真正稳定贡献的 core packs。

建议顺序：

1. baseline pack ablation
2. filter/weighting 对照
3. raw vs calibrated 评分拆开

成功标准：

- 形成一套“更小、更稳”的 baseline feature set

## Phase 3：重新定义 derivatives 的使用方式

目标：让 derivatives 从“全局堆料”改成“条件增益”。

建议顺序：

1. regime gating
2. derivatives only in selected states
3. 先验证 15m，再验证 5m

成功标准：

- 衍生品加入后，mean AUC 提升且 std 不显著恶化

## Phase 4：如果还要显著提升，再动标签

目标：提升问题本身的信噪比。

建议顺序：

1. thresholded label
2. neutral class / drop ambiguous samples
3. meta-label tradeability stage

成功标准：

- retained samples 上的方向准确率明显提升

---

## 9. 最可能带来真实提升的前三项

如果只能做 3 件事，我建议按这个顺序：

1. **把评估改成多 fold walk-forward**
2. **把 15m 提升为正式 benchmark，并对 baseline packs 做减法**
3. **对 5m 标签引入阈值或中性样本机制**

理由：

- 第 1 项提升结论可信度
- 第 2 项提升 baseline 质量
- 第 3 项才是真正提升“准确率上限”的关键杠杆

---

## 10. 一句话总结

当前结果差，核心不是“模型还不够复杂”，而是：

> **5m 目标噪声太高，当前验证设计又太弱，导致新增特征主要在放大训练集可分性，而没有形成稳定可泛化的预测能力。**

要提高准确率，最优先的不是继续堆特征，而是：

> **先重建评估，再收缩特征，再重新定义哪些样本值得预测。**

