# 20260612 CatBoost Continuation & Reversal Hybrid Router 实验报告

## 实验背景与目的
在之前的研究中，我们独立训练了两种高度特化的“专家模型”：
1. **顺势专家 (Continuation Expert)**: `20260611_catboost_continuation_side_coordinate_search`，该模型在特定的日历微观特征切片下只做与开盘第一分钟方向相同 (same_side) 的交易，胜率极高，但主要捕获趋势延续的利润。
2. **反转专家 (Reversal Expert)**: `20260611_catboost_reversal_sample_weight_search/reversal_weight_8`，该模型使用了 8.0 倍的样本惩罚权重，强制学习微观反转特征，且被硬性配置为只做与第一分钟方向相反 (opposite_side) 的反转开仓。

**混合路由实验 (Hybrid Router)** (`20260612_catboost_continuation_reversal_hybrid_router`) 的目的是：将上述两个在各自擅长领域表现极佳的底层专家模型结合起来，构建一个“智能路由器 (Router)”。当两者信号发生冲突，或者面临不同的市场环境时，由 Router 决定听从哪个专家的信号，从而试图在满足最低覆盖率约束 (Coverage >= 0.70) 的情况下，最大化全局选择分数 (Selection Score)。

在该框架下，代码内部执行了**两个核心子实验 (策略)**来对比不同的路由机制：`conflict_margin` 和 `gate_hybrid`。

---

## 实验一：Conflict Margin 路由策略 (Experiment 3)

### 策略原理
这是一种**无参数 (基于启发式规则)** 的路由策略。它不训练额外的神经网络，而是直接比较“顺势专家”和“反转专家”各自给出的概率置信度。
*   当两位专家同时给出相同的开仓方向时，直接采纳。
*   当两者的信号或倾向发生**冲突**时，计算它们各自预测概率偏离 0.5 的绝对距离（即各自的置信度强度）。如果两者的置信度差距超过了设定的阈值 `conflict_margin`，则 Router 会判定置信度更强的那位专家胜出并听从其方向；否则，为规避高风险，Router 选择 `ABSTAIN` (放弃交易)。
*   实验网格搜索了多个 Margin 阈值：`[0.03, 0.05, 0.07, 0.10, 0.12, 0.15]`。

### 实验结果 (`experiment3_conflict_margin`)
*   **最佳 Margin 参数**: `0.15`
*   **覆盖率 (Coverage)**: `69.95%` (略微不及 `>= 70%` 的硬约束)。
*   **总体胜率 (Accepted Sample Accuracy)**: `47.79%` (表现极差，跌破 50% 抛硬币胜率)。
*   **路由分数 (Router Score)**: `-0.42`
*   **结论**: 这种基于静态置信度绝对值比较的方法表现非常糟糕，严重破坏了原有专家的边界，导致整体收益跌负。

---

## 实验二：Gate Model 门控混合路由策略 (Experiment 4)

### 策略原理
这是一种**基于机器学习的元学习 (Meta-Learning)** 路由机制。
*   额外训练了一个小型的 CatBoost 分类器 (`gate_model`, depth=4)。
*   **特征集扩大**: 相比于底层专家，Gate 模型加入了如 `momentum`, `acceleration`, `range`, `liquidity`, `volume`, `session` 等更宏观/中观的环境特征，意在判断当前市场属于什么 Regime。
*   **目标函数**: Gate 模型的目标是预测当前样本是顺势走势 (`p_cont`) 还是反转走势。
*   **双独立阈值门控**: 根据 Gate 模型输出的概率 `p_cont`，使用 `tau_cont` 和 `tau_rev` 两个动态阈值来做路由分发：
    *   如果 `p_cont >= tau_cont`，则认为当前大概率是顺势环境，路由器屏蔽反转专家，全权听从顺势专家的开仓决定。
    *   如果 `p_cont <= 1 - tau_rev`，则认为极大概率是反转环境，路由器听从反转专家的决定。

### 实验结果 (`experiment4_gate_hybrid`)
*   **最佳阈值组合**: `tau_cont = 0.65`, `tau_rev = 0.65`。
*   **覆盖率 (Coverage)**: `38.40%`。
*   **总体胜率 (Accepted Sample Accuracy)**: `75.45%`。
*   **顺势专家的表现**: `continuation_accepted_accuracy` 高达 `100%` (顺势专家的胜率潜能在门控保护下被完全释放)。
*   **反转专家的表现**: `reversal_accepted_accuracy` 跌至 `0.0%` (门控模型由于未能精确识别反转环境，在放行的反转样本上全军覆没)。
*   **结论**: 门控网络 (`gate_hybrid`) 相比于 `conflict_margin` 大幅提高了开仓胜率，但它是以**严重牺牲覆盖率**为代价的。它的实际覆盖率仅为 38.4%，远不及业务设定的 70% 门槛要求。且从 Router Score (-0.28) 来看，叠加 Router 后并没有达到 1+1 > 2 的系统性优化效果，反而让反转模型彻底失效。

---

## 最终总结

根据 `report.json` 的整体评判，整个 `20260612_catboost_continuation_reversal_hybrid_router` 实验被定性为 **未能超越基线 (`improved_over_accepted_baseline: false`)**。

虽然底层的“顺势专家”和“反转专家”在各自独立的评估报告中表现惊艳，但我们通过实验证明：**尝试在它们之上叠加一层硬性路由器（无论是基于置信度Margin比较，还是使用引入宏观特征的分类器进行硬路由切流）都无法有效地调和两者的冲突**。

后续策略迭代可能需要：
1. 放弃在最后一步做硬性（Hard Routing）切割决策。
2. 转而使用特征层面的软融合（Soft Ensemble），例如将专家模型输出的预测概率、隐藏层状态等作为基础特征，连同宏观状态特征一起喂给一个全局排序模型，让模型自然地学习如何平衡这两种市场形态。