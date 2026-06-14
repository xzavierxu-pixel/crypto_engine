# Polymarket BTC 5m Price Estimator：CatBoost 分位数回归 Baseline

## 1. 实验目标

在已有 BTC 5 分钟 UP/DOWN 方向预测模型基础上，建立一个独立的 `price_estimator` 模块，用于预测方向模型选中 side 的未来 4 分钟最低成交价分位数。

本实验只做一个 baseline：

```text
使用所有已有 feature pack
训练 CatBoost Quantile Regression
输出 Q70 / Q80 / Q90
选择分位数预测最准确的回归模型作为 price estimator baseline
```

本阶段不接入实盘下单，不做 EV 优化，不做多模型路线对比。

---

## 2. 建模假设

已有方向模型负责判断买 UP 还是 DOWN。

价格模型只回答：

```text
如果方向模型选中的 side 最终 resolve 成功，
这个赢家 token 在剩余 4 分钟里最低能成交到什么价格？
```

因此训练样本只保留：

```text
selected_win = True
```

即：

```text
如果 selected_side = UP，只保留最终 UP resolve 成功的样本
如果 selected_side = DOWN，只保留最终 DOWN resolve 成功的样本
```

---

## 3. 样本与目录要求

所有实现、配置、数据处理脚本、训练结果和评估报告都放在：

```text
price_estimator/
```

建议结构：

```text
price_estimator/
  configs/
  data/
  scripts/
  models/
  reports/
  README.md
```

价格模型不重新构建 feature，只读取现有 feature pack 文件。

时间切分必须和当前方向预测模型保持一致，包括：

```text
train / validation / test split
market_id 范围
时间窗口
样本过滤逻辑
```

---

## 4. selected side 定义

对每个样本，先使用方向模型输出 `p_up` 决定 selected side：

```python
if p_up >= 0.5:
    selected_side = "UP"
    selected_win = outcome_up == 1
else:
    selected_side = "DOWN"
    selected_win = outcome_up == 0
```

买入方向必须作为模型特征：

```text
selected_side
```

也可以保留：

```text
p_up
p_side = max(p_up, 1 - p_up)
direction_confidence = abs(p_up - 0.5)
p_bin
```

---

## 5. Target 定义

目标变量使用：

```text
lowest_trade_price_next4
```

定义为：

```text
在决策时点之后、market close 之前的未来 4 分钟内，
selected_side 对应 token 的所有 trade 中，
满足 side = sell 且 takerOnly = true 的成交记录里的最低 price。
```

即：

```text
target_raw = lowest_trade_price_next4
target = logit(target_raw)
```

logit transform：

```python
eps = 0.01
price_clip = np.clip(lowest_trade_price_next4, eps, 1 - eps)
target = np.log(price_clip / (1 - price_clip))
```

预测后反变换：

```python
pred_price = 1 / (1 + np.exp(-pred_logit))
```

最终输出：

```text
pred_q70
pred_q80
pred_q90
```

---

## 6. 最低价出现时间处理

生成并保存最低价出现时间字段：

```text
lowest_trade_time_next4
time_to_lowest_trade_sec
```

含义：

```text
lowest_trade_time_next4: 未来 4 分钟内最低成交价第一次出现的时间
time_to_lowest_trade_sec: 从决策时点到最低价出现的秒数
```

重要规则：

```text
time_to_lowest_trade_sec 属于未来路径信息。
默认不能作为线上可用 price model 输入特征，否则会产生数据泄漏。
```

本实验中它的用途是：

```text
1. 作为 target 相关字段保存
2. 用于分层评估，例如最低价出现在早段/中段/尾段时模型是否稳定
3. 可选做 oracle upper-bound 对照，但不能作为 baseline 正式输入特征
```

如果强行把 `time_to_lowest_trade_sec` 作为输入，实验报告必须明确标记为：

```text
oracle / leakage diagnostic only
```

不得作为可部署 baseline。

---

## 7. Feature 使用原则

不单独构建新特征。

模型输入直接使用现有所有 feature pack：

```text
all_existing_feature_packs
```

额外允许加入的非原始 feature 字段：

```text
selected_side
p_up
p_side
direction_confidence
p_bin
```

禁止加入：

```text
future return
future price path
future trade path
lowest_trade_price_next4
lowest_trade_time_next4
time_to_lowest_trade_sec
resolve 后 token price
```

---

## 8. 模型方案

只使用 CatBoost。

分别训练三个分位数模型：

```text
Q70: CatBoostRegressor(loss_function="Quantile:alpha=0.70")
Q80: CatBoostRegressor(loss_function="Quantile:alpha=0.80")
Q90: CatBoostRegressor(loss_function="Quantile:alpha=0.90")
```

推荐基础参数：

```python
CatBoostRegressor(
    loss_function="Quantile:alpha={alpha}",
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,
    random_seed=42,
    eval_metric="Quantile:alpha={alpha}",
    od_type="Iter",
    od_wait=100,
    verbose=100
)
```

分类特征至少包括：

```text
selected_side
p_bin
```

如 feature pack 中已有其他 categorical columns，应一并传入 CatBoost 的 `cat_features`。

---

## 9. 评估指标

核心评估目标不是方向准确率，而是分位数是否校准准确。

### 9.1 Coverage

```text
coverage_q70 = mean(y_true <= pred_q70)
coverage_q80 = mean(y_true <= pred_q80)
coverage_q90 = mean(y_true <= pred_q90)
```

理想结果：

```text
Q70 coverage ≈ 70%
Q80 coverage ≈ 80%
Q90 coverage ≈ 90%
```

### 9.2 Pinball Loss

每个分位数分别计算 pinball loss：

```text
pinball_q70
pinball_q80
pinball_q90
```

越低越好。

### 9.3 Sharpness

在 coverage 合格的前提下，预测价格越低越好：

```text
mean(pred_q70)
mean(pred_q80)
mean(pred_q90)
```

### 9.4 分层评估

至少按以下维度分层：

```text
selected_side
p_bin
p_side bucket
time_to_lowest_trade_sec bucket
market time bucket
```

其中 `time_to_lowest_trade_sec bucket` 只用于评估，不用于 baseline 输入。

---

## 10. Baseline 验收标准

第一版 baseline 至少输出：

```text
1. Q70 / Q80 / Q90 在 validation 和 test 上的 coverage
2. Q70 / Q80 / Q90 的 pinball loss
3. Q70 <= Q80 <= Q90 的 crossing rate
4. 按 selected_side 和 p_bin 的 conditional coverage
5. 按 time_to_lowest_trade_sec bucket 的 coverage stability
6. 模型 feature importance
7. 预测结果 parquet / csv
8. CatBoost model artifact
```

模型选择规则：

```text
优先选择 test coverage 最接近目标分位数的模型。
如果 coverage 接近，则选择 pinball loss 更低的模型。
如果 pinball loss 接近，则选择预测价格更低、sharpness 更好的模型。
```

---

## 11. 最简实现流程

```text
1. 在 price_estimator/ 下读取已有 feature pack 和方向模型 split。
2. 合并方向模型输出 p_up。
3. 生成 selected_side、selected_win、p_side、p_bin。
4. 只保留 selected_win = True 的样本。
5. 构造 target_raw = lowest_trade_price_next4。
6. target = logit(target_raw)。
7. 使用所有已有 feature pack + selected_side / p_side / p_bin 训练 CatBoost Quantile。
8. 分别训练 Q70 / Q80 / Q90。
9. 预测后 sigmoid 反变换回 0-1 价格。
10. 检查 Q70 <= Q80 <= Q90，如有 crossing，记录 crossing rate 并做后处理。
11. 输出 report、prediction file、model artifact。
```

---

## 12. 最终产物

本实验完成后，`price_estimator/` 应至少包含：

```text
price_estimator/
  configs/
    catboost_quantile_baseline.yaml

  scripts/
    build_price_target.py
    train_catboost_quantile.py
    evaluate_quantile_model.py

  models/
    catboost_q70.cbm
    catboost_q80.cbm
    catboost_q90.cbm

  reports/
    quantile_baseline_report.md
    quantile_metrics.json

  data/
    price_estimator_train.parquet
    price_estimator_valid.parquet
    price_estimator_test.parquet
    predictions_test.parquet
```

---

## 13. 一句话总结

本实验的唯一目标是：

```text
在 price_estimator/ 中，用现有全部 feature pack 和 CatBoost Quantile Regression，
基于 logit(lowest_trade_price_next4) 训练 Q70/Q80/Q90，
得到第一个可复用的赢家最低成交价预测 baseline。
```

# important
1.拉取数据的代码可以参考price_estimator\fetch_btc5m_sell_trades_from_refs.py，已有resolve的label：artifacts\data_v2\labels\polymarket_resolved，该文件涵盖了约 5 个月的历史数据：
   * 开始时间: 2025-12-18 04:25:00+00:00
   * 结束时间: 2026-05-10 23:50:00+00:00
2.从api获取的数据可能包括很多列，但是你只需要保存需要的内容比如price，timestamp，conditionId,slug等，尽量减少存储数据的体积，拉取完整的sell side takeronly=true的数据，按天存成parquet。尽量并行下载，可以用5个线程加快速度