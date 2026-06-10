# version3 市场价、买入价、预测概率与真实 EV 关系分析

生成时间：2026-06-10  
数据源：

- server：`version3:/home/ec2-user/fortune_bot`
- summary：`artifacts/logs/execution_engine/summaries`
- CLOB matched fills：`ClobClient.get_trades`
- 本地 JSON：`artifacts/reports/execution_engine/version3_last24h_resolved_analysis.json`

本报告继续使用真实 matched fills，不使用 submitted order 整单估算。

## 结论

亏损不能简单归因为“很多订单 `预测概率 - 买入价 < 0`”。真实情况更接近：

1. 最大亏损来自看起来 `p - buy_price` 很正的订单。
2. 这些订单通常挂得比市场 ask 低很多，静态看便宜，但只有市场反向时才容易成交。
3. 成交本身携带坏信息：如果预测方向错，订单更容易被打到；如果预测方向对，订单可能不成交。
4. 所以 EV 过滤应该估计：

```text
EV_submit = P(fill | order context)
          * (P(win | fill, order context) - E(fill_price | fill, order context))
```

而不是：

```text
EV_naive = model_probability - current_market_price
```

也不是单纯：

```text
EV_naive = model_probability - limit_buy_price
```

## 当前整体结果

| 指标 | 数值 |
|---|---:|
| submitted resolved orders | 147 |
| filled orders | 121 |
| order fill rate | 82.31% |
| size fill rate | 81.94% |
| matched fills | 152 |
| filled accuracy | 46.71% |
| filled avg price | 0.5489 |
| filled PnL | -49.993503 |
| ROI on filled cost | -12.49% |

平均成交价 0.5489 意味着成交后至少需要约 54.89% 的胜率才能打平；实际成交后胜率只有 46.71%。

## 按 naive edge 分桶

这里的 naive edge 定义为：

```text
naive_edge = model_confidence - limit_buy_price
```

| naive edge bucket | submitted | fill rate | filled acc | avg fill price | filled PnL |
|---|---:|---:|---:|---:|---:|
| `<-0.20` | 7 | 71.43% | 100.00% | 0.8200 | +5.400000 |
| `-0.20--0.10` | 9 | 100.00% | 77.78% | 0.8151 | -2.200000 |
| `-0.10-0.00` | 27 | 96.30% | 73.08% | 0.8000 | -11.738302 |
| `0.00-0.05` | 12 | 75.00% | 55.56% | 0.6678 | -6.060000 |
| `0.05-0.10` | 8 | 75.00% | 83.33% | 0.6059 | +9.370000 |
| `>=0.10` | 84 | 78.57% | 27.27% | 0.3775 | -44.765201 |

关键点：`>=0.10` 这个“静态看最有 edge”的桶，贡献了最大亏损 `-44.77`。它不是因为价格贵，而是因为成交后的条件胜率只有 27.27%。

这说明简单 `p - limit_price` 会把许多低价挂单判断为高 EV，但真实成交样本是被逆向选择过的。

## 按挂单价与当时 best ask 的距离

这里比较：

```text
limit_buy_price - best_ask_at_decision
```

| limit vs best ask | submitted | fill rate | filled acc | avg fill price | filled PnL |
|---|---:|---:|---:|---:|---:|
| `-0.05-0.00` | 71 | 90.14% | 70.31% | 0.7156 | -7.441543 |
| `-0.10--0.05` | 23 | 95.65% | 31.82% | 0.4440 | -18.607546 |
| `-0.20--0.10` | 53 | 66.04% | 20.00% | 0.3111 | -23.944414 |

越远离 best ask 的低价挂单，成交率下降，但成交后的胜率崩得更厉害。`-0.20--0.10` 虽然平均成交价只有 0.3111，但成交后胜率只有 20.00%，仍亏 `-23.94`。

这正是“便宜但容易在错的时候成交”的逆向选择。

## 按当时市场 best ask 分桶

| best ask bucket | submitted | fill rate | filled acc | avg fill price | filled PnL |
|---|---:|---:|---:|---:|---:|
| `<0.45` | 35 | 80.00% | 7.14% | 0.2187 | -25.387476 |
| `<0.55` | 21 | 66.67% | 35.71% | 0.4192 | -3.730000 |
| `<0.65` | 28 | 82.14% | 47.83% | 0.5119 | -4.105324 |
| `<0.75` | 21 | 90.48% | 57.89% | 0.6431 | -10.602401 |
| `<0.85` | 22 | 86.36% | 63.16% | 0.7968 | -19.040000 |
| `>=0.85` | 20 | 90.00% | 100.00% | 0.8766 | +12.871698 |

低 market price 不是优势。`best_ask < 0.45` 的市场里，系统买得很便宜，但成交后胜率只有 7.14%，亏损最大。

高 market price 也不是天然坏。`best_ask >= 0.85` 反而全胜，因为这些可能是市场和模型方向都高度一致的窗口。

## 为什么简单 EV 会误判

简单 EV 假设：

```text
成交是随机抽样
P(win | fill) ≈ model_probability
```

但线上挂单不是随机成交。对于 BUY YES/NO 的限价单：

```text
如果市场继续朝模型方向走：
  低价挂单可能不成交，错过收益

如果市场反向打下来：
  低价挂单更可能成交，但此时模型方向更可能错
```

所以实际需要关注：

```text
P(win | fill, limit_price, best_ask, side, confidence, first_minute_state)
```

而不是只看：

```text
P(win) from model
```

本次最典型的证据是：

```text
naive_edge >= 0.10:
  submitted_orders: 84
  fill_rate: 78.57%
  filled_accuracy: 27.27%
  avg_fill_price: 0.3775
  PnL: -44.77
```

静态看这些订单很便宜、edge 很大；真实成交后它们是最大亏损来源。

## 是否应该加 EV 筛选

应该加，但不能加简单的 `model_probability - market_price` 过滤。

推荐先做一个 disabled-by-default 的 fill-adjusted EV 过滤：

```text
expected_ev_per_share =
    p_fill(context)
  * (p_win_given_fill(context) - expected_fill_price(context))
```

其中 context 至少包括：

```text
side
model_confidence
limit_buy_price
best_bid
best_ask
limit_price - best_ask
confidence - limit_price
first_minute_agrees
first_minute_reversal proxy
hour / regime
```

实际下单规则可以是：

```text
trade only if expected_ev_per_share >= min_expected_ev
```

同时建议加两个硬保护：

```text
1. reject if empirical P(win | fill, bucket) <= fill_price + margin
2. reject if bucket sample is too small and price is not extremely favorable
```

## 短期可测试规则

不要直接上线拟合规则。建议先 replay：

1. 阻止 `naive_edge >= 0.10` 且 `limit_price <= best_ask - 0.10` 的远离市场低价挂单。
2. 阻止 `confidence < 0.80` 的中低置信度订单，或提高这一区间的 min EV。
3. 对 `best_ask < 0.45` 的市场做强过滤，因为该桶成交后胜率只有 7.14%。
4. 对 YES 侧单独提高门槛，因为 YES 侧亏损明显更大。

这些规则都必须在 runtime cache 和真实 matched fill 口径上回放，不能只看 submitted order。

## 建议实现方向

新增一个配置关闭的 execution filter：

```yaml
execution_edge:
  enabled: false
  mode: fill_adjusted
  min_expected_ev_per_share: 0.02
  min_bucket_samples: 50
  fallback_reject_uncertain: false
```

报告必须同时输出：

```text
submitted_orders
fill_rate
size_fill_rate
filled_accuracy
avg_fill_price
filled_pnl
realized_ev_per_submitted_order
```

上线前验收标准：

```text
1. matched-fill PnL 改善
2. filled accuracy 提高
3. 不只是减少成交数量
4. 每小时正确数目标不能明显恶化
5. offline selection_score 不作为该 execution filter 的唯一依据
```

## 最终判断

version3 这次亏损是“成交选择性 + 条件胜率下降”的问题，不是单纯静态负 EV 问题。

应该加 EV 筛选，但 EV 必须是成交调整后的：

```text
成交概率 * 成交后胜率边际
```

尤其要防止一种错误直觉：低价订单 `p - price` 很高，不代表真实 EV 高。它可能只是更容易在预测方向已经变坏时成交。
