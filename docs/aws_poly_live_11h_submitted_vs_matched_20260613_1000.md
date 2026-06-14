# aws-poly 线上过去 11 小时收益分析：submitted vs matched

## 范围

- 服务器：`aws-poly`
- 远端目录：`/home/ubuntu/opt/crypto_engine`
- 北京时间窗口：2026-06-12 23:00 至 2026-06-13 10:00
- UTC 窗口：2026-06-12T15:00:00+00:00 至 2026-06-13T02:00:00+00:00
- 数据源：线上 `artifacts/logs/execution_engine/summaries`、Gamma resolved outcome、Polymarket CLOB `get_trades` matched fills
- 机器可读 JSON：`artifacts/reports/execution_engine/aws_poly_live_11h_submitted_vs_matched_20260613_1000.json`
- 远端原始 Markdown：`artifacts/reports/execution_engine/aws_poly_live_11h_submitted_vs_matched_20260613_1000.md`

## 核心结论

这 11 小时不是“模型完全不准”，主要问题是 **submitted 阶段看起来更好，matched 后准确率和收益被成交选择削弱**。

| metric | value |
|---|---:|
| resolved cycles | 96 |
| accepted cycles | 96 |
| signal coverage | 100.00% |
| signal accuracy | 69.79% |
| submitted orders | 96 |
| matched order ids | 81 |
| order fill rate by id | 84.38% |
| submitted size | 480.0000 |
| matched size | 400.0555 |
| size fill rate | 83.34% |
| submitted replay accuracy | 69.79% |
| submitted replay PnL | +45.3000 |
| submitted replay ROI | 15.64% |
| matched fill accuracy | 63.27% |
| matched avg fill price | 0.6005 |
| matched PnL | +14.8506 |
| matched ROI | 6.18% |

## 预测准确率 vs 成交质量

submitted replay accuracy 是 69.79%，matched fill accuracy 降到 63.27%，下降约 6.53 个百分点。submitted replay PnL 是 +45.30，真实 matched fill PnL 只有 +14.85。

这说明两件事同时存在：

1. 模型方向在 submitted 全集上是有边际的，69.79% 不低。
2. 实际成交集合比 submitted 全集差，成交后的准确率只有 63.27%，且平均成交价 0.6005，对应盈亏平衡命中率约 60.05%，利润空间只剩约 3.22 个百分点。

## Submitted orders：按 0.1 下单价格分桶

| limit bucket | orders | accuracy | size | avg limit | avg ask-limit | avg conf-limit | replay pnl | replay roi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2-0.3 | 1 | 0.00% | 5.0000 | 0.2800 | 0.0100 | 0.3066 | -1.4000 | -100.00% |
| 0.3-0.4 | 2 | 50.00% | 10.0000 | 0.3700 | 0.0100 | 0.2313 | +1.3000 | 35.14% |
| 0.4-0.5 | 11 | 36.36% | 55.0000 | 0.4527 | 0.0100 | 0.1688 | -4.9000 | -19.68% |
| 0.5-0.6 | 32 | 71.88% | 160.0000 | 0.5497 | 0.0822 | 0.0960 | +27.0500 | 30.76% |
| 0.6-0.7 | 29 | 68.97% | 145.0000 | 0.6500 | 0.0517 | 0.0602 | +5.7500 | 6.10% |
| 0.7-0.8 | 20 | 90.00% | 100.0000 | 0.7350 | 0.0780 | 0.0518 | +16.5000 | 22.45% |
| 0.8-0.9 | 1 | 100.00% | 5.0000 | 0.8000 | 0.0600 | 0.0644 | +1.0000 | 25.00% |

## Matched fills：按 0.1 成交价格分桶

| fill bucket | fills | accuracy | shares | avg fill | avg ask-limit | avg conf-fill | pnl | roi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2-0.3 | 1 | 0.00% | 5.0000 | 0.2800 | 0.0100 | 0.3066 | -1.4000 | -100.00% |
| 0.3-0.4 | 3 | 66.67% | 9.9903 | 0.3700 | 0.0100 | 0.2213 | +1.2941 | 35.01% |
| 0.4-0.5 | 11 | 18.18% | 44.9927 | 0.4511 | 0.0100 | 0.1576 | -10.2964 | -50.73% |
| 0.5-0.6 | 31 | 64.52% | 130.0970 | 0.5488 | 0.0613 | 0.1023 | +13.7017 | 19.19% |
| 0.6-0.7 | 32 | 62.50% | 129.9828 | 0.6481 | 0.0419 | 0.0662 | +0.7529 | 0.89% |
| 0.7-0.8 | 18 | 88.89% | 74.9926 | 0.7360 | 0.0650 | 0.0554 | +9.7983 | 17.75% |
| 0.8-0.9 | 2 | 100.00% | 5.0000 | 0.8000 | 0.0600 | 0.0644 | +1.0000 | 25.00% |

## 方向拆分

| side | submitted orders | submitted accuracy | submitted replay pnl | matched fills | matched accuracy | matched avg fill | matched pnl | matched ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NO | 49 | 63.27% | +13.8500 | 45 | 53.33% | 0.5567 | -3.5531 | -3.27% |
| YES | 47 | 76.60% | +31.4500 | 53 | 71.70% | 0.6421 | +18.4037 | 13.98% |

NO 侧是主要拖累：submitted 阶段已经弱于 YES，matched 后准确率进一步降到 53.33%，导致真实 matched PnL 为负。

## Edge 拆分

| edge bucket | submitted accuracy | submitted replay pnl | matched accuracy | matched pnl | matched ROI |
|---|---:|---:|---:|---:|---:|
| -0.10-0.00 | 66.67% | +0.2000 | 66.67% | +0.2000 | 2.04% |
| 0.00-0.05 | 77.78% | +15.5000 | 63.64% | +2.4042 | 3.84% |
| 0.05-0.10 | 77.78% | +26.9000 | 75.68% | +17.3992 | 17.83% |
| 0.10-0.20 | 50.00% | -4.9500 | 46.43% | -10.2469 | -18.52% |
| >=0.20 | 62.50% | +7.6500 | 62.50% | +5.0941 | 34.20% |

最稳定的是 `0.05-0.10` edge 桶；`0.10-0.20` 反而亏损，说明高模型边际不一定代表高质量成交，可能包含强趋势反转或 maker adverse selection。

## 成交角色拆分

| role | fills | accuracy | shares | avg fill | pnl | ROI |
|---|---:|---:|---:|---:|---:|---:|
| maker | 95 | 62.11% | 385.0555 | 0.6010 | +8.6506 | 3.74% |
| taker | 3 | 100.00% | 15.0000 | 0.5867 | +6.2000 | 70.45% |

maker fills 是主力，但质量一般。真实收益主要被 maker 成交后的 62.11% 命中率和 0.6010 均价压薄。

## 为什么收益很少

1. **成交后的样本变差**：submitted 准确率 69.79%，matched 准确率 63.27%。这说明真实成交不是 submitted 全集的随机抽样，成交质量有 adverse selection。
2. **成交价接近盈亏平衡线**：matched 平均成交价 0.6005，要求至少约 60.05% 命中率打平；实际 63.27% 只高出约 3.22 个百分点，所以 PnL 只有 +14.85。
3. **0.4-0.5 桶拖累明显**：matched 0.4-0.5 桶准确率只有 18.18%，PnL -10.30。便宜成交不一定是好成交，低价 maker fill 可能集中在错误方向。
4. **0.6-0.7 桶几乎打平**：该桶 matched 准确率 62.50%，平均成交价 0.6481，低于盈亏平衡要求，因此只贡献 +0.75，ROI 0.89%。
5. **主要正收益来自 0.5-0.6 和 0.7-0.8 桶**：0.5-0.6 桶 +13.70，0.7-0.8 桶 +9.80，但被 0.4-0.5 桶和部分高价错单抵消。
6. **NO 侧成交质量差**：NO submitted replay 为 +13.85，但 matched 后变成 -3.55，说明 NO 侧的真实成交选择明显更差。

## 判断

本窗口主要瓶颈不是“预测准确率低”，而是 **成交质量把预测优势压薄**。submitted 层面方向准确率和理论收益都可以；matched 后准确率下降、平均成交价接近盈亏平衡线，导致真实收益很少。

下一步应把执行过滤重点放在 matched fill EV：提高 `confidence - executable/fill price` 过滤，重点审查 0.4-0.5 桶、NO 侧成交和 maker adverse selection，而不是只看 submitted signal accuracy。
