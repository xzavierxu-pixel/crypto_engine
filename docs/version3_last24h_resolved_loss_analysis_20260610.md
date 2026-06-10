# version3 过去 24 小时 resolved 成交表现复盘

生成时间：2026-06-10 02:37:53 UTC  
分析窗口：2026-06-09 02:37:53 UTC 至 2026-06-10 02:37:53 UTC  
服务器可用 summary 覆盖：约 2026-06-09 12:15 UTC 至 2026-06-10 02:25 UTC  
数据源：

- server：`version3:/home/ec2-user/fortune_bot`
- summary 日志：`artifacts/logs/execution_engine/summaries`
- CLOB matched trades：`py_clob_client_v2.client.ClobClient.get_trades`
- resolved outcome：Polymarket Gamma closed market outcome
- 本地结果：`artifacts/reports/execution_engine/version3_last24h_resolved_analysis.json`

## 结论

version3 这段 resolved matched fill 是亏损的。

| 指标 | 数值 |
|---|---:|
| resolved matched fills | 147 |
| wins / losses | 71 / 76 |
| accuracy | 48.30% |
| matched shares | 711.233438 |
| cost | 396.119957 |
| avg fill price | 0.5569 |
| PnL | -45.854536 |
| ROI on cost | -11.58% |

亏损不是因为“所有高价单都错”，而是三个因素叠加：

1. 真实 matched fill accuracy 只有 48.30%，低于平均成交价 0.5569 对应的盈亏平衡胜率。
2. 低价桶和中高价桶同时亏：低价单虽然便宜，但命中率极低；0.75-0.85 价格桶命中率 66.67%，仍不足以覆盖高成本。
3. 第一分钟反转和 first-minute conflict 仍是主要亏损来源；尤其 maker fills 被动成交后出现明显 adverse selection。

## 当前线上配置特征

从 server live 日志和 `execution_engine/config.yaml` 看，version3 当前运行形态是：

| 配置项 | 数值 |
|---|---:|
| active artifact | `execution_engine/deploy/baseline` |
| feature_count | 1805 |
| t_up / t_down | 0.525 / 0.46 |
| first leg size | 6.0 |
| second leg | disabled |
| execution_edge | disabled |
| first price mode | `limit_config_best_ask_offset` |

这意味着系统交易更密集、第一腿单量更大，而且没有 `min_edge` 过滤来阻止低 EV 成交。

## 按方向拆分

| side | fills | wins | losses | accuracy | avg price | PnL | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| YES | 86 | 37 | 49 | 43.02% | 0.5065 | -39.684543 | -19.02% |
| NO | 61 | 34 | 27 | 55.74% | 0.6264 | -6.169993 | -3.29% |

主要亏损来自 YES 侧。YES 的平均价格不高，但准确率只有 43.02%，不足以覆盖 0.5065 的成本。

## 按成交价格拆分

| price bucket | fills | accuracy | avg price | PnL | ROI |
|---|---:|---:|---:|---:|---:|
| 0.00-0.45 | 45 | 13.33% | 0.2691 | -29.458509 | -49.54% |
| 0.45-0.55 | 31 | 45.16% | 0.5012 | -5.659522 | -8.24% |
| 0.55-0.65 | 20 | 60.00% | 0.6011 | -0.508203 | -0.95% |
| 0.65-0.75 | 9 | 55.56% | 0.7053 | -8.380000 | -22.42% |
| 0.75-0.85 | 24 | 66.67% | 0.8045 | -12.740000 | -13.31% |
| 0.85-1.01 | 18 | 100.00% | 0.8820 | +10.891698 | +13.38% |

最反直觉的点：亏损最大的是低价桶 `0.00-0.45`，不是最高价桶。低价桶只有 13.33% 命中率，虽然单价便宜，仍然亏了 -29.46。

高价也有问题：`0.75-0.85` 虽然 66.67% 命中，但平均价格 0.8045，盈亏平衡要求约 80.45% 命中率，所以仍亏 -12.74。

## 按置信度拆分

| confidence bucket | fills | accuracy | avg price | PnL |
|---|---:|---:|---:|---:|
| 0.50-0.60 | 36 | 25.00% | 0.4039 | -21.770000 |
| 0.60-0.70 | 51 | 43.14% | 0.5007 | -16.164311 |
| 0.70-0.80 | 28 | 50.00% | 0.6762 | -22.813548 |
| 0.80-0.90 | 30 | 80.00% | 0.7472 | +12.463323 |
| 0.90-1.01 | 2 | 100.00% | 0.7791 | +2.430000 |

0.50-0.80 置信度区间合计亏约 -60.75，是主要问题。0.80 以上反而是正收益。

## 第一分种行为和反转

| 分组 | fills | accuracy | avg price | PnL | ROI |
|---|---:|---:|---:|---:|---:|
| 模型方向与第一分钟一致 | 81 | 64.20% | 0.7076 | -18.594880 | -6.72% |
| 模型方向与第一分钟不一致 | 66 | 28.79% | 0.3727 | -27.259656 | -22.86% |

第一分钟一致时，方向准确率明显更高，但价格也更高，平均 0.7076，导致 64.20% 的命中率仍然不够赚钱。

第一分钟不一致时，平均价格低，但命中率只有 28.79%，亏损更严重。

按第一分钟是否反转看：

| first-minute reversal | fills | accuracy | avg price | PnL |
|---|---:|---:|---:|---:|
| False | 99 | 52.53% | 0.5574 | -12.755160 |
| True | 48 | 39.58% | 0.5560 | -33.099376 |

反转窗口贡献了 -33.10 的亏损，是最明确的结构性亏损来源。

## Maker / taker

| role | fills | accuracy | avg price | PnL | ROI |
|---|---:|---:|---:|---:|---:|
| maker | 143 | 46.85% | 0.5475 | -49.994536 | -13.29% |
| taker | 4 | 100.00% | 0.8275 | +4.140000 | +20.85% |

这段亏损主要来自 maker fills。不是因为主动 taker 追价，而是挂单被成交时质量差：便宜单大量成交在错误方向上，高价 maker 单又需要更高胜率。

## 最差订单样本

| t0 UTC | side | actual | price | size | PnL | confidence | first-minute agrees | reversal |
|---|---|---|---:|---:|---:|---:|---|---|
| 2026-06-09 16:35 | YES | NO | 0.82 | 6.0 | -4.92 | 0.7992 | True | True |
| 2026-06-10 00:50 | NO | YES | 0.79 | 6.0 | -4.74 | 0.7622 | True | True |
| 2026-06-10 02:20 | YES | NO | 0.79 | 6.0 | -4.74 | 0.6779 | True | True |
| 2026-06-09 14:00 | YES | NO | 0.78 | 6.0 | -4.68 | 0.7316 | True | True |
| 2026-06-09 14:40 | YES | NO | 0.72 | 6.0 | -4.32 | 0.7553 | True | True |

这些最差订单有共同点：模型方向与第一分钟一致，但最终 resolved 反转。系统在第一分钟方向上付了较高价格，后四分钟反向结算，单笔损失接近整笔成本。

## 为什么 summary 整单估算会误导

如果只看 summary 里提交订单并假设整单成交，本次估算会得到：

| 口径 | accuracy | avg price | PnL |
|---|---:|---:|---:|
| submitted order estimate | 58.33% | 0.5368 | +37.91 |
| true matched fills | 48.30% | 0.5569 | -45.85 |

差异说明：必须用 CLOB matched fills 分析，不能只用 server summary 的 submitted/live order 记录。真实成交有选择性，成交到的订单比提交全集更差。

## 亏损原因归纳

1. `execution_edge.enabled=false`，没有按 `confidence - fill_price` 做最小 EV 过滤。
2. `t_up=0.525/t_down=0.46` 比当前 accepted baseline 更宽，覆盖高但弱信号更多；0.50-0.80 confidence 区间整体亏损。
3. 第一腿 size=6 且第二腿关闭，亏损集中在单一价格层，没有低价第二腿分散成本。
4. maker fill 出现 adverse selection：低价成交并不代表便宜优势，0.00-0.45 桶只有 13.33% 命中。
5. 第一分种反转仍未被过滤，反转窗口 PnL -33.10。
6. YES 侧明显弱于 NO 侧，YES fills 亏 -39.68，是方向不对称问题。

## 建议

1. 先停止用 submitted summary 评估盈亏，所有 live 复盘必须以 matched fills 为准。
2. 恢复或测试 `execution_edge` 过滤，至少记录并回放 `confidence - fill_price`。
3. 对 0.50-0.80 confidence 区间做回放门控；当前数据显示该区间是主要亏损区。
4. 单独评估 YES 侧过滤或更高 YES threshold，因为 YES 侧是最大亏损来源。
5. 针对 first-minute reversal 做过滤，但不能只用“是否跟第一分钟一致”；一致时仍可能因价格过高亏钱。
6. 下一轮 replay 应固定 execution config，只比较 artifact/postprocess，并报告真实 matched-fill EV。
