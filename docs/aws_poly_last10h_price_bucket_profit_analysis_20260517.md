# aws-poly 过去 10 小时收益与价格分桶分析

## 范围

- 生成时间：2026-06-13
- UTC 窗口：2026-05-16T17:25:00+00:00 至 2026-05-17T03:25:00+00:00
- SGT/北京时间窗口：2026-05-17T01:25:00+08:00 至 2026-05-17T11:25:00+08:00
- summary 来源：`tmp_aws_poly_logs/summaries`
- resolved outcome 来源：Gamma closed market outcome，本地缓存落在 `artifacts/state/execution_engine/aws_poly_gamma_outcome_cache.json`
- 机器可读结果：`artifacts/reports/execution_engine/aws_poly_last10h_price_bucket_profit_analysis_20260517.json`

## 口径说明

这份报告把 aws-poly 的 summary 文件与 Gamma resolved outcome 连接起来。summary 里能看到信号、best bid/ask、下单请求、响应 `status`，以及少量 `order_statuses.size_matched` 字段；它不是完整的 CLOB matched-trade 对账流水。

PnL 使用两种口径：

- `replay PnL`：假设 summary 中所有 submitted/live 订单都按提交价格完整成交。这是“如果都成交”的理论重放口径。
- `matched-status PnL`：只计算 summary 中明确显示 `matched` 或 `size_matched > 0` 的部分。这个比 submitted replay 更保守，但仍不如用 CLOB trade history 按 orderID 对账可靠。

成交概率拆成三类：`submit_success_rate` 是下单 API 成功率；`matched_order_rate / matched_share_rate` 是 summary 中可见 matched 比例；`immediate_cross_rate` 是 `order_price >= best_ask` 的立即可成交代理。

## 总览

| metric | value |
|---|---:|
| resolved cycles | 67 |
| accepted cycles | 67 |
| signal coverage | 100.00% |
| signal accuracy | 64.18% |
| submitted/replayed orders | 134 |
| order-level accuracy | 64.18% |
| submit success rate | 100.00% |
| matched order rate from summary | 0.75% |
| matched share rate from summary | 0.00% |
| immediate cross rate | 0.00% |
| avg order price | 0.4397 |
| avg best_ask - order_price | 0.2188 |
| avg confidence - order_price | 0.2586 |
| replay cost | 573.5500 |
| replay PnL | +266.4500 |
| replay ROI on cost | 46.46% |
| matched-status cost | 0.0000 |
| matched-status PnL | +0.0000 |
| matched-status ROI | NA |

## 按 0.1 下单价格分桶

| price_bucket | orders | shares | accuracy | submit_success_rate | matched_order_rate | matched_share_rate | immediate_cross_rate | avg_price | avg_best_ask | avg_order_minus_ask | avg_ask_minus_order | avg_edge_vs_price | cost | pnl | roi | matched_cost | matched_pnl | matched_roi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2-0.3 | 2 | 20.0 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.2750 | 0.3850 | -0.1100 | 0.1100 | 0.5115 | 5.5000 | +14.5000 | 263.64% | 0.0000 | +0.0000 | NA |
| 0.3-0.4 | 8 | 75.0 | 50.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.3387 | 0.4312 | -0.0925 | 0.0925 | 0.3006 | 25.4500 | +9.5500 | 37.52% | 0.0000 | +0.0000 | NA |
| 0.4-0.5 | 65 | 615.0 | 63.08% | 100.00% | 1.54% | 0.00% | 0.00% | 0.4025 | 0.6669 | -0.2645 | 0.2645 | 0.2931 | 247.6000 | +147.4000 | 59.53% | 0.0000 | +0.0000 | NA |
| 0.5-0.6 | 59 | 590.0 | 66.10% | 100.00% | 0.00% | 0.00% | 0.00% | 0.5000 | 0.6893 | -0.1893 | 0.1893 | 0.2063 | 295.0000 | +95.0000 | 32.20% | 0.0000 | +0.0000 | NA |


## 按方向拆分

| side | orders | accuracy | matched_order_rate | avg_price | avg_ask_minus_order | cost | pnl | roi | matched_cost | matched_pnl | matched_roi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NO | 50 | 76.00% | 0.00% | 0.4396 | 0.2056 | 211.8000 | +163.2000 | 77.05% | 0.0000 | +0.0000 | NA |
| YES | 84 | 57.14% | 1.19% | 0.4398 | 0.2267 | 361.7500 | +103.2500 | 28.54% | 0.0000 | +0.0000 | NA |


## 按腿拆分

| leg | orders | accuracy | matched_order_rate | matched_share_rate | avg_price | avg_ask_minus_order | cost | pnl | roi | matched_cost | matched_pnl | matched_roi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 67 | 64.18% | 1.49% | 0.00% | 0.4897 | 0.1688 | 328.1000 | +101.9000 | 31.06% | 0.0000 | +0.0000 | NA |
| second | 67 | 64.18% | 0.00% | 0.00% | 0.3897 | 0.2688 | 245.4500 | +164.5500 | 67.04% | 0.0000 | +0.0000 | NA |


## 按响应状态拆分

| response_status | orders | accuracy | matched_order_rate | avg_price | avg_ask_minus_order | cost | pnl | roi | matched_cost | matched_pnl | matched_roi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| live | 133 | 64.66% | 0.00% | 0.4400 | 0.2203 | 569.5500 | +270.4500 | 47.48% | 0.0000 | +0.0000 | NA |
| matched | 1 | 0.00% | 100.00% | 0.4000 | 0.0200 | 4.0000 | -4.0000 | -100.00% | 0.0000 | +0.0000 | NA |


## submitted replay 最差订单样本

| t0 | side | actual | leg | response_status | price | size | size_matched | best_ask | ask_minus_order | confidence | edge_vs_price | pnl | matched_pnl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-05-16 17:45 | NO | YES | first | live | 0.5000 | 10.0000 | 0.0000 | 0.8000 | 0.3000 | 0.6425 | 0.1425 | -5.0000 | -0.0000 |
| 2026-05-16 17:50 | YES | NO | first | live | 0.5000 | 10.0000 | 0.0000 | 0.5700 | 0.0700 | 0.6053 | 0.1053 | -5.0000 | -0.0000 |
| 2026-05-16 18:05 | YES | NO | first | live | 0.5000 | 10.0000 | 0.0000 | 0.6500 | 0.1500 | 0.8026 | 0.3026 | -5.0000 | -0.0000 |
| 2026-05-16 18:15 | YES | NO | first | live | 0.5000 | 10.0000 | 0.0000 | 0.6300 | 0.1300 | 0.7827 | 0.2827 | -5.0000 | -0.0000 |
| 2026-05-16 18:20 | YES | NO | first | live | 0.5000 | 10.0000 | 0.0000 | 0.5700 | 0.0700 | 0.6167 | 0.1167 | -5.0000 | -0.0000 |
| 2026-05-16 19:30 | NO | YES | first | live | 0.5000 | 10.0000 | 0.0000 | 0.7000 | 0.2000 | 0.8111 | 0.3111 | -5.0000 | -0.0000 |
| 2026-05-16 19:20 | YES | NO | first | live | 0.5000 | 10.0000 | 0.0000 | 0.7200 | 0.2200 | 0.6433 | 0.1433 | -5.0000 | -0.0000 |
| 2026-05-16 18:35 | YES | NO | first | live | 0.5000 | 10.0000 | 0.0000 | 0.5600 | 0.0600 | 0.5467 | 0.0467 | -5.0000 | -0.0000 |
| 2026-05-16 21:20 | YES | NO | first | live | 0.5000 | 10.0000 | 0.0000 | 0.6300 | 0.1300 | 0.7128 | 0.2128 | -5.0000 | -0.0000 |
| 2026-05-16 21:00 | NO | YES | first | live | 0.5000 | 10.0000 | 0.0000 | 0.5100 | 0.0100 | 0.6058 | 0.1058 | -5.0000 | -0.0000 |


## submitted replay 最好订单样本

| t0 | side | actual | leg | response_status | price | size | size_matched | best_ask | ask_minus_order | confidence | edge_vs_price | pnl | matched_pnl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-05-16 20:20 | NO | NO | second | live | 0.2600 | 10.0000 | 0.0000 | 0.3700 | 0.1100 | 0.7718 | 0.5118 | +7.4000 | +0.0000 |
| 2026-05-16 19:55 | NO | NO | second | live | 0.2900 | 10.0000 | 0.0000 | 0.4000 | 0.1100 | 0.8013 | 0.5113 | +7.1000 | +0.0000 |
| 2026-05-16 20:20 | NO | NO | first | live | 0.3600 | 10.0000 | 0.0000 | 0.3700 | 0.0100 | 0.7718 | 0.4118 | +6.4000 | +0.0000 |
| 2026-05-16 19:55 | NO | NO | first | live | 0.3900 | 10.0000 | 0.0000 | 0.4000 | 0.0100 | 0.8013 | 0.4113 | +6.1000 | +0.0000 |
| 2026-05-16 22:05 | NO | NO | second | live | 0.3900 | 10.0000 | 0.0000 | 0.5000 | 0.1100 | 0.6534 | 0.2634 | +6.1000 | +0.0000 |
| 2026-05-16 19:35 | NO | NO | second | live | 0.4000 | 10.0000 | 0.0000 | 0.6400 | 0.2400 | 0.7826 | 0.3826 | +6.0000 | +0.0000 |
| 2026-05-17 01:35 | NO | NO | second | live | 0.4000 | 10.0000 | 0.0000 | 0.8500 | 0.4500 | 0.9214 | 0.5214 | +6.0000 | +0.0000 |
| 2026-05-17 01:40 | NO | NO | second | live | 0.4000 | 10.0000 | 0.0000 | 0.8800 | 0.4800 | 0.9549 | 0.5549 | +6.0000 | +0.0000 |
| 2026-05-17 01:45 | YES | YES | second | live | 0.4000 | 10.0000 | 0.0000 | 0.7100 | 0.3100 | 0.6153 | 0.2153 | +6.0000 | +0.0000 |
| 2026-05-17 01:50 | YES | YES | second | live | 0.4000 | 10.0000 | 0.0000 | 0.6200 | 0.2200 | 0.7583 | 0.3583 | +6.0000 | +0.0000 |


## 为什么收益很少

1. 可确认 matched 规模很低。summary 口径下 matched share rate 是 0.00%。即使 submitted replay 显示理论 PnL 为正，也几乎没有可确认成交份额能转化成真实收益。
2. 订单基本没有跨 best ask。`immediate_cross_rate` 是 0.00%，平均 `best_ask - order_price` 是 0.2188。挂得便宜能降低成本，但也显著降低真实成交概率。
3. 理论收益主要来自“假设成交”的便宜挂单。第二腿在 replay 中 ROI 更高，但 summary 里没有确认成交；如果第二腿长期 live，低价 optionality 不会体现在真实 PnL 中。
4. 下单价格决定盈亏平衡命中率。二元合约买入价为 `p` 时，需要约 `p` 的命中率才能打平。高价桶命中率不差也可能利润很薄；低价桶则必须真的成交才有意义。
5. 方向准确率和执行 EV 不是同一件事。这个窗口的 signal accuracy 是 64.18%，但执行层没有把 `confidence - executable price` 作为硬过滤，也没有足够 confirmed fill，因此收益很少。

## 结论

过去 10 小时 aws-poly 的信号并不是完全失效：resolved signal accuracy 为 64.18%，submitted replay 甚至显示理论收益为正。但可确认成交几乎没有，且所有订单都低于 best ask，导致理论收益无法有效落地。

所以“收益很少”的核心原因是执行转化，而不是单纯方向预测：挂单距离 best ask 太远、confirmed fill 太少、低价腿没有兑现。下一步应拉取 CLOB `get_trades`，按 orderID 回连 summary，再用真实 fill price、matched shares、maker/taker、partial fill 和 settlement PnL 重建同样的 0.1 价格桶分析。
