# Manifest 字段指南：qa_manifest.json

`qa_manifest.json` 是数据管道的最后一道关卡。它通过一系列硬性约束检查，确保规范化后的数据质量能够支持高精度的回测和实盘。

## 1. 深度解析：数据的双重粒度 (Granularity)

在数据管道中，“粒度”是一个多维概念。QA 报告通过以下字段和检查项来确保不同维度的粒度正确无误：

### A. 业务/时间粒度 (Temporal Granularity / Interval)
这在 `tables` 数组中由 `interval` 字段体现。
*   **固定周期 (Interval-based)**: 如 `1m`, `1h`, `1d`。QA 会根据该值激活“连续性检查”。
*   **事件驱动 (Event-driven)**: 如 `trades`, `aggTrades`, `bookTicker`。它们的 `interval` 通常为 `null`。QA 对这类数据的重点在于“单调性”和“原子性”。

### B. 来源粒度 (Source Granularity)
虽然 `qa_manifest.json` 本身不直接展示 `monthly` 或 `daily` 标签（这些在 `schema_manifest.json` 中定义），但 QA 的核心任务是验证**混合粒度合并后的质量**：
*   **月度+日度无缝衔接**: QA 会检查合并后的 Parquet 文件在月度包和日度包的交界处是否出现了重复行或时间戳回退。
*   **数据列一致性**: 确保 `source_granularity` 这一列被正确注入，以便下游特征工程可以识别哪些数据源自更稳定的月度存档，哪些源自最新的日度更新。

---

## 2. 汇总统计 (`summary`)

*   **`table_count`**: 扫描到的 Parquet 总数。
*   **`table_pass_count`**: 所有的 `checks` 都通过的表数量。
*   **`table_fail_count`**: 至少有一项 `checks` 失败的表数量。
*   **`cross_table_check_count`**: 运行的跨源对齐检查项数量（例如 BTC 现货与永续的对比）。

---

## 3. 单表检查详情 (`tables`)

### A. 核心元数据
*   **`row_count`**: 文件总行数。
*   **`passed`**: 总体验证是否通过。只有当所有关键检查项均为 `true` 时，此项才为 `true`。

### B. 基础检查项 (`checks`)
*   **`required_columns_present`**: 是否包含该数据类型必须具备的列（如 K 线必须有 Open/High/Low/Close/Volume/Timestamp）。
*   **`non_empty`**: 确保文件不是只有表头而没有数据的空表。
*   **`timestamp_monotonic_increasing`**:
    **最高优先级检查**：验证时间戳是否严格递增。如果为 `false`，说明在合并不同粒度的源文件（如 2024-05 月度包和 2024-05-01 日度包）时产生了逻辑冲突。
*   **`duplicate_timestamp_symbol_rows`**: 计数同一时间点重复的行。
    *   对于 **1m K 线**：该值必须为 0。
    *   对于 **逐笔成交**：通过 `trade_id` 或 `agg_trade_id` 进行更深度的去重验证。
*   **`has_negative_value_violation`**: 检查价格 (Price)、成交量 (Volume) 等物理意义上不可能为负数的列是否存在负值。

### C. 连续性检查 (Continuity) - 针对 Interval 数据
*   **`strict_1m_continuity`**: 仅针对 `interval: "1m"` 的数据。
    *   **验证逻辑**: 检查任意连续两行之间的时间戳差值是否恰好为 60,000 毫秒。
    *   **业务意义**: 如果不连续，技术指标（如 RSI, MACD）的计算将会产生偏移。
*   **`gap_count_gt_1m`**: 如果不连续，记录缺失了多少分钟的数据。这有助于量化数据的“破碎度”。

### D. 特殊逻辑检查
*   **`event_stream_aggregatable`**: 针对逐笔成交（Trades），检查其架构和时间戳是否足以被正确地聚合（Aggregated）为 OHLCV 结构。这要求数据必须完整且时间戳单调。

---

## 4. 跨表一致性 (`cross_table_checks`)

这部分反映了数据系统的整体同步质量，特别是在处理不同数据源的“对齐粒度”时：

*   **`overlap_count`**: 计算两张表（如 `Spot BTC` 和 `UM Futures BTC`）在 1 分钟精度下，有多少个共同的时间点。
*   **`alignable`**: 这是一个门槛标志。如果两张表虽然都有数据，但时间段完全错开，则无法进行联动分析。
*   **`forward_fill_ready`**: 专门针对资金费率（Funding Rate）等低频数据。QA 会验证这些数据是否可以在 1 分钟的“细粒度”时间格上进行安全填充，而不产生未来的偏见（Look-ahead bias）。

