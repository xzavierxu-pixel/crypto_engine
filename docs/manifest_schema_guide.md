# Manifest 字段指南：schema_manifest.json

`schema_manifest.json` 记录了从原始 CSV 环境向 Parquet 规范化数据环境转换的完整过程。它是特征工程和模型训练阶段最重要的审计依据。

## 1. 全局元数据

*   **`generated_at`**: 规范化处理完成的时间戳。
*   **`source_version`**: 定义了规范化规则的版本（例如 `binance_v2`）。如果后续清洗逻辑发生重大变化，该版本号会升级。
*   **`output_root`**: 指向规范化数据存放的 `normalized` 文件夹的根路径。

---

## 2. `normalized_outputs` (规范化输出明细)

这是一个对象数组，每个对象对应一个生成的 Parquet 文件。

### A. 逻辑定位
*   **`market_family` / `data_type` / `symbol` / `interval`**: 定义了该数据的业务属性。
*   **`output_path`**: 该 Parquet 文件的绝对存储路径。

### B. 源数据追踪 (Lineage)
*   **`source_files`**: 列表，记录了哪些原始 CSV 文件被合并进了这个 Parquet 文件。对于历史数据，这通常包括一个或多个月度文件，加上最近几个月的日度文件。
*   **`source_file_count`**: 参与合并的源文件总数。
*   **`skipped_redundant_daily_source_file_count`**:
    **非常重要**：在下载阶段，系统可能会同时下载月度包和该月内的日度包。在规范化阶段，系统会优先使用月度包，并自动丢弃重复的日度 CSV。该字段记录了被丢弃的冗余日度文件数量。

### C. 数据清洗记录 (Cleaning Stats)
*   **`dropped_duplicate_or_out_of_order_event_rows`**:
    记录了在写入 Parquet 之前，因为以下原因被剔除的行数：
    1.  **重复行**：具有完全相同的 Timestamp 和 ID 的事件。
    2.  **乱序行**：时间戳早于已处理行的事件（系统只保证单调递增写入）。
    3.  **覆盖冲突**：在合并月度/日度数据时产生的重叠行。

### D. 数据架构与内联质量检查 (Schema & Inline QA)
*   **`schema` 对象**: 反映了文件的物理列结构以及**规范化过程中的实时质量统计**。
    *   **基础字段**:
        *   `row_count`: 最终写入 Parquet 的总行数。
        *   `columns`: 包含字段名列表。
        *   `dtypes`: 字段的物理类型映射（如 `int64`, `float64`）。
        *   `start` / `end`: 该文件覆盖的第一个和最后一个时间戳。
    *   **`qa` 内联子对象**: 
        > **注意**：这里的 QA 统计是在写入 Parquet 的过程中实时计算的（基于内存流），用于快速反馈转换过程中的异常。
        *   `duplicate_timestamp_symbol_rows`: 记录在合并多个源文件时，由于时间重叠产生的重复行总数。
        *   `timestamp_monotonic_increasing`: 布尔值。如果在写入过程中发现任何时间戳回退（乱序），此项将变为 `false`。
        *   `null_count_by_column`: 每一列缺失值（NaN）的精确计数。对于核心特征（如 price），理想情况下应为 0。
        *   `strict_1m_continuity`: (仅限 1m K线) 实时验证每一行之间是否严格间隔 60 秒。
        *   `gap_count_gt_1m`: (仅限 1m K线) 实时统计发现的时间空隙数量。

---

## 3. 常见问题：为什么有两个 QA 来源？

你可能会发现 `schema_manifest.json` 里的 `qa` 字段与独立的 `qa_manifest.json` 文件内容相似。它们的区别在于：

1.  **执行时机**: `schema_manifest` 里的 QA 是在**写入数据时**生成的“过程指标”；而 `qa_manifest` 是在所有数据落盘后，通过读取 Parquet 文件运行的“最终验收指标”。
2.  **深度不同**: `qa_manifest` 会进行更复杂的“跨表对齐”和“业务逻辑校验”，而 `schema_manifest` 里的 QA 仅关注单流写入的正确性。
3.  **互补性**: 如果规范化过程中 `dropped_duplicate_or_out_of_order_event_rows` 很高，你会在 `schema_manifest` 中看到记录，这能帮你定位是哪个源文件出了问题。
