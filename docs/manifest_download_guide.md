# Manifest 字段指南：download_manifest.json

`download_manifest.json` 是数据采集阶段（Data Acquisition）的核心元数据文件。它详细记录了系统与 Binance Vision 服务器交互的所有请求、下载结果以及潜在的错误信息。

## 1. 根节点字段 (Root Fields)

*   **`generated_at`** (string, ISO-8601):
    清单文件的生成时间。这标志着数据采集任务结束并转入规范化阶段的时间点。
*   **`provider`** (string):
    数据来源标识。在当前系统中通常固定为 `"binance"`。
*   **`start_date`** (string, YYYY-MM-DD):
    在 `settings.yaml` 中配置的回填起始日期。系统会根据此日期计算出需要采集的所有月份和日期。
*   **`as_of_date`** (string, YYYY-MM-DD):
    执行脚本时的参考日期。通常决定了日度数据的采集终点（通常是 `as_of_date - 1`）。
*   **`zip_retention`** (string):
    原始压缩包的保留策略。默认为 `"deleted_after_extract"`，表示在 CSV 文件成功解压到本地后，原始 ZIP 文件会被立即删除以节省空间。

---

## 2. `summary` 字段 (任务汇总)

该对象提供了对整个下载任务的全局统计：

*   **`total_requests`** (int):
    系统根据日期范围和交易对列表计算出的总请求数（包括月度和日度请求）。
*   **`successful_requests`** (int):
    状态为 `downloaded` 的请求总数。
*   **`failed_requests`** (int):
    状态属于错误类（如 `missing`, `checksum_mismatch`, `error`）的请求总数。注意：`skipped` 和 `unavailable_by_listing` 不被计为错误。
*   **`status_counts`** (object):
    各种状态值的分布映射（例如 `{"downloaded": 352, "skipped": 569, ...}`）。
*   **`has_errors`** (boolean):
    如果存在任何一个失败请求，该值为 `true`。

---

## 3. 条目明细字段 (Entry Fields)

清单中包含三个数组：`downloaded` (新下载), `unavailable_by_listing` (服务器不存在), `missing_or_failed` (下载/校验出错)。每个条目的结构如下：

### A. `request` 对象 (原始请求参数)
*   **`market_family`**: 市场类型（`spot`, `futures_um`, `futures_cm`, `option`）。
*   **`data_type`**: 数据类别（如 `klines`, `aggTrades`, `fundingRate`）。
*   **`symbol`**: 交易对（如 `BTCUSDT`）。
*   **`interval`**: K 线周期（如 `1m`, `1h`），非 K 线数据为 `null`。
*   **`granularity`**: 粒度标识（`monthly` 或 `daily`）。
*   **`period_label`**: 时间标签（如 `2024-05` 代表月度，`2024-05-01` 代表日度）。
*   **`url`**: 下载 ZIP 文件的完整 URL。
*   **`checksum_url`**: 下载校验和文件的完整 URL。
*   **`raw_dir`**: 数据解压后存放在本地的绝对路径。
*   **`object_key`**: S3 存储桶中的对象键。
*   **`filename`**: 本地保存的文件名。

### B. 状态与校验信息
*   **`status`**:
    *   `downloaded`: 本次任务新下载并解压成功。
    *   `skipped`: 检测到本地已存在解压后的 CSV，未重复操作。
    *   `unavailable_by_listing`: 通过 S3 Bucket Listing 确认服务器端不存在该文件。
    *   `missing`: 尝试下载时返回 404。
    *   `checksum_mismatch`: 下载成功但 SHA256 校验不匹配。
    *   `extract_failed`: ZIP 文件损坏无法解压。
*   **`checksum_status`**:
    *   `verified`: SHA256 匹配。
    *   `mismatch`: SHA256 不匹配。
    *   `missing`: 服务器未提供校验文件。
    *   `not_rechecked_existing_extract`: 对于 `skipped` 的文件，系统默认不再重新校验。
*   **`expected_checksum`**: 远端提供的 SHA256 哈希。
*   **`actual_checksum`**: 本地计算的 SHA256 哈希。
*   **`zip_deleted`**: 是否已删除 ZIP 文件。
*   **`message`**: 详细的错误描述或状态说明。
*   **`extracted_files`**: 列表，记录该条目解压出的所有 `.csv` 文件的绝对路径。
