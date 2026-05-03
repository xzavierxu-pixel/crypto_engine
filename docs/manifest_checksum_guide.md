# Manifest 字段指南：file_checksums.json

`file_checksums.json` 是专门用于审计数据完整性的文件。它记录了每一个原始压缩包在下载时的校验细节，确保输入数据流未经篡改且未发生传输损坏。

## 1. 核心字段定义

*   **`generated_at`** (string, ISO-8601):
    校验记录的生成时间。
*   **`results`** (array):
    包含所有下载请求校验结果的列表。

---

## 2. `results` 条目字段详解

每个校验结果对象包含以下字段：

*   **`market_family`** (string):
    数据所属的市场家族，如 `spot`（现货）、`futures_um`（U本位合约）。
*   **`data_type`** (string):
    数据类型，如 `klines`（K线）、`trades`（逐笔交易）。
*   **`symbol`** (string):
    交易对名称，如 `BTCUSDT`。
*   **`interval`** (string|null):
    如果是 K 线数据，记录频率（如 `1m`, `5m`）；否则为 `null`。
*   **`granularity`** (string):
    数据的时间粒度，`monthly` 代表月度包，`daily` 代表日度包。
*   **`period_label`** (string):
    对应的时间标签（YYYY-MM 或 YYYY-MM-DD）。
*   **`checksum_status`** (string):
    校验的最终状态：
    *   `verified`: 本地计算的哈希与官方提供的 `.CHECKSUM` 文件内容完全一致。
    *   `mismatch`: 哈希不匹配，这通常意味着下载过程中发生了文件损坏。
    *   `missing`: 远端服务器未找到对应的校验和文件。
    *   `error`: 在下载或读取校验和文件时发生了 HTTP 错误。
    *   `skipped`: 如果配置文件中关闭了校验功能，或文件被标记为已跳过下载。
*   **`expected_checksum`** (string|null):
    从 Binance Vision 官方下载的校验码内容。
*   **`actual_checksum`** (string|null):
    对本地下载的 `.zip` 文件内容进行 SHA256 计算得到的 64 位十六进制哈希值。
*   **`status`** (string):
    下载任务的执行状态（冗余记录自 `download_manifest`），方便在查看校验失败时快速定位文件下载状态。

## 3. 常见审计场景

1.  **数据损坏排查**: 如果发现 `checksum_status` 为 `mismatch`，应立即删除本地对应的 `raw` 目录并重新运行下载脚本。
2.  **数据丢失确认**: 如果 `status` 为 `missing` 且 `checksum_status` 为 `not_attempted`，说明远端确实缺失该日期的数据，需要通过插值或多源补偿逻辑处理。
