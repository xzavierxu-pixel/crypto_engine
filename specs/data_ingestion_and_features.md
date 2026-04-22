# 数据获取与特征计算逻辑规格说明 (Data Ingestion & Feature Calculation)

本档详细说明了 `crypto-engine` 项目中数据从原始获取到特征生成的全生命周期逻辑。

---

## 1. 数据拉取 (Data Fetching)

系统支持两类主要数据源：现货 K 线数据和衍生品指标数据。

### 1.1 现货 K 线 (Spot OHLCV)
- **来源**: [Binance Vision](https://data.binance.vision/) (归档数据)
- **脚本**: `scripts/download_binance_vision_data.py`
- **逻辑**:
    - 优先尝试下载按月封装的 `.zip` 压缩包。
    - 若当月数据尚未归档（例如当前月份），则自动回退到按日下载并合并。
    - **处理**: 将原始微秒/毫秒时间戳统一转换为毫秒，并导出为 Freqtrade 兼容的 `.feather` 格式及训练用的 `.parquet` 格式。
    - **列定义**: `date` (ms), `open`, `high`, `low`, `close`, `volume`。

### 1.2 衍生品数据 (Derivatives)
- **来源**: Binance FAPI (Futures API) 和 Deribit API。
- **脚本**: `scripts/download_derivatives_public_data.py`
- **指标**:
    - **Funding Rate**: 资金费率（通常每 8 小时更新）。
    - **Open Interest (OI)**: 全网持仓量（5m/15m 频率）。
    - **Basis**: 基差（永续合约价格与现货价格之差）。
    - **Volatility Index**: 波动率索引（通过 Deribit API 获取 DVOL，作为期权市场的情绪代理）。
- **存储**: 统一存储在 `artifacts/data/derivatives/` 目录下，采用 Parquet 格式以保留高精度数值。

---

## 2. 数据对齐与合并 (Alignment Logic)

这是防止“未来函数” (Look-ahead Bias) 的核心层。

### 2.1 衍生品内部合并 (`merge_derivatives_frames`)
- 位置: `src/data/derivatives/aligner.py`
- 逻辑: 将 Funding、OI、Basis 等不同频率的数据按时间戳进行 `outer` 合并，并使用 `ffill()` (向前填充) 填充缺失值。这确保了在任何时间点都有最新的已知指标状态。

### 2.2 衍生品与现货对齐 (`align_derivatives_to_spot`)
- **核心算子**: `pd.merge_asof(..., direction="backward")`
- **逻辑**: 
    - 以现货 K 线的时间戳为基准。
    - 对于每一根现货 K 线，匹配**早于或等于**该时间戳的最新衍生品记录。
    - **安全性**: 严格禁止匹配未来数据（即 `direction="forward"`），确保模型在实时推理时只能看到当时已经公布的衍生品指标。

---

## 3. 特征计算流水线 (Feature Engineering)

特征计算由 `src/features/builder.py` 统筹。

### 3.1 核心流程 (`build_feature_frame`)
1. **标准化**: 将 OHLCV 数据清洗为标准格式。
2. **数据挂载**: 如果配置启用了衍生品，`DerivativesFeatureStore` 会通过上述对齐逻辑将衍生品列（前缀为 `raw_` 或 `derivatives_`）挂载到原始 DataFrame。
3. **特征包转换**: 
    - 系统根据配置中的 `feature_profile` 加载多个“特征包”（Feature Packs）。
    - 每个包（如 `momentum`, `volatility`, `derivatives_funding`）独立计算衍生特征。
    - 特征包通过注册表 (`src/features/registry.py`) 动态加载，方便扩展。
4. **清理**: 计算完成后，删除所有临时的 `raw_` 原始指标列，仅保留最终特征。
5. **网格对齐**: 根据 `grid_minutes`（如 15m）选择对应的行，确保特征矩阵的时间粒度与预测目标一致。

### 3.2 特征包示例
- **基础包**: 计算 RSI, MACD, Bollinger Bands 等经典指标。
- **衍生品包**: 计算资金费率的变化率、持仓量异常波动、期现价差率等高阶特征。

---

## 4. 训练与实盘的一致性 (Parity)

为了彻底消除线下训练与线上推理的逻辑差异：
- **逻辑复用**: `build_feature_frame` 函数同时被 `scripts/build_dataset.py` (训练) 和 `src/services/feature_service.py` (实盘) 调用。
- **配置驱动**: 所有的特征参数（参数窗口、对齐方式、包含哪些包）全部定义在 `config/settings.yaml` 中，作为系统的“单一事实来源”。

---

## 5. 校验机制
- 系统会在构建特征后验证是否存在 `NaN` 或 `Inf`。
- 在 `tests/` 目录下包含多个 `test_train_live_feature_parity_*.py` 脚本，用于对比历史回测特征与模拟实盘特征的数值完全一致。
