# Execution Engine

线上执行引擎用于把当前项目的 BTC/USDT 5m baseline 信号接到 Polymarket BTC 5m UP/DOWN 市场。它不是新模型系统：模型、校准器、特征列、阈值和核心特征逻辑继续复用 `src/`。

默认配置是 `paper` 且 `orders.enabled=false`，不会提交真实订单。

## Baseline

默认部署 baseline：

```text
execution_engine/deploy/baseline
```

该目录已包含线上运行必需的 baseline 文件，并应随 git 部署：

```text
artifact_manifest.json
catboost_lgbm_logit_blend.binary.pkl
platt_logit.binary.pkl
metrics.json
threshold_search.json
```

实际 inference 必需的是 `artifact_manifest.json`、模型 pkl 和校准器 pkl；`metrics.json` 与 `threshold_search.json` 用于部署审计和确认阈值来源。

关键约束：

```yaml
model_plugin: catboost_lgbm_logit_blend
calibration_plugin: platt_logit
t_up: 0.585
t_down: 0.335
feature_count: 516
```

阈值从 artifact manifest 读取。不要在执行代码里硬编码 `0.5` 作为信号阈值。

## Files

```text
execution_engine/
  deploy/baseline/
  config.example.yaml
  config.py
  artifacts.py
  realtime_data.py
  feature_runtime.py
  polymarket_v2.py
  order_plan.py
  run_once.py
  prewarm.py
  scheduler/
  scripts/
```

真实配置文件为 `execution_engine/config.yaml`，已加入 `.gitignore`。

## Linux Deploy

准备代码：

```bash
cd /opt
sudo mkdir crypto_engine
sudo chown -R $USER:$USER /opt/crypto_engine
git clone -b baseline https://github.com/xzavierxu-pixel/crypto_engine.git crypto_engine
```

安装依赖：

```bash
python3 --version
sudo apt update
sudo apt install -y python3.12-venv python3-pip git
rm -rf .venv
bash execution_engine/scripts/install_linux.sh
```

脚本会先安装 `execution_engine/requirements.txt` 中的运行时依赖，包括 `scikit-learn==1.7.2`、`catboost`、`lightgbm` 和 `xgboost`。这些包是加载当前 baseline 模型插件和校准器所需的依赖。
脚本还会从 `https://github.com/Polymarket/py-clob-client-v2.git` 安装 v2 CLOB client。当前 v2 README 的公开用法是从包根导入 `ApiCreds`、`ClobClient`、`OrderArgs`、`OrderType`、`PartialCreateOrderOptions` 和 `Side`。

如果已经创建过 `.venv` 且 prewarm 报 `ModuleNotFoundError: No module named 'sklearn'`，在服务器上补装依赖即可：
```bash
. .venv/bin/activate
python -m pip install -r execution_engine/requirements.txt
```

如果已经装到了其他 `scikit-learn` 版本，按 artifact 训练版本重装：
```bash
python -m pip install --force-reinstall scikit-learn==1.7.2
```

检查并编辑配置：

```bash
cp execution_engine/config.example.yaml execution_engine/config.yaml
vim execution_engine/config.yaml
```

确认 baseline 文件存在：

```bash
ls execution_engine/deploy/baseline
```

## Credentials

live mode 需要 Polymarket CLOB v2 凭证。不要把真实值写进 git：

```bash
cat > execution_engine/secrets.env <<'EOF'
POLYMARKET_PRIVATE_KEY=...
CLOB_API_KEY=...
CLOB_SECRET=...
CLOB_PASS_PHRASE=...
EOF
chmod 600 execution_engine/secrets.env
```

手动 shell 运行时：

```bash
set -a
. execution_engine/secrets.env
set +a
```

## Paper Smoke Test

先保持：

```yaml
runtime:
  mode: paper
orders:
  enabled: false
```

可选 prewarm：

```bash
. .venv/bin/activate
python execution_engine/prewarm.py --config execution_engine/config.yaml --cache-output artifacts/state/execution_engine/prewarm --print-json
```

这会从 Binance 拉取当前 1m/1s kline 和 aggTrades 数据，使用 baseline 特征列构建 runtime feature frame，并写出 `minute.parquet`、`second.parquet`、`agg_trades.parquet`、`second_level_sampled.parquet`、`features.parquet` 和 `summary.json`。

运行一次 paper cycle：

```bash
. .venv/bin/activate
python execution_engine/run_once.py --config execution_engine/config.yaml --mode paper --print-json
```

输出会写入：

```text
artifacts/logs/execution_engine/live.jsonl
artifacts/logs/execution_engine/summaries/
```

## Live Mode Before Start

启用 live 前必须完成：

1. `execution_engine/config.yaml` 使用部署 baseline：

```yaml
baseline:
  artifact_dir: execution_engine/deploy/baseline
```

2. 确认 `execution_engine/secrets.env` 已配置：

```bash
grep -E 'POLYMARKET_PRIVATE_KEY|CLOB_API_KEY|CLOB_SECRET|CLOB_PASS_PHRASE' execution_engine/secrets.env
```

3. 确认 Polymarket 主网配置：

```yaml
polymarket:
  host: https://clob.polymarket.com
  gamma_base_url: https://gamma-api.polymarket.com
  chain_id: 137
```

4. 至少跑一次 paper mode，并检查 summary 里的 `market.slug`、`target_token_id`、`best_bid`、`p_up`、`t_up`、`t_down`、`orders`。

5. 钱包已授权、资金充足，并确认 `signature_type` / `funder` 是否符合你的钱包模式。

6. 确认幂等状态路径可写：

```bash
mkdir -p artifacts/state/execution_engine
touch artifacts/state/execution_engine/idempotency.json
```

7. 最后才切换 live：

```yaml
runtime:
  mode: live
orders:
  enabled: true
```

## Systemd Timer

只使用 systemd timer 做生产定时执行，避免重复触发同一窗口。

复制示例：

```bash
sudo cp execution_engine/scheduler/execution-engine.service.example /etc/systemd/system/execution-engine.service
sudo cp execution_engine/scheduler/execution-engine.timer.example /etc/systemd/system/execution-engine.timer
sudo systemctl daemon-reload
sudo systemctl enable --now execution-engine.timer
```

修改 `execution_engine/config.yaml` 或 `execution_engine/secrets.env` 后：

```bash
sudo systemctl daemon-reload
sudo systemctl restart execution-engine.timer
sudo systemctl start execution-engine.service
```

查看状态：

```bash
systemctl list-timers execution-engine.timer
journalctl -u execution-engine.service -n 100 --no-pager
```

停用：

```bash
sudo systemctl disable --now execution-engine.timer
```

## Maintenance

查看最近 summary：

```bash
ls -lt artifacts/logs/execution_engine/summaries | head
```

查看审计：

```bash
tail -n 50 artifacts/logs/execution_engine/live.jsonl
```

清理幂等状态只应在确认不会重复下单后进行：

```bash
rm artifacts/state/execution_engine/idempotency.json
```

## Troubleshooting

`market_not_found`：Polymarket 当前 BTC 5m slug 未找到，检查系统时钟、Gamma API 和预测窗口对齐。

`missing_best_bid`：目标 token 没有 best bid，执行引擎会跳过下单。

`Runtime feature frame is missing ... baseline features`：实时数据不能构造 baseline 所需特征，检查 1m/1s 字段、lookback 和 second-level profile。

`py-clob-client-v2 is required`：部署环境缺少 v2 client，重新运行安装脚本。

`ModuleNotFoundError: No module named 'sklearn'`：部署环境缺少模型/校准运行时依赖，执行 `. .venv/bin/activate` 后运行 `python -m pip install -r execution_engine/requirements.txt`。

`InconsistentVersionWarning` for `LabelEncoder` / `LogisticRegression`：当前 artifact 使用 `scikit-learn 1.7.2` 保存，执行 `python -m pip install --force-reinstall scikit-learn==1.7.2`。

`idempotency_key_already_seen`：同一窗口同一 token 已提交过两单计划，默认跳过，防止重复下单。
