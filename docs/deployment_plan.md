# Deployment 文件夹方案设计文档

> 本文档只描述**拟引入的改动**，不修改任何现有代码。实施时请严格按本文件列出的目录结构、文件清单和内容骨架落地。所有新增文件均位于仓库根的新建目录 `deployment/` 下；除 `.gitignore` 外，**不修改任何现有文件**。

---

## 1. 目标与范围

在 Linux 服务器（推荐 Ubuntu 22.04 LTS，systemd ≥ 249）上，围绕现有 `scripts/` 入口（`download_binance_vision_data.py`、`build_dataset.py`、`train_model.py`、`run_live_signal.py`、`run_shadow.py`）建立一套完整的：

1. **一次性引导（bootstrap）** —— 拉取仓库、建虚拟环境、装依赖、建数据目录、写 systemd 单元。
2. **周期性工作流调度** —— 数据刷新、训练、实时信号/影子执行。
3. **运维与监控** —— 状态查看、日志收集与轮转、健康检查、磁盘清理。
4. **代码更新** —— 拉取新版本、校验、按需重建数据集/重训练、平滑重启。
5. **失败回滚** —— 保留最近 N 份工件，一键回滚到上一个健康版本。

**非目标**：不引入 Docker、K8s、Ansible、Prometheus 等额外基础设施；只用 bash + systemd + logrotate + journalctl 实现最小闭环，后续可以在此骨架上替换/叠加。

---

## 2. 设计原则

- **不重复业务逻辑**：所有命令都调用 `scripts/*.py`，deployment 层只做"编排 + 幂等 + 日志"。
- **单一配置源**：沿用 `config/settings.yaml`。部署层自己的参数集中在 `deployment/.env`，由 `_common.sh` 统一加载。
- **幂等可重入**：任意脚本可重复执行，失败不会造成半成品（用 `set -euo pipefail` + 原子 `mv` + 带时间戳的临时目录）。
- **强隔离**：所有运行时写入的路径（数据、工件、日志、运行时状态）全部在仓库外（默认 `/var/lib/crypto-engine`、`/var/log/crypto-engine`），**仓库本身保持只读**。这样 `git pull` 永不冲突。
- **最小权限**：systemd 单元以非 root 的 `crypto` 用户运行；`ProtectSystem=strict`、`ReadWritePaths` 显式授权。
- **观测优先**：每个周期性脚本结束时写一个 JSON 状态行到 `$STATE_DIR/<job>.last.json`（exit_code、duration、git_sha、artifact_path），供 `status.sh` 与未来的 Prometheus textfile exporter 直接读。
- **RTK 友好**：所有运维命令都是一行 bash，可直接用 `rtk bash deployment/scripts/xxx.sh` 包装（见 `AGENTS.md` / `RTK.md`）。

---

## 3. 目录结构（全部为新增）

```
deployment/
├── README.md                               # 部署总览 + 快速上手
├── .env.example                            # 运维参数模板（复制为 .env 使用）
├── systemd/
│   ├── crypto-engine-data-refresh.service  # 数据刷新 oneshot
│   ├── crypto-engine-data-refresh.timer    # 每日 00:15 UTC
│   ├── crypto-engine-train.service         # 训练 oneshot
│   ├── crypto-engine-train.timer           # 每周一 01:00 UTC
│   ├── crypto-engine-live.service          # 实时信号 oneshot
│   └── crypto-engine-live.timer            # 每 5 分钟，对齐网格
├── scripts/
│   ├── _common.sh                          # 公共库：加载 .env、日志、锁、状态写入
│   ├── bootstrap.sh                        # 首次安装
│   ├── update.sh                           # 拉取新代码并按需重建
│   ├── refresh-data.sh                     # 调 download_binance_vision_data + build_dataset
│   ├── run-train-cycle.sh                  # 调 train_model，写入 artifact 符号链接
│   ├── run-live-cycle.sh                   # 调 run_live_signal（或 run_shadow）
│   ├── healthcheck.sh                      # 检查最近一次各 job 的退出码与 staleness
│   ├── rotate-artifacts.sh                 # 清理过期模型/数据/日志
│   └── tail-logs.sh                        # 快捷 journalctl -fu
├── monitoring/
│   └── status.sh                           # 打印所有 timer 状态 + 最近一次 JSON 状态
└── logrotate/
    └── crypto-engine                       # /etc/logrotate.d 的配置文件
```

**命名约定**：所有 systemd 单元统一前缀 `crypto-engine-`，所有脚本使用 kebab-case，所有环境变量使用 `CE_` 前缀。

---

## 4. 运行时布局（服务器上，仓库外）

```
/opt/crypto-engine/                    # 只读 git 工作副本
└── <仓库内容>

/var/lib/crypto-engine/                # 可写数据根
├── venv/                              # Python 虚拟环境
├── data/
│   ├── raw/                           # Binance Vision 下载结果
│   └── training/                      # build_dataset.py 产物（parquet）
├── artifacts/
│   ├── models/
│   │   ├── 2026-04-22T01-00-00Z/      # 每次训练一个带时间戳目录
│   │   └── current -> 2026-04-22T01-00-00Z  # 符号链接，live 只读这里
│   └── signals/                       # run_live_signal 输出
├── state/                             # *.last.json, *.lock
└── audit/                             # 审计 JSONL（对应 src/execution/audit.py）

/var/log/crypto-engine/                # 脚本 stdout/stderr 镜像（systemd 同时写 journal）
├── refresh-data.log
├── train.log
└── live.log
```

这套布局的关键点：
- `current` 符号链接是**原子发布**的核心：训练成功后脚本先写新目录再 `ln -sfn` 替换，live 进程启动时只读 `current`，不会读到半成品。
- 审计目录 `audit/` 必须与 `src/execution/audit.py` 的默认写入路径一致（实施时从 `config/settings.yaml` 的 `execution.audit.log_path` 读取并透传给脚本；如当前默认路径不同，通过 `.env` 的 `CE_AUDIT_DIR` 覆盖并在脚本里传 `--audit-dir`）。

---

## 5. 文件内容骨架

> 以下为每个新增文件的**内容说明**（不是最终成品代码）。实施时照此骨架写即可，所有脚本第一行统一 `#!/usr/bin/env bash` + `set -euo pipefail`。

### 5.1 `deployment/.env.example`

集中所有部署期参数，包含（并给出注释默认值）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `CE_REPO_DIR` | `/opt/crypto-engine` | 仓库根 |
| `CE_DATA_DIR` | `/var/lib/crypto-engine` | 运行时数据根 |
| `CE_LOG_DIR` | `/var/log/crypto-engine` | 日志目录 |
| `CE_VENV_DIR` | `${CE_DATA_DIR}/venv` | 虚拟环境 |
| `CE_USER` | `crypto` | systemd 运行用户 |
| `CE_CONFIG` | `${CE_REPO_DIR}/config/settings.yaml` | 业务配置 |
| `CE_PAIR` | `BTCUSDT` | Binance symbol |
| `CE_TIMEFRAME` | `1m` | K 线周期 |
| `CE_DATA_WINDOW_DAYS` | `400` | 每次刷新的回溯窗口 |
| `CE_STAGE1_MODEL_PLUGIN` | `lightgbm_stage1` | Stage 1 训练插件名 |
| `CE_STAGE2_MODEL_PLUGIN` | `lightgbm_stage2` | Stage 2 训练插件名 |
| `CE_RUN_MODE` | `shadow` | `shadow` 或 `live` |
| `CE_KEEP_MODEL_VERSIONS` | `10` | 保留最近多少次训练 |
| `CE_KEEP_LOG_DAYS` | `30` | 日志保留天数 |
| `CE_LOCK_TIMEOUT_SEC` | `600` | flock 超时 |
| `CE_ARTIFACT_STALE_HOURS` | `26` | healthcheck 判定数据过期的阈值 |
| `CE_LIVE_STALE_MINUTES` | `15` | healthcheck 判定 live 过期的阈值 |

### 5.2 `deployment/scripts/_common.sh`

公共库，所有其他脚本第一行 `source "$(dirname "$0")/_common.sh"`：

- 加载 `deployment/.env`（必要时从 `.env.example` 报错提示）。
- 定义 `log_info` / `log_warn` / `log_err`（带时间戳与颜色，非 TTY 下自动禁用颜色）。
- 定义 `with_lock <name> <cmd...>`：基于 `flock -w "$CE_LOCK_TIMEOUT_SEC" "$CE_DATA_DIR/state/<name>.lock"`，防止同一 job 并发。
- 定义 `write_state <job> <exit_code> <started_at> <ended_at> [artifact]`：向 `$CE_DATA_DIR/state/<job>.last.json` 原子写入一行 JSON。
- 定义 `activate_venv`：`source "$CE_VENV_DIR/bin/activate"`。
- 定义 `run_python <script> [args...]`：固定 `PYTHONPATH="$CE_REPO_DIR"`，统一入口，便于将来切 uv / poetry。
- 定义 `git_sha`：`git -C "$CE_REPO_DIR" rev-parse --short HEAD`。

### 5.3 `deployment/scripts/bootstrap.sh`

一次性首装，**必须幂等**。按顺序：

1. 断言以 root 执行（需要创建用户和 systemd）。
2. `id -u "$CE_USER" || useradd --system --home-dir "$CE_DATA_DIR" --shell /usr/sbin/nologin "$CE_USER"`。
3. `apt-get install -y python3-venv python3-pip git build-essential logrotate`。
4. 创建目录树 `/var/lib/crypto-engine/{venv,data,artifacts,state,audit}`、`/var/log/crypto-engine`，`chown -R $CE_USER:$CE_USER`。
5. `git clone` 或 `git -C $CE_REPO_DIR pull` 到 `$CE_REPO_DIR`（只读）。
6. 以 `$CE_USER` 身份 `python3 -m venv $CE_VENV_DIR && pip install -r $CE_REPO_DIR/requirements.txt`（若仓库用 `pyproject.toml`，改 `pip install -e .`；实施前 verify）。
7. `install -m 0644 deployment/systemd/*.service /etc/systemd/system/` 以及 `*.timer`。
8. `install -m 0644 deployment/logrotate/crypto-engine /etc/logrotate.d/crypto-engine`。
9. `systemctl daemon-reload`，`systemctl enable --now *.timer`。
10. 执行一次 `refresh-data.sh` + `run-train-cycle.sh`，为 live 产生首个 `current` 符号链接。

### 5.4 `deployment/scripts/update.sh`

代码更新，**不中断运行中的 live cycle**：

1. `with_lock update`：防止自己重入；不与 live lock 竞争（live 是短任务，5 分钟周期内自然窗口）。
2. `git -C $CE_REPO_DIR fetch --tags --prune`。
3. 解析 `$1`（可选）作为目标 ref，默认 `origin/main`；`git -C $CE_REPO_DIR reset --hard <ref>`。
4. 对比 `requirements.txt` / `pyproject.toml` 的哈希，变更时 `pip install ...`。
5. 对比 `config/settings.yaml` 与 `src/core/versioning.py` 中 `CORE_FEATURE_VERSION` / `CORE_LABEL_VERSION`：
   - 若 feature/label 版本变化 → 标记 `need_rebuild=1`；
   - 若仅模型超参变化 → 标记 `need_retrain=1`。
6. 按标记顺序调用 `refresh-data.sh` / `run-train-cycle.sh`（都是幂等的）。
7. 写 state `update.last.json`。
8. **不触碰** live timer；下一个 5 分钟 tick 会自动用新 `current`。

### 5.5 `deployment/scripts/refresh-data.sh`

1. `with_lock data-refresh`。
2. 算窗口：`end=$(date -u +%F)`，`start=$(date -u -d "${CE_DATA_WINDOW_DAYS} days ago" +%F)`。
3. `run_python scripts/download_binance_vision_data.py --pair "$CE_PAIR" --timeframe "$CE_TIMEFRAME" --start "$start" --end "$end" --output "$CE_DATA_DIR/data/raw/${CE_PAIR}_${CE_TIMEFRAME}.parquet"`。（参数名以 `scripts/download_binance_vision_data.py` 实际 CLI 为准，实施时先 `--help` 对齐。）
4. `run_python scripts/build_dataset.py --input <raw> --output "$CE_DATA_DIR/data/training/${CE_PAIR}_5m.parquet" --config "$CE_CONFIG" --horizon 5m`。
5. 写 `refresh-data.last.json`（含行数、时间跨度、文件大小）。

### 5.6 `deployment/scripts/run-train-cycle.sh`

1. `with_lock train`。
2. `ts=$(date -u +%Y-%m-%dT%H-%M-%SZ)`；`out="$CE_DATA_DIR/artifacts/models/$ts"`；`mkdir -p "$out"`。
3. `run_python scripts/train_model.py --input <training.parquet> --output-dir "$out" --config "$CE_CONFIG" --horizon 5m`。
4. **校验产物**：`test -s "$out/model.*"` 且 `test -s "$out/metrics.json"`；解析 metrics.json，对 AUC / logloss 做下限断言（阈值由 `.env` 中 `CE_MIN_AUC`、`CE_MAX_LOGLOSS` 给出，默认放宽到 0.5 / 1.0）。
5. 原子发布：`ln -sfn "$out" "$CE_DATA_DIR/artifacts/models/current.new" && mv -Tf "$CE_DATA_DIR/artifacts/models/current.new" "$CE_DATA_DIR/artifacts/models/current"`。
6. 调用 `rotate-artifacts.sh` 保留最近 `$CE_KEEP_MODEL_VERSIONS` 份。
7. 写 `train.last.json`（含 metrics 摘要、git_sha、config_hash）。

### 5.7 `deployment/scripts/run-live-cycle.sh`

1. `with_lock live --non-blocking`（若已持锁则立即退出 0，保证 timer 不堆积）。
2. 分支：
   - `CE_RUN_MODE=shadow` → `run_python scripts/run_shadow.py ...`
   - `CE_RUN_MODE=live`   → `run_python scripts/run_live_signal.py ...`
3. 必传参数：`--config "$CE_CONFIG"`、`--model-dir "$CE_DATA_DIR/artifacts/models/current"`、`--input <最新 raw/训练数据>`、`--output-dir "$CE_DATA_DIR/artifacts/signals/"`。实际参数名以脚本 `--help` 为准。
4. 捕获 exit code；写 `live.last.json`（含信号、是否下单、guard reason）。
5. 失败**不让 systemd 重试**（已是 5 分钟 timer），仅记录。

### 5.8 `deployment/scripts/healthcheck.sh`

作为 `monitoring/status.sh` 的子例程，也可 `--exit-on-fail` 单独跑在 cron 里做报警：

- 读 `$CE_DATA_DIR/state/*.last.json`：
  - 任一 `exit_code != 0` → FAIL；
  - `refresh-data` 的 `ended_at` 距今 > `CE_ARTIFACT_STALE_HOURS` → FAIL；
  - `live` 的 `ended_at` 距今 > `CE_LIVE_STALE_MINUTES` → FAIL；
  - `current` 符号链接不存在或指向不可读路径 → FAIL。
- 输出多行 `OK|FAIL: <check> <detail>`；`--exit-on-fail` 时任一 FAIL 则 exit 1。

### 5.9 `deployment/scripts/rotate-artifacts.sh`

- 按 mtime 倒序保留 `artifacts/models/*/` 最近 `$CE_KEEP_MODEL_VERSIONS` 份，其余删除；**永远不删 `current` 指向的那一份**。
- 按天数保留 `artifacts/signals/*.jsonl` 与 `$CE_LOG_DIR/*.log`（`CE_KEEP_LOG_DAYS`）。
- 同步 `audit/`：按月归档为 `audit/archive/YYYY-MM.tar.zst`，保留原始最近 7 天。

### 5.10 `deployment/scripts/tail-logs.sh`

可选参数 `$1 ∈ {data,train,live,all}`；对应 `journalctl -fu crypto-engine-<job>.service`（`all` = `-fu crypto-engine-*.service`）。

### 5.11 `deployment/monitoring/status.sh`

一页纸的运维仪表板，输出：

1. `systemctl list-timers 'crypto-engine-*'` 的截取。
2. 每个 `state/*.last.json` 的关键字段：`job`, `exit_code`, `age`, `artifact`。
3. `current` 符号链接指向、模型 metrics 摘要。
4. 磁盘使用：`du -sh $CE_DATA_DIR/{data,artifacts,audit}`。
5. 调用 `healthcheck.sh` 汇总 OK/FAIL。

### 5.12 systemd 单元文件

所有 `.service` 都是 `Type=oneshot`，共用模板：

```ini
[Unit]
Description=Crypto Engine <job>
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=${CE_USER}
Group=${CE_USER}
WorkingDirectory=${CE_REPO_DIR}
EnvironmentFile=${CE_REPO_DIR}/deployment/.env
ExecStart=/usr/bin/env bash ${CE_REPO_DIR}/deployment/scripts/<script>.sh
StandardOutput=append:${CE_LOG_DIR}/<job>.log
StandardError=append:${CE_LOG_DIR}/<job>.log
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=5
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
NoNewPrivileges=true
ReadWritePaths=${CE_DATA_DIR} ${CE_LOG_DIR}

[Install]
WantedBy=multi-user.target
```

> 注：`EnvironmentFile` 不支持变量展开，所以实际文件里是写死的绝对路径；`.env` 里的值必须和这里一致。`bootstrap.sh` 里在 `install` systemd 单元之前用 `envsubst` 从 `.env` 渲染出最终文件。

Timers：

| 单元 | `OnCalendar` | `RandomizedDelaySec` |
|---|---|---|
| `data-refresh` | `*-*-* 00:15:00 UTC` | `120` |
| `train` | `Mon *-*-* 01:00:00 UTC` | `300` |
| `live` | `*:0/5:00` | `0`（严格对齐 5 分钟网格） |

`live.timer` 额外设置 `AccuracySec=1s`，并且对应 service 加 `TimeoutStartSec=240`（< 5 分钟）。

### 5.13 `deployment/logrotate/crypto-engine`

```
/var/log/crypto-engine/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su crypto crypto
}
```

### 5.14 `deployment/README.md`

提供给运维的速查表，结构：

1. **一键安装**：`sudo bash deployment/scripts/bootstrap.sh`。
2. **常用命令**（全部可用 `rtk` 前缀）：
   - 状态：`bash deployment/monitoring/status.sh`
   - 拉新代码：`sudo -u crypto bash deployment/scripts/update.sh [ref]`
   - 手动刷数据：`sudo -u crypto bash deployment/scripts/refresh-data.sh`
   - 手动训练：`sudo -u crypto bash deployment/scripts/run-train-cycle.sh`
   - 跟日志：`bash deployment/scripts/tail-logs.sh live`
   - 回滚模型：`ln -sfn artifacts/models/<old> .../current`（步骤详列）。
3. **切换 shadow ↔ live**：编辑 `.env` 的 `CE_RUN_MODE`，无需重启 timer（下个 tick 生效）。
4. **故障排查清单**：timer 未触发 / lock 文件残留 / `current` 指向缺失 / venv 包损坏 / Binance 429 等逐条给出处理方式。
5. **卸载**：`systemctl disable --now crypto-engine-*.timer`，删除 systemd 单元，`rm -rf /var/lib/crypto-engine /var/log/crypto-engine`。

---

## 6. 端到端工作流

```
 ┌──────────────────────┐  每天 00:15 UTC
 │ data-refresh.timer   │───────────────┐
 └──────────────────────┘               ▼
                                download_binance_vision_data.py
                                        │
                                        ▼
                                 build_dataset.py
                                        │
                                        ▼
                                 data/training/*.parquet

 ┌──────────────────────┐  每周一 01:00 UTC
 │ train.timer          │───────────────┐
 └──────────────────────┘               ▼
                                  train_model.py
                                        │
                                        ▼
                              artifacts/models/<ts>/
                                        │
                                        ▼
                              ln -sfn → models/current

 ┌──────────────────────┐  每 5 分钟（对齐 UTC :00/:05/...）
 │ live.timer           │───────────────┐
 └──────────────────────┘               ▼
                         run_live_signal.py  或  run_shadow.py
                                        │（读 models/current）
                                        ▼
                              artifacts/signals/*.jsonl  +  audit/*.jsonl

 人工:  sudo -u crypto bash deployment/scripts/update.sh
        │
        ▼ 版本差异→触发 refresh-data / train；live 下个 tick 自动读新 current
```

---

## 7. 代码更新流程（详解）

1. 开发者在本机合并 PR 到 `main`。
2. 服务器上：
   ```bash
   rtk sudo -u crypto bash /opt/crypto-engine/deployment/scripts/update.sh
   ```
3. `update.sh` 的决策矩阵：

| 变更文件 | 操作 |
|---|---|
| `deployment/**` | 复制 systemd 单元 + `daemon-reload`；不碰数据/模型 |
| `requirements.txt` / `pyproject.toml` | `pip install` |
| `src/core/versioning.py` (feature/label 版本↑) | `refresh-data.sh` → `run-train-cycle.sh` |
| `config/settings.yaml` 的 model/signal 段 | `run-train-cycle.sh` |
| 其他 `src/**` | 仅等下一个 live tick 生效 |

4. 若训练失败，`update.sh` **不替换 `current`**（因为 `run-train-cycle.sh` 只有校验通过才 `ln -sfn`），live 继续用老模型，同时 `status.sh` 会显示 train FAIL。
5. 回滚：`ls artifacts/models/` 取上一版本目录，手动 `ln -sfn <old> artifacts/models/current`，无需重启服务。

---

## 8. 监控与告警

**本次范围内**（最小闭环）：
- `status.sh`：人工巡检，**5 秒读完**整个系统状态。
- `healthcheck.sh --exit-on-fail`：可由 cron 每 10 分钟跑一次，失败时 `|| mail -s ...`（邮件地址由 `.env` 的 `CE_ALERT_EMAIL` 提供）。
- `journalctl` + logrotate：保留 30 天。

**未来扩展锚点**（本次不实现，但结构预留）：
- `state/*.last.json` 格式与 Prometheus node_exporter textfile collector 兼容，后续直接写一个 `.prom` 文件即可接入。
- `audit/*.jsonl` 直接可被 Loki/Vector 采集。

---

## 9. 安全注意事项

- `.env` 可能包含 Polymarket API key，必须 `chmod 600` 且属主为 `root:crypto`；`README.md` 中强调。
- `run_live_signal.py` 在 `CE_RUN_MODE=live` 时才允许真实下单；`_common.sh` 启动 live cycle 前显式打印当前 mode 到日志。
- 所有 systemd 单元启用 `NoNewPrivileges=true` / `ProtectSystem=strict` / `ReadWritePaths` 白名单；不使用 root。
- `update.sh` 只接受来自 `origin/main` 或显式 tag/commit，不从工作区任意分支 reset，避免误部署未审查代码。
- 审计目录 `audit/` 严禁被 `rotate-artifacts.sh` 意外删除；rotate 前对归档目标做 `test -s` 校验。

---

## 10. 实施顺序（建议）

1. `deployment/.env.example` + `README.md`（把约定钉死）。
2. `scripts/_common.sh`（所有其他脚本的基建）。
3. `scripts/refresh-data.sh` + 手动跑通。
4. `scripts/run-train-cycle.sh` + 校验原子发布。
5. `scripts/run-live-cycle.sh` + `shadow` 模式跑通。
6. `systemd/*`（先 `--now` 单个 timer 验证）。
7. `logrotate/crypto-engine` + `monitoring/status.sh` + `healthcheck.sh`。
8. `scripts/update.sh` + `bootstrap.sh`（最后把一次性装机串起来）。

每一步都应有一条 `pytest` 或 `bash -n` 静态检查、以及一次 dry-run（`CE_DATA_DIR` 指向 `/tmp/ce-test`）。

---

## 11. 需要在实施阶段最终确认的开放问题

实施时先运行以下命令核对 CLI 参数，避免脚本里写错：

```bash
rtk python scripts/download_binance_vision_data.py --help
rtk python scripts/build_dataset.py --help
rtk python scripts/train_model.py --help
rtk python scripts/run_live_signal.py --help
rtk python scripts/run_shadow.py --help
```

同时确认：

1. 依赖安装方式（`requirements.txt` vs `pyproject.toml`）。
2. `config/settings.yaml` 里是否已有 `execution.audit.log_path`，若有则以它为准，否则用 `.env` 的 `CE_AUDIT_DIR` 并通过 CLI 传入。
3. `train_model.py` 是否接受部署层拟传的 `--validation-window-days`、`--horizon` 等参数，并确认两阶段插件选择继续由 `config/settings.yaml` 驱动。
4. live 推理脚本是否支持直接从实时交易所拉最近 1m K 线，还是依赖本地 `data/raw/*.parquet`；若后者，`live.timer` 前需要先确保 `refresh-data` 产物在 `$CE_LIVE_STALE_MINUTES` 内（healthcheck 已覆盖，但 `run-live-cycle.sh` 也要显式断言）。

---

**本文档交付物边界**：仅本 `docs/deployment_plan.md`。实施该方案时，请以此为唯一规范创建 `deployment/` 目录及其下全部文件；若实施过程中发现与本文档不一致，优先更新本文档后再改脚本。
