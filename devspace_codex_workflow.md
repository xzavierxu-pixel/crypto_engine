# DevSpace + ChatGPT + 本地 Codex 协作工作流

本文档按 <https://github.com/Waishnav/devspace> 的方式配置 DevSpace。DevSpace 自己运行本地 MCP server；公网 HTTPS 隧道只负责把 ChatGPT 的请求转发到本地 `127.0.0.1:7676`。

核心分工：

```text
本地 Codex Goal 模式 = 任务主控、实现、测试、提交
网页版 ChatGPT       = 通过 DevSpace MCP 读代码、做方案、做审查
DevSpace             = 本地 workspace MCP server
Cloudflare tunnel    = 公网 HTTPS -> http://127.0.0.1:7676
```

## 当前项目目标

当前主要优化目标：

```text
Optimize validation-set sum_pnl.
```

`selection_score`、coverage、accepted accuracy、signal count 仍作为诊断和风险控制指标，不能用 PnL 改进掩盖泄漏、离线/在线不一致或报告缺失。

## 本地 DevSpace 状态

当前 DevSpace 本地服务：

```text
Local MCP URL: http://127.0.0.1:7676/mcp
Allowed root: C:\Users\ROG\Desktop
Project: C:\Users\ROG\Desktop\crypto_engine_version1
```

当前 Cloudflare quick tunnel：

```text
Public base URL: https://minority-inn-nursery-chest.trycloudflare.com
Public MCP URL:  https://minority-inn-nursery-chest.trycloudflare.com/mcp
```

quick tunnel URL 是临时地址。重启 `cloudflared tunnel --url ...` 后如果 URL 变化，必须重新执行：

```powershell
npx -y @waishnav/devspace config set publicBaseUrl <new_https_url>
npx -y @waishnav/devspace doctor
```

并重启 DevSpace server。

## 正确配置方式

不要使用 OpenAI Secure MCP Tunnel 连接 DevSpace。Waishnav/devspace 的正确模型是：

```text
ChatGPT -> public HTTPS URL -> cloudflared/ngrok -> local DevSpace server
```

不要在 ChatGPT 里填：

```text
http://127.0.0.1:7676/mcp
```

应该填：

```text
https://<public-host>/mcp
```

当前应填：

```text
https://minority-inn-nursery-chest.trycloudflare.com/mcp
```

## 启动顺序

1. 启动 DevSpace server：

```powershell
npx -y @waishnav/devspace serve
```

2. 启动 Cloudflare quick tunnel：

```powershell
C:\tools\cloudflared\cloudflared.exe tunnel --url http://127.0.0.1:7676 --no-autoupdate
```

3. 从 cloudflared 日志里复制 `https://*.trycloudflare.com` URL。

4. 写入 DevSpace public base URL：

```powershell
npx -y @waishnav/devspace config set publicBaseUrl https://<your-trycloudflare-host>
```

5. 重启 DevSpace server，让 allowed host 和 OAuth metadata 生效。

6. 验证：

```powershell
npx -y @waishnav/devspace doctor
curl.exe -i https://<your-trycloudflare-host>/.well-known/oauth-protected-resource/mcp
```

成功时 `doctor` 应显示：

```text
Local MCP URL: http://127.0.0.1:7676/mcp
Public MCP URL: https://<your-trycloudflare-host>/mcp
Allowed hosts: ..., <your-trycloudflare-host>
```

OAuth metadata 应返回 200，并包含：

```json
{
  "resource": "https://<your-trycloudflare-host>/mcp",
  "resource_name": "DevSpace"
}
```

## 网页 ChatGPT 配置

在 ChatGPT 网页端：

```text
Settings -> Apps & Connectors -> Advanced settings
Enable Developer mode
Apps & Connectors -> Create
Name: DevSpace - crypto_engine
Connection / MCP Server URL: https://<public-host>/mcp
Authenticate / Authorize
Scan tools
Create
```

当前 URL：

```text
https://minority-inn-nursery-chest.trycloudflare.com/mcp
```

DevSpace 会要求 owner password approval，这是正常的。按网页授权流程完成后，再 scan tools。

## 推荐职责边界

| 角色 | 适合做什么 | 不适合做什么 |
| --- | --- | --- |
| 本地 Codex Goal 模式 | 拆任务、读本地代码、调用 ChatGPT 审查、改文件、跑测试、跑训练、保存报告、提交 | 把实现责任完全交给网页端 |
| 网页版 ChatGPT + DevSpace | 阅读指定代码、提出方案、找风险、审查 diff、检查实验结论 | 直接修改本地文件、替代本地验证 |
| DevSpace MCP | 提供本地 workspace 文件、搜索、shell、diff 上下文 | 暴露 secrets 或无边界开放整个机器 |

## 本地 Goal 模式推荐写法

后续使用 Goal 命令时，可以只写任务目标。项目 `AGENTS.md` 已默认要求：

```text
网页 ChatGPT 通过 DevSpace 做读代码、方案、审查。
本地 Codex 做实现、测试、实验和提交。
```

如果需要显式写 Goal：

```text
Goal：优化 validation set 上的 sum_pnl。

要求：
- 本地 Codex 作为主控，先读本地代码和项目约束
- 必要时让网页 ChatGPT 通过 DevSpace 做读代码、方案或审查
- 本地 Codex 自己完成实现、测试、报告和 git 检查
- 网页端输出只能作为分析意见，不能替代本地验证
```

## 给网页 ChatGPT 的首次测试 Prompt

```text
Use DevSpace - crypto_engine.

Open the available workspace and inspect:
- AGENTS.md
- README.md
- git status
- project structure
- test setup

Do not edit any files.
Explain the architecture, relevant modules, and a concrete implementation plan for optimizing validation-set sum_pnl.
```

## 读代码 Prompt

```text
你是这个项目的代码阅读协作者。请只阅读和分析我提供或 DevSpace 可访问的相关文件，不要修改文件。

目标：理解 <模块/函数/流程> 是否符合项目约束。

重点检查：
- 是否使用 polymarket_resolved 标签
- 是否存在未来信息或 label-derived feature
- 是否保持 offline/online 一致
- sum_pnl 计算是否来自 validation set
- threshold、coverage、label、horizon 是否来自配置或 artifact
- report 是否包含必要指标

输出：
1. 代码路径和职责总结
2. 潜在问题
3. 推荐本地 Codex 验证的点
```

## 方案 Prompt

```text
你是方案设计协作者。本地 Codex 会负责最终实现和测试。

目标：为优化 validation-set sum_pnl 设计一个小步、低风险、可验证的方案。

必须包含：
- metric being improved: validation sum_pnl
- files affected
- reason it may improve sum_pnl
- how leakage and offline/online consistency are preserved
- diagnostic metrics to keep reporting
- tests or reports to verify it
- rollback plan

限制：
- 不新增复杂模型，除非必要且隔离
- 不改变 label/horizon/timestamp alignment/feature semantics
- 不把 full-train in-sample 指标当作 validation 改进
- 不建议硬编码阈值或业务参数
```

## 审查 Prompt

```text
你是 review 协作者。请按 bug/risk/test gap 优先级审查，不要只给风格建议。

输入：
- git diff
- 相关测试输出
- validation sum_pnl 指标
- coverage / accepted accuracy / signal count 诊断

重点判断：
- 是否有泄漏风险
- 是否破坏 shared core 单一事实源
- 是否破坏 offline/online 一致性
- sum_pnl 是否来自 validation set
- required metrics 是否完整
- 实验结论是否真实对比 validation baseline

输出顺序：
1. 阻断问题
2. 中等风险问题
3. 测试缺口
4. 是否可以继续跑实验/提交
```

## 安全边界

不要传给网页端：

- `execution_engine/secrets.env`
- API key、私钥、助记词、交易凭据
- CLOB secret/passphrase
- 未脱敏的账户地址和资金信息
- 不相关的大型数据文件

可以传：

- 相关源码片段
- `git diff`
- 测试输出
- report 指标摘要
- config 中非敏感的业务参数

## 常见问题

- `Invalid Host: <public-host>`：DevSpace server 没有加载新的 `publicBaseUrl`，执行 `config set publicBaseUrl` 后重启 `devspace serve`。
- 网页端无法访问 localhost：正常，ChatGPT 只能用公网 HTTPS URL。
- quick tunnel URL 变了：重新设置 `publicBaseUrl` 并重启 DevSpace。
- OAuth/owner approval：正常，按 ChatGPT 连接器授权流程完成。
- `connection refused 127.0.0.1:7676`：DevSpace server 没运行。

## 一句话总结

```text
Waishnav/devspace 的正确接法是：ChatGPT 连接公网 HTTPS /mcp，公网隧道转发到本地 DevSpace；本地 Codex 继续负责实现和测试。
```
