# Polymarket CLOB v2 下单问题排查手册

## 0. 当前现象

你遇到的错误包括：

```text
sig=0 -> maker address not allowed, please use the deposit wallet flow
sig=1/2 -> invalid signature
/order -> the order signer address has to be the address of the API KEY
```

这些错误说明问题大概率不是网络问题，而是 **signer、API key、funder、signature_type、wallet flow 没有对齐**。

---

## 1. 先区分几个地址

### 1.1 Signer address

由 `POLYMARKET_PRIVATE_KEY` 推出来的地址。

```python
from eth_account import Account
import os

signer = Account.from_key(os.getenv("POLYMARKET_PRIVATE_KEY")).address
print("signer =", signer)
```

它的作用：

- 生成 / 派生 `CLOB_API_KEY`
- 签订单
- 对于 v2，下单时 API key owner 必须和 order signer 对齐

---

### 1.2 CLOB API key 对应地址

这三个值：

```env
CLOB_API_KEY=...
CLOB_SECRET=...
CLOB_PASS_PHRASE=...
```

必须由当前 `POLYMARKET_PRIVATE_KEY` 对应的 signer 生成。

如果：

```text
API key owner != order signer
```

就会报：

```text
the order signer address has to be the address of the API KEY
```

修复方式：用当前 private key 重新生成一套 CLOB credentials。

---

### 1.3 Profile 页 API-only address

你在 UI 里看到的：

```text
Do not send funds to this address. This address is for API use only.
```

这个地址 **不是充值地址**。

它可能是旧的 proxy/API address。  
如果走 `POLY_PROXY = 1`，它可能是 `funder` 候选。  
但如果 CLOB 明确提示：

```text
please use the deposit wallet flow
```

那就不要继续把它当作 deposit wallet。

---

### 1.4 Deposit / Add Funds 页面地址

这是入金地址，用于从外部钱包或交易所转 USDC/USDC.e 到 Polymarket。

注意：

- 它是充值入口
- 不一定等于 CLOB v2 下单用的 `funder`
- 不要仅凭它判断 wallet flow

---

### 1.5 Deposit wallet address

这是 `signature_type=3 / POLY_1271` 时要填的 `funder`。

它通常需要通过 relayer / builder flow 确认或推导：

```python
deposit_wallet = relayer.get_expected_deposit_wallet()
print("deposit_wallet =", deposit_wallet)
```

---

## 2. Signature Type 怎么选

| 类型 | ID | 适用场景 | funder 应该填 |
|---|---:|---|---|
| EOA | 0 | 普通独立钱包，自付 gas | EOA 地址 |
| POLY_PROXY | 1 | 老 Polymarket proxy wallet flow | proxy wallet address |
| GNOSIS_SAFE | 2 | Gnosis Safe flow | Safe address |
| POLY_1271 | 3 | Deposit wallet flow / 新 API flow | deposit wallet address |

你目前最关键的错误是：

```text
sig=0 -> maker address not allowed, please use the deposit wallet flow
```

这强烈说明你应该优先排查：

```python
signature_type = 3
funder = DEPOSIT_WALLET_ADDRESS
```

而不是继续试 `0/1/2`。

---

## 3. USDC 和 pUSD 的关系

你在 Polymarket UI 里看到的是 USDC / cash / portfolio 视图。

但在 Polymarket v2 底层，交易抵押品是 **pUSD**。

通常你从 UI deposit USDC/USDC.e 后，Polymarket 的入金流程会把它包装成 pUSD。

所以你一般 **不需要手动把 UI 里的 USDC 转成 pUSD**。

真正需要确认的是：

```text
这笔 pUSD / 交易资金到底在哪个 wallet flow 下？
```

也就是：

- proxy wallet?
- deposit wallet?
- EOA?
- 还是只是 UI portfolio 聚合显示？

UI 里的：

```text
Portfolio $83.01
```

不能直接说明钱在哪个地址。

---

## 4. 最小排查步骤

### Step 1：打印 signer

```python
from eth_account import Account
import os

private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
print("signer =", Account.from_key(private_key).address)
```

---

### Step 2：用同一个 private key 重新生成 CLOB API credentials

```python
import os
from py_clob_client_v2 import ClobClient

client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,
    key=os.getenv("POLYMARKET_PRIVATE_KEY"),
)

creds = client.create_or_derive_api_key()

print("CLOB_API_KEY =", creds.api_key)
print("CLOB_SECRET =", creds.api_secret)
print("CLOB_PASS_PHRASE =", creds.api_passphrase)
```

然后把 `.env` 里的三项全部替换：

```env
CLOB_API_KEY=...
CLOB_SECRET=...
CLOB_PASS_PHRASE=...
```

不要只替换其中一个。

---

### Step 3：确认 deposit wallet address

如果你现在被要求走 deposit wallet flow，就需要确认真正的 deposit wallet address。

推荐用 relayer 推导：

```python
deposit_wallet = relayer.get_expected_deposit_wallet()
print("DEPOSIT_WALLET_ADDRESS =", deposit_wallet)
```

不要把下面两个地址误当成 deposit wallet：

```text
Profile 页 API-only address
Deposit / Add Funds 页面充值地址
```

---

### Step 4：用 signature_type=3 初始化 CLOB v2

```python
import os
from py_clob_client_v2 import ClobClient, ApiCreds, SignatureTypeV2

api_creds = ApiCreds(
    api_key=os.getenv("CLOB_API_KEY"),
    api_secret=os.getenv("CLOB_SECRET"),
    api_passphrase=os.getenv("CLOB_PASS_PHRASE"),
)

client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,
    key=os.getenv("POLYMARKET_PRIVATE_KEY"),
    creds=api_creds,
    signature_type=SignatureTypeV2.POLY_1271,
    funder=os.getenv("DEPOSIT_WALLET_ADDRESS"),
)
```

如果你的 SDK 版本没有 `SignatureTypeV2.POLY_1271`，可以尝试：

```python
signature_type=3
```

---

### Step 5：检查 balance / allowance

如果 type 3 后错误从：

```text
invalid signature
```

变成：

```text
not enough balance
not enough allowance
insufficient balance
```

这通常说明方向对了，剩下是 deposit wallet 的资金或授权问题。

此时需要检查：

- deposit wallet 是否部署
- pUSD 是否在 deposit wallet 下
- deposit wallet 是否 approve 了 CLOB trading contracts
- 是否调用过 balance allowance sync / update

---

## 5. 常见错误解释

### 错误 1

```text
the order signer address has to be the address of the API KEY
```

原因：

```text
生成 CLOB_API_KEY 的 private key
和
下单 client 里的 private key
不是同一个 signer
```

修复：

```text
用当前 POLYMARKET_PRIVATE_KEY 重新 create_or_derive_api_key()
```

---

### 错误 2

```text
maker address not allowed, please use the deposit wallet flow
```

原因：

```text
你在用 signature_type=0 / EOA flow，
但 CLOB v2 要求这个账号走 deposit wallet flow。
```

修复：

```text
使用 signature_type=3 / POLY_1271
funder 使用 deposit wallet address
```

---

### 错误 3

```text
invalid signature
```

可能原因：

```text
signature_type 和 funder 不匹配
用了 proxy address 但 signature_type=3
用了 deposit wallet address 但 signature_type=1
private key 和 API key 不匹配
订单签名方式不是当前 wallet flow 要求的签名方式
```

---

## 6. 建议的最终 `.env` 结构

```env
POLYMARKET_PRIVATE_KEY=0x...

# CLOB credentials: 必须由上面的 private key 生成
CLOB_API_KEY=...
CLOB_SECRET=...
CLOB_PASS_PHRASE=...

# 如果走 deposit wallet flow
POLYMARKET_SIGNATURE_TYPE=3
DEPOSIT_WALLET_ADDRESS=0x...
```

如果你最终确认是 legacy proxy flow，才使用：

```env
POLYMARKET_SIGNATURE_TYPE=1
POLYMARKET_FUNDER=0x你的proxy/API-only地址
```

但基于你当前错误，优先排查 `signature_type=3`。

---

## 7. 最终判断逻辑

```text
如果 sig=0 报：
maker address not allowed, please use the deposit wallet flow

=> 优先使用 signature_type=3 / POLY_1271
=> funder 必须是真正的 deposit wallet address
=> 需要 relayer / builder flow 来确认 wallet、部署、授权
```

```text
如果 type=3 后仍然 invalid signature

=> 检查：
1. DEPOSIT_WALLET_ADDRESS 是否正确
2. CLOB_API_KEY 是否由当前 private key 生成
3. SDK 是否真的是 py-clob-client-v2
4. 是否混用了 v1 参数名或 v1 client
5. 是否需要先部署 deposit wallet / 做 allowance
```

---

## 8. 当前最推荐下一步

1. 打印 signer address
2. 用这个 private key 重新生成 CLOB credentials
3. 使用 relayer 推导 `DEPOSIT_WALLET_ADDRESS`
4. 用 `signature_type=3` + `DEPOSIT_WALLET_ADDRESS` 初始化 CLOB v2
5. 如果报 balance/allowance，再处理 deposit wallet 授权
6. 如果继续 invalid signature，再检查 SDK 版本和 v2 参数是否正确
