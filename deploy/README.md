# 部署到公网（域名 + Caddy + 每人独立账号）

把本机自用的 RAG 服务变成公网可访问、每人一个账号的形态。

架构很简单：**Caddy 在 443 上收请求、做认证，把用户名注入 `X-User-ID` 头，
转发给只听 127.0.0.1 的后端。**

```
公网 → Caddy :443（TLS + basic_auth）
         │  注入 X-User-ID: <认证出的用户名>
         ↓
       后端 :8000（只绑回环，信任这个头）
```

> ⚠️ **后端信任 `X-User-ID`，没有任何密码学保证。** 所以「后端只听 127.0.0.1」
> 是整套方案里承重的一环，不是可选优化。别把 `--host` 改成 `0.0.0.0`。

---

## 0. 先确认能不能做（硬前提，别跳过）

这一步决定后面所有配置是否成立。**两件事都要查，任何一件不通过就得换出口方案。**

### 0.1 拿真实 WAN IP 时先绕开系统代理

本机开着 Clash Verge（系统代理 `127.0.0.1:7897`），而 macOS 自带的 `/usr/bin/curl`
会读系统代理设置——直接 curl 拿到的是**代理节点的 IP**，不是你的宽带出口。

```bash
scutil --proxy | grep -E 'HTTPEnable|HTTPProxy|HTTPPort'   # 确认代理开着
curl -sS --noproxy '*' https://icanhazip.com               # 必须加 --noproxy '*'
```

### 0.2 是不是 CGNAT

和路由器 WAN 口拿到的 IP 对比：

- 两者**相同** → 有公网 IP ✅
- 路由器 WAN 是 `100.64.x.x` ~ `100.127.x.x` → **CGNAT，端口转发无效** ❌
- 不同但都不是 100.64 段 → 可能多层 NAT，要问运营商

### 0.3 运营商有没有封 80/443 入站

**这一条最容易被忽略，而且会直接杀死「Caddy 自动 TLS」。** 国内家宽普遍封禁
80/443 入站。Caddy 的自动签证书有两条挑战路径，都要入站端口：

| 挑战 | 需要 | 被封的后果 |
|---|---|---|
| HTTP-01 | 80 入站 | 签不出证书 |
| TLS-ALPN-01 | 443 入站 | 签不出证书 |

**必须从外网测，局域网内测永远「成功」**（NAT 回环会骗你）。手机**关掉 Wi-Fi 用
蜂窝数据**，或者用任意外部主机：

```bash
nc -vz <你的公网IP> 443
nc -vz <你的公网IP> 80
```

（80 和 443 要**分别**测：443 通而 80 不通是常见情况，那时 Caddy 会自动退到
TLS-ALPN-01，配置不用改。）

### 0.4 三条出路

| 情况 | 方案 | 代价 |
|---|---|---|
| 80/443 都通 | 本文件后续步骤，Caddy 自动 TLS | 无 |
| 只是 80 不通 | 同上，Caddy 会自动用 TLS-ALPN-01 | 无 |
| 443 也不通，但域名 DNS 托管在 Cloudflare/DNSPod/AliDNS | **DNS-01 挑战**：Caddyfile 里加 `tls { dns cloudflare {env.CF_API_TOKEN} }` | 官方 `caddy` 二进制不含 DNS 插件，要用 `xcaddy build --with github.com/caddy-dns/cloudflare` 自编译 |
| 不想折腾端口/备案 | **Tailscale**（本机已装好并登录，还有第二台节点） | 每人的身份由 Tailscale 账号体系给，比 basic_auth 强；但外部人必须先加入你的 tailnet |

> 境内主机对公网提供 80/443 服务涉及 **ICP 备案**，未备案可能被阻断。这是合规
> 问题，不是技术问题——用 Tailscale 或 DNS-01 + 非标端口可以绕开。

### 0.5 这台机器适合当服务器吗

代码这边无所谓，但运行环境有三个硬伤，早点知道比较好：

- **会休眠**（`pmset` 显示 `sleep 1`、`womp 0`），合盖/空闲就断服
- **项目在外接卷 `/Volumes/xiaoPai`**，卷没挂载服务就起不来（`KeepAlive` 会疯狂重试）
- **8GB 内存**，上面还跑着 Chroma、Ollama、Clash、Tailscale；Wi-Fi 也不是稳定链路

真想长期对外，把服务挪到一台常开的机器上更省心（本机 tailnet 里已有一台 `yuhuimac-mini`）。

另外，**Ollama（`127.0.0.1:11434`）必须常开**：检索时的 embedding 全靠它，它挂了
问答就报错（它有自己的 launchd 服务 `com.ollama.xiaopai`，已经满足）。模型也在
外接卷上，所以卷的问题同时影响 Ollama 和本服务。

## 1. 打开代码侧的开关

编辑 `.env`，加两行：

```bash
# 公网必须开：策略文件缺失时拒绝服务，而不是静默退化成「谁都能进」。
# 不加这个开关，删掉 config/access_policy.json 就能让整套权限系统失效。
REQUIRE_ACCESS_POLICY=1

# 可选但推荐：与 Caddyfile 里同一行的值必须逐字相同。
# 开了之后，即使 8000 端口被意外暴露，光伪造 X-User-ID 也进不来。
# 生成：openssl rand -hex 32
TRUSTED_PROXY_SECRET=
```

并确认敏感文件权限（含明文 API Key）：

```bash
chmod 600 .env config/runtime_config.json
```

## 2. 写访问策略

```bash
cp config/access_policy.example.json config/access_policy.json
```

然后按实际情况改三个地方：

| 字段 | 含义 | 注意 |
|---|---|---|
| `admins` | 管理员 user id 列表 | 管理员能改模型配置、审待审文档、看审计日志 |
| `users` | `user_id → {groups}` | **user_id 必须与 Caddy `basic_auth` 里的用户名逐字一致**，否则后端 401 |
| `documents` | `doc_id → {group}` | 想看 `doc_id` 就打开知识库页，或读 `.chroma*/` 里的元数据 |

两条容易踩的语义：

- **`public` 组不是「未登录可见」**，是「每个已认证用户可见」。真正的外部边界是
  「有没有账号」，不是这个组。
- 一个文档要能被看见，需要它的**知识库组**和**文档组**都在该用户的 groups 里
  （交集语义）。`admins` 里的 id **不能**当成组来用。

改完策略后重启后端生效。**用户列表改了，Caddy 的密码文件也要同步改。**

## 3. 配置 Caddy

```bash
brew install caddy
```

生成密码哈希（每加一个人做一次）：

```bash
caddy hash-password --plaintext '你想设的密码'
```

把输出填进 `deploy/Caddyfile` 的 `basic_auth` 段，并把 `rag.example.com` 换成你的域名。
注意哈希里有 `$`，在 shell 里传参要用**单引号**，否则会被展开打坏。

**加人/删人之后，两边必须对上**——用户名写错是这一步最容易漏、症状又最迷惑的错
（登录成功但每个请求都 401）。用这个脚本对账：

```bash
python3 - <<'PY'
import json, re
policy = json.load(open('config/access_policy.json'))
users = set(policy.get('users', {}))
caddy = set(re.findall(r'^\t(\w+)\s+\$', open('deploy/Caddyfile').read(), re.M))
print("策略里有、Caddy 里没有（登不进来）:", sorted(users - caddy) or "无")
print("Caddy 里有、策略里没有（登录后 401 Unknown user）:", sorted(caddy - users) or "无")
PY
```

然后：

```bash
caddy fmt --overwrite deploy/Caddyfile
caddy validate --config deploy/Caddyfile
```

`caddy validate` 只查语法，**查不出占位符对不对**，所以务必做第 6 节的伪造测试。

> `{http.auth.user.id}` 是 Caddy 暴露「basic_auth 认证出的用户名」的占位符（已核对
> 官方文档）。两条相关约束也一并核对过：写代理头必须用 `reverse_proxy` 的
> `header_up`（官方明确说 `request_header` 不是 proxy-aware 的）；`basic_auth` 在
> 指令序里排在 `reverse_proxy` 之前，所以 `header_up` 求值时这个占位符已经就绪。

起服务——**443 是特权端口，必须用 root**：

```bash
sudo brew services start caddy
```

普通的 `brew services start caddy` 以当前用户身份运行，绑不上 443，会起来就退出。

域名 A 记录指向本机公网 IP，路由器把 80 和 443 转发到这台机器，证书会自动签发续期。

**只转发 80/443，不要转发 8000。**

## 4. 把后端装成常驻服务

```bash
cp deploy/com.ragserver.xiaopai.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ragserver.xiaopai.plist
```

检查：

```bash
launchctl list | grep ragserver        # 第一列是 PID 就说明在跑
tail -f ~/Library/Logs/ragserver.err.log
lsof -nP -iTCP:8000 -sTCP:LISTEN       # 应该只见 127.0.0.1:8000
```

改了 plist 要 `launchctl bootout gui/$(id -u)/com.ragserver.xiaopai` 再 `bootstrap`，
launchd 不会自动重载。

## 5. 备份

```bash
./deploy/backup.sh                  # 备份到 ~/rag-backups/<时间戳>/
./deploy/backup.sh --with-index     # 连向量索引一起备（体积大）
```

必须定期跑。其中 **`data/vector_audit.jsonl` 是只追加的审计日志，重建不出来**；
`.env`、`runtime_config.json`、`access_policy.json` 同理。向量索引和 Markdown 缓存
可以重建，但要重新入库全部文档、走 MinerU 的还要花额度。

备份目录含明文密钥，脚本已设成 700——**别放进同步盘或 Git**。

## 6. 验证（逐条做，别跳）

```bash
DOMAIN=rag.example.com
```

1. **匿名被挡**（`-u` 都不给）：
   ```bash
   curl -sS -o /dev/null -w '%{http_code}\n' https://$DOMAIN/api/status
   ```
   期望 `401`。

2. **身份伪造测试（最关键的一条）**：

   ```bash
   # alice 建一个对话
   curl -sS -u alice:密码 -X POST https://$DOMAIN/api/conversations
   # 记下返回的 id，然后 bob 登录、手动伪造 alice 的身份
   curl -sS -u bob:密码 -H 'X-User-ID: alice' https://$DOMAIN/api/conversations
   ```

   期望：**看不到 alice 的那个对话**。若能看到 → `header_up -X-User-ID` 那条剥离没生效。
   若一律 401 → `{http.auth.user.id}` 占位符名不对（身份传成了空）。

3. **后端端口从外面够不到**：外部网络执行 `nc -vz $DOMAIN 8000`，期望拒绝或超时。

4. **TLS 与跳转**：
   ```bash
   curl -vI https://$DOMAIN 2>&1 | grep -E 'subject|issuer|HTTP'
   curl -sI http://$DOMAIN | head -1        # 期望 308
   ```

5. **`/api/status` 字段分级**——用非管理员账号：
   ```bash
   curl -sS -u bob:密码 https://$DOMAIN/api/status | python3 -m json.tool | grep -E 'data_dir|supported_files|markdown_cache'
   ```
   期望三个都是 `null` / `[]` / `{}`（问答页要用的字段仍在）。

6. **共享密钥**（只在启用了它时测）：
   ```bash
   curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/api/status
   ```
   期望 `401`——本机直连也得有密钥。

7. **ACL 生效**：用只有 `finance` 组的账号拉文档列表，不应出现归 `hr` 的文档。

8. **回归**：`pytest -q` 应 226 全绿。

## 7. 回滚

想退回本机自用：

```bash
launchctl bootout gui/$(id -u)/com.ragserver.xiaopai
brew services stop caddy
# 从 .env 里删掉 REQUIRE_ACCESS_POLICY 与 TRUSTED_PROXY_SECRET
python -m app.main web            # 照旧本机跑
```

`scripts/dev_proxy.py` 仍是本地开发用的身份代理，但它**默认把任何人都当成管理员
`alice`**（未知 cookie 会回退到默认用户），所以**绝对不能拿它当生产入口**。

## 8. 仍然存在的风险

这套配置**没有做限流**（你的选择）。已知会一直存在的：

| 风险 | 缓解 |
|---|---|
| 任何已认证用户可无限消耗 LLM token 与 MinerU 额度 | 账号保持少而可信；盯 API 账单。审计日志只记入库，不记问答 |
| **问答队列没有深度上限**：`answer_jobs.py` 无条件把任务丢进 `ThreadPoolExecutor`，`max_concurrent_answers=8` 只限制「同时执行」不限制「排队」。1000 次点击 = 1000 个任务，**每个最终都会打远程 LLM、都会花钱** | 现在是开放的（你选了不限流）。要收口就是在 `create()` 里加一句「排队数 ≥ N 就拒绝」，与既有的 `max_upload_jobs` 写法对称，约 5 行。顺带也能让停机更快（退出会等排队任务跑完） |
| Ollama 是单点，但**不是**瓶颈：它只做 embedding（bge-small，几十毫秒），问答的 LLM 在远程。`OLLAMA_NUM_PARALLEL=4` 与 `max_concurrent_answers=8` 不冲突 | 真正要盯的是远程 API 的配额与延迟。另外别在这台机器上 `ollama run` 别的模型——`OLLAMA_MAX_LOADED_MODELS=1` 会把 embedding 模型挤出去 |
| 直连 8000 就绕过一切（`X-User-ID` 无签名） | 绑回环是主要防线；共享密钥是第二道。两者都别关 |
| **basic_auth 没有「登出」**：HTTP Basic 凭据由浏览器按站点缓存，用户关掉浏览器才会清 | 出事故要踢人时，改那个人的密码并重启 Caddy。要真正的会话管理得换 forward_auth（Authelia / oauth2-proxy）或 Tailscale |
| `public` 组文档对每个已认证用户可见 | 认真设 groups；新入库文档默认进 `public`，要显式指定组 |
| **`documents` 里没写的文档 = 公开，不是拒绝**（`access.py` 的 `document_group` 默认返回 `public`） | 所以 doc_id 打错一个字符的后果是**意外公开**而不是意外隐藏。收口时用知识库页的下拉框改，别手写 JSON |
| basic_auth 单密码、无 2FA、无锁定 | 密码用 ≥16 位随机串；要更强身份换 Tailscale（每人真账号）或 forward_auth |
| 只有变更审计，没有读取审计 | 谁「看」了什么没有记录。要追查得看 Caddy 的 access log |
| 待审文件与上传原件明文落在磁盘上 | 涉密材料不要走这条路 |
| 这台 Mac 不能休眠，外接卷不能弹出 | 电源设置禁休眠；见第 0.5 节 |
