# Local RAG Knowledge Workspace

面向中文 PDF、Markdown、TXT 以及视频/音频文件的本地知识库问答应用。文档通过 OpenAI-compatible Embedding 服务写入 Chroma，问题经过向量检索后交给 OpenAI-compatible Chat 模型生成带 `chunk_id` 引用的回答。

## 核心能力

- 多轮对话：追问会先用对话历史改写成独立问题再检索（「它的优势是什么？」能正确解析指代），回答仍基于原问题与历史生成。
- 对话记录保存在服务端并按用户隔离：左侧列表随时切回历史对话，刷新或换浏览器都不丢。
- 四个页面按职责划分：**问答**、**知识库**、**设置**、**审计**（哈希路由，切换不刷新，问答记录保留）。
- 浏览器拖拽上传，后台增量索引，同名文档安全替换旧切片。
- PDF 文本提取与扫描件 OCR 分流。
- **可选的文档结构化提取**：上传或 CLI 入库时勾选「结构化提取」，把文档转成 Markdown 后再切片，
  保留标题、列表、表格结构。支持两个引擎：**MinerU 在线解析**（真实版面分析，适用 PDF 等复杂版面）
  和 **markitdown 本地转换**（适用自带结构的 docx/pptx/xlsx/html）。引擎在配置弹窗选择。
- 稳定的 `doc_id`、`chunk_id`、页码和章节元数据。
- 结构化文档按 Markdown 标题层级切分，切片不跨章节，且每片都带真实章节路径。
- 转换结果按内容哈希缓存，重新入库不会重复消耗 MinerU 额度。
- 本地 Ollama Embedding 与远程 LLM 可独立配置。
- 网页内直接配置 LLM / Embedding 的请求地址、API Key 与模型名，保存后立即生效，无需修改 `.env`。
- 模型配置按端点分为「本地」和「云端」两组：本地组读取本机 Ollama 的模型目录并显示体积，云端组读取 OpenAI 兼容的 `/models`；两组列表互不混用。
- 真实展示问题编码、向量搜索、答案生成的状态和耗时。
- 答案按 Markdown 渲染成标题、列表、加粗和行内代码，而不是一整段纯文本。
- 回答里的 `[source: doc_id#chunk_id]` 渲染成可点击标签，点击直接跳转到原文对应切片并高亮。
- 展示 Top-K 的 Chroma 距离、来源和完整原始切片文本。
- 页面下拉框可切换向量检索与 BM25 关键词检索，两者共用已入库切片和 Top-K。
- Enter 发送，Shift+Enter 换行。
- 多问答并发；索引写入与问答读取通过进程内读写锁隔离。
- 可选的知识库级和文档级访问控制；权限在向量/BM25 检索前过滤。
- 一个文档可同时属于多个访问组；管理员在「知识库」页弹层勾选（可切「按组」视图看每个组能
  看到哪些文档），**只改元数据、不重新向量化**。
- 无入库权限的用户可以提交文档，进入待审区由管理员审核通过后才入库。
- 每次向量库写入都留审计记录，网页端可查看（仅管理员）。

## 目录

```text
app/                    CLI、Web API、后台任务与业务服务
app/web_static/         原生 HTML/CSS/JavaScript 页面
  index.html            顶栏导航 + 四个页面区块
  styles.css            设计令牌、页面骨架、各页组件
  js/core.js            共享状态、哈希路由、身份与外壳状态
  js/views/*.js         问答 / 知识库 / 设置 / 审计 四个视图
  js/app.js             注册视图并启动路由
chains/                 LLM 和回答 Prompt
rag/                    文档加载、OCR、切片、Chroma 与检索结果
data/raw/               本地知识库文件，不提交真实文档
scripts/                macOS 双击启动/停止脚本
tests/                  不访问真实外部服务的测试
```

主入口是 `app/main.py`，公共业务逻辑集中在 `app/service.py`。模型接入配置在
`app/config.py`（运行时覆盖 + 持久化）和 `app/model_catalog.py`（模型发现）；
索引建库用的 Embedding 模型记录在 `rag/index_manifest.py`。

## 安装

要求 Python 3.10+。本机使用 Conda 环境 `langchain`：

```bash
conda activate langchain
pip install -r requirements.txt
```

创建 `.env` 文件并配置 Chat 服务。Embedding 可以复用 Chat 服务，也可以独立指向本地 Ollama：

```bash
OPENAI_API_KEY=your-chat-key
OPENAI_BASE_URL=https://your-chat-endpoint/v1
LLM_MODEL=your-chat-model

EMBEDDING_MODEL=bge-small-zh:q8_0
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1

DATA_DIR=data/raw
CHROMA_PERSIST_DIR=.chroma

RETRIEVAL_METHOD=hybrid
RETRIEVAL_TOP_K=20
SIMILARITY_THRESHOLD=0.35
CANDIDATE_POOL_SIZE=100

# 可选：配置后「结构化提取」走 MinerU 在线版面分析
MINERU_API_TOKEN=
MINERU_MODEL_VERSION=vlm
STRUCTURIZE_PROVIDER=markitdown
```

也可以在网页右上角的「模型接入配置」里直接填写请求地址、API Key 和模型名。保存后写入
`config/runtime_config.json`（已加入 `.gitignore`，明文保存），并覆盖 `.env` 中的
对应项，重启后依然生效。留空表示沿用 `.env` 默认值；Embedding 的地址/Key 留空时
会自动复用 LLM 的配置。启用权限隔离时，只有管理员能读写该配置。

弹窗内分「对话模型（LLM）」和「向量模型（Embedding）」两组，每组右上角会标明该端点
是**本地**还是**云端**（以主机是否为回环地址判断，如 `127.0.0.1`）。模型名是一个
下拉框，末尾带「自定义…」选项用于手动输入：

- **本地**端点（如本机 Ollama）通过 `/api/tags` 读取模型目录，下拉项会带上磁盘体积
  （如 `bge-m3:q8_0 · 605.2 MB`）。
- **云端**端点通过 OpenAI 兼容的 `/models` 读取，通常是模型 id 列表。
- 端点不可达或不支持列表时，会自动切到「自定义…」的自由输入框并预填当前模型，
  不会把配置卡死。

两组的模型列表各自来自自己的端点，不会混在一起。**更换 Embedding 模型会改变向量空间，
必须用新模型重新入库全部文档**，因此保存前会弹出二次确认，选中与已保存值不同的模型时
该提示也会标红。

每次入库（CLI `ingest` 或网页上传）都会在 Chroma 目录里写入 `index_manifest.json`，
记录这次建库用的 Embedding 模型。弹窗里会显示「当前索引由 X 建立」；一旦它与当前配置
不一致（例如直接改了 `.env` 而没重新入库），页面顶部会出现常驻红色告警，直到重新入库。
升级到本版本之前建立的索引没有这份记录，需要在下一次入库后才会生效。

Chat 端点指向 `opencode.ai` 时，程序会自动附加该网关要求的 `x-opencode-session` 头
（进程内稳定，用于路由和 prompt 缓存）并发送自定义 `User-Agent`；其他 OpenAI 兼容端点
不会收到这两个头。部分网关会拒绝通用 HTTP 库的 User-Agent，因此模型列表请求复用同一个
`User-Agent`。

启用权限隔离时，复制 `config/access_policy.example.json` 为
`config/access_policy.json`，并设置 `KNOWLEDGE_BASE_ID` 和
`ACCESS_POLICY_FILE`。策略中的用户通过 `groups` 关联到知识库和文档的访问组；
文档权限使用索引生成的 `doc_id`。修改策略或新增知识库后，需要重新索引对应文件。
当前一个运行实例通过 `KNOWLEDGE_BASE_ID` 选择一个知识库；多个知识库可以共用 Chroma
集合，但生产环境还需要在认证层提供知识库选择/列表接口，并在服务层把选择值校验为该用户
有权访问的知识库。

账号能否登录由 `config/web_auth.json` 决定：顶层 `secret` 用于签发会话，`users` 里每个
用户名对应一个 `password_hash`（pbkdf2_sha256，可用 `python -c "from app.web_auth import
hash_password; print(hash_password('密码'))"` 生成）。一个账号要**同时**存在于
`access_policy.json` 的 `users` 和 `web_auth.json` 的 `users` 才能登录。账号的创建、密码
重置、所属组分配和删除都在网页「用户」页完成（仅管理员）；账号与密码由部署者自行配置，仓库不提供预置凭据。

视频/音频文件（mp4/mkv/mov/webm/avi、mp3/wav/m4a/aac/flac/ogg）入库时先由 **ffmpeg**
抽取音频，再调一个 OpenAI 兼容的 `audio/transcriptions` 接口（ASR）转写成文字，按时间戳
分段后切片入库——答案引用会带出 `MM:SS` 时间范围。转写服务在网页「设置」页的「语音识别
（视频/音频转写）」里配置（请求地址留空复用对话模型地址，填好「转写模型」即启用）；
未配置模型时上传媒体文件会明确报「语音识别未配置」。转写依赖服务器已安装 `ffmpeg`
（macOS：`brew install ffmpeg`；Linux：`apt install ffmpeg`）。

切换 Embedding 模型后必须使用新的 Chroma 目录重新建库，不要混用不同模型的向量。

## 使用

构建完整索引：

```bash
python -m app.main ingest
```

结构化提取（保留标题、列表、表格）：

```bash
python -m app.main ingest --structurize
```

删除当前数据目录中已经不存在的旧切片：

```bash
python -m app.main ingest --rebuild
```

命令行问答：

```bash
python -m app.main ask "Transformer 为什么使用注意力机制？"
```

启动 Web：

```bash
python -m app.main web
```

访问 <http://127.0.0.1:8000>。macOS 可以双击 `scripts/start_web.command` 启动并用 Google Chrome 打开，双击 `scripts/stop_web.command` 停止。

## 页面

| 路由 | 内容 |
|---|---|
| `#/ask` | 默认落地页。左侧对话记录、中间对话流与检索参数、右侧证据栏（问答过程 / 匹配内容）；三条分隔线都能拖，右下角还能直接切对话模型 |
| `#/library` | 索引统计、上传、处理任务、待审核、已入库文档（标题 / 来源 / 切片 / 切片方式 / 最近入库 / 访问组）；点文档原地打开查看器 |
| `#/settings` | 模型接入配置（LLM / Embedding / MinerU / 语音识别）与索引状态、运行目录 |
| `#/audit` | 向量库写入记录，可按事件类型与操作者筛选（仅管理员） |
| `#/eval` | 检索评估：给问题标出「本该被召回的切片」（仅管理员） |
| `#/users` | 用户管理：建号、改密码、分配所属组、授予/收回管理员权限、删除（仅管理员） |

导航在顶栏，`知识库` 上会显示待审数量角标；`设置`、`审计`、`评估`、`用户` 对非管理员隐藏。

在问答页点答案里的引用会**跳到知识库页**并打开对应文档、定位并高亮那个切片——文档
查看器不再占据问答区。

前端用原生 ES 模块，无构建步骤；服务端已满足条件（`.js` 返回正确的 MIME 类型、
CSP `default-src 'self'` 允许同源模块）。

## 对话与多轮

对话记录存在服务端（`.web_conversations.json`），按用户隔离——登录身份不同就只看到自己的
对话；没有配置访问策略时（开放模式）共用一个 `open-mode` 桶。左侧「新对话」开始新会话，
点历史条目可回看全部轮次，引用仍可点击跳转。

追问的检索走两步：先把「历史 + 新问题」交给 LLM 改写成不依赖上下文的独立问题，再拿它检索；
答题时给模型的仍是**原问题 + 历史**（改写只服务检索）。没有历史时过程栏的「问题改写」
显示「已跳过」。

```
问题改写 → 问题编码 → 向量搜索 → 生成答案
```

- 只有最近 **8 轮**会送给模型（存储上限是 200 轮），避免上下文和成本随对话无限增长
- 送给模型的每一轮答案还会再截断（prompt 里 2000 字、改写用 500 字），长答案不会撑爆上下文
- 每轮存的是切片**引用**而不是正文，所以回看时证据栏只显示用过哪些切片，正文在知识库页查看
- 单条答案截断 8000 字；每个用户保留最近 100 个对话
- 生成答案期间切换对话是安全的：迟到的答案会被丢弃，不会画进别的对话

## 回答与引用

回答用轻量 Markdown 解析后渲染成真实 DOM（标题、有序/无序列表、加粗、行内代码、分隔线），
全部通过 `createElement` + `textContent` 构建，不使用 `innerHTML`，避免模型输出注入。

回答中的 `[source: doc_id#chunk_id]` 会被解析成可点击标签：显示为「文档标题 ¶切片号」，
悬停显示完整的 `doc_id · chunk_id`。点击后打开该文档并跳转到对应切片，目标切片会高亮并
滚动到视口中央。若引用只写到 `doc_id` 而不带 `#chunk_id`，则只打开文档开头。

跳转依赖 `GET /api/documents/:doc_id` 的 `focus` 参数：后端按 `chunk_id` 查出它在文档中的
真实位置，再把该页向前偏移 2 个切片后返回，所以前端不需要假设「切片号等于位置」。

## 文档结构化

PDF 只携带视觉排版，不含「这是二级标题」这类语义信息，所以需要版面分析模型才能拿到真实
结构。项目提供两个引擎，在配置弹窗的「文档解析（结构化提取）」里选：

| 引擎 | 适用场景 | 要求 |
|---|---|---|
| **markitdown** | 文档本身已带结构（docx/pptx/xlsx/html） | 无（本地运行） |
| **MinerU** | PDF 等复杂版面，真正识别标题层级 | 需要 API Token |

> 实测提醒：markitdown 对**学术 PDF** 效果不如项目内置的 PyMuPDF 提取路径（会把标题页
> 识别成表格、把英文单词粘连）。所以处理 PDF 时建议用 MinerU 引擎。

### 配置 MinerU

1. 到 <https://mineru.net/apiManage/docs> 注册并创建 API Token
2. 网页右上角齿轮 → 「文档解析（结构化提取）」→ 
   - 解析引擎选 **MinerU**
   - 填入 Token（也可写进 `.env` 的 `MINERU_API_TOKEN`）
   - 保存后立即生效
3. 上传文档时勾选「结构化提取」；或在 CLI 用 `python -m app.main ingest --structurize`

解析精度可选 `vlm`（默认，精度最高）或 `pipeline`（更快，精度较低），对应
`MINERU_MODEL_VERSION`。

### 调用流程

`rag/mineru_client.py` 走的是官方精准解析 API（异步三步）：

1. `POST /api/v4/file-urls/batch` → 拿到 `batch_id` 和预签名上传地址
2. `PUT <上传地址>` → 上传文件原始字节
3. 轮询 `GET /api/v4/extract-results/batch/{batch_id}` → `state=done` 后取 `full_zip_url`
4. 下载 zip，读取其中的 `full.md`

限制：单文件 ≤ 200MB、≤ 200 页。轮询超时默认 1800 秒。MinerU 失败会**直接报错**而不是静默
退回 markitdown —— 否则会用一个明显更差的解析结果污染索引。

> ⚠️ 文档会上传到 mineru.net 第三方服务，涉密材料不要选 MinerU 引擎。

### 使用 markitdown（本地）

不需要额外配置，在配置弹窗选「markitdown」引擎即可。适合处理 docx/pptx/xlsx/html 等
自带结构的格式。对 PDF 效果较差，不如直接用内置的 PyMuPDF 路径。

## 按文档管理访问权限

管理员在「知识库」页的「已入库文档」表格里，每行访问组列是一个按钮，点开弹层可以给文档
勾选**一个或多个**访问组，改完立即生效。同一个文档可以同时属于多个组（如
`{hr, finance}`），每个组都能检索到它。

关键在于**不需要重新向量化**：访问组本来是在切片时写进每个分片元数据的，所以事后修改通常
意味着重建索引。这里只重写元数据的访问组字段，向量原封不动——实测同一文档改组后，alice
（有 hr）5/5 命中来自该文档，bob（无 hr）0/5。

切片元数据里对每个组存了一个布尔 flag（`access_<组>=true`），检索时用这些 flag 的 `$or`
过滤；同时保留旧的 `document_access_group` 标量字段（主组），兼容从未改过组的存量切片。

修改会同时落到两处：

1. `config/access_policy.json` 的 `documents` 段（`{"groups": ["hr", "finance"]}`）——决定
   **以后**重新索引时用什么组
2. Chroma 里该文档所有分片的元数据——决定**现在**检索怎么过滤

管理员还可以把「已入库文档」面板右上角切到**「按组」**视图：每个访问组一行，列出能看到该组
文档的清单（`public` 自动对所有用户开放，没有文档的组也会显示为空）。

约束：

- 只接受策略里**已存在**的组（`known_groups`），防止拼错把文档变成谁都看不见；至少保留一个
  组，取消最后一个勾选会被拦下
- 需要启用访问控制（存在策略文件）；开放模式下没有过滤，谈不上分组
- 管理员 id **不会**出现在可选组里——它是身份不是组，`user_context` 从不把用户自己的 id
  放进 groups，指派过去会让文档对所有人隐形
- 管理员的文档列表会显示全部文档（便于管理），检索与文档查看也**绕过组过滤**（管理员
  是最高的权限身份）；非管理员仍严格按组过滤。

## 切片策略

两类文档走两条路，由 `structured_by` 元数据决定（只有经过结构化提取的文档才带它）：

| 文档 | 切分方式 | 章节路径 |
|---|---|---|
| 结构化（MinerU） | 先按 Markdown 标题层级分节，节内再按 800 字切 | 真实层级，如 `2 The Skip-gram Model > 2.2 Negative Sampling` |
| 未结构化（PyMuPDF 直抽） | 固定长度 + 分隔符优先级（原有行为） | 从切片内的标题推断 |

**为什么不能一刀切**：未结构化文档里的 `## ` 是 PyMuPDF 用长度启发式造的，日期、正文句子、
公式都会被标成「标题」。按它切只会把句子从中间劈开，检索更差。

「知识库」页的「切片方式」列直接读这个字段：带 `structured_by` 就显示 `标题对齐·<引擎>`，
没有就是 `固定长度`。所以那一列显示的是**入库时实际走的哪条路**，不是当前配置——事后改
`structurize_provider` 不会改变已入库文档的显示。

实测同一篇论文（MinerU 解析）：

| | 切片 | `section_path` 为空 |
|---|---|---|
| 改前 | 61 | **47（77%）** |
| 改后 | 60 | **1**（标题/作者块，本来就不属于任何章节） |

只能靠切片**内部**的标题推断时，77% 的切片根本不知道自己属于哪一节——这正是「先按章节切」
解决的问题。

路径会去掉开头与文档标题重复的那一级：文档标题几乎总是 h1，而证据卡片是右截断显示，
让它占掉每行开头会把真正有用的「2.2 Negative Sampling」挤没。

> **暂未做**：把章节路径拼进 embedding 文本（contextual retrieval，通常比单纯对齐边界提升
> 更大）。它需要让「参与 embedding 的文本」≠「存储的文本」，而 langchain 的 Chroma 包装层
> 没有 `add_embeddings`，得绕到底层 collection；另一种朴素做法（直接拼进正文）会破坏
> 「连续阅读」视图依赖的字符偏移。
>
> 当初还有一个理由是「没有评估集，改完测不出收益」——**这个前提已经不存在了**，见下面的
> 「检索评估」。现在的流程是：先攒一批标注好的问题，跑一份基准快照，再动这块代码，然后
> `eval compare` 两份快照。注意这类改动不改存储文本，所以标注不会失效。

### 参数

`CHUNK_SIZE=800`、`CHUNK_OVERLAP=120`，两类切分共用。未结构化文档优先按 Markdown 标题、
段落、换行和中文标点切分。

不同文档场景应通过真实问题集评估 `Recall@K`、MRR、引用正确率和拒答准确率后再调整，
不应只凭经验更换参数。

## Markdown 缓存

结构化提取是最贵的一步（MinerU 是远程计费 API）。转换结果按 **文件内容 + 引擎 + 模型版本**
的 sha256 缓存到 `.markdown-cache/`：

- 改了文件内容 → 自动失效（不需要比对 mtime）
- 换 `pipeline` ↔ `vlm` → 重新解析（输出确实不同）
- 重命名文件 → 仍然命中（内容没变）
- 产物为空 → 不缓存，避免挡住重试

实测：重新入库同一篇文档，第二次**零 MinerU 调用**（缓存文件时间戳未变）。

**覆盖不到的情况**：轮换了 MinerU Token，或 MinerU 自己升级了模型——键不变会拿到旧结果。
删掉 `.markdown-cache/` 即可强制重解析；也可以设 `MARKDOWN_CACHE_DISABLED=1` 临时关闭。
缓存目录可在「设置 → 索引状态」里看到（含条目数和占用）。

## 写入审计

每一次对向量库的写入都会追加一条记录到 `data/vector_audit.jsonl`（JSONL，已 gitignore），
覆盖三条写入路径：

| 事件 | 触发方式 |
|---|---|
| `ingest.directory` | CLI `python -m app.main ingest` |
| `ingest.files` | 网页上传（或审核通过后的实际入库） |
| `delete.documents` | 网页删除文档 |
| `staging.submitted` | 无入库权限的用户提交待审 |
| `staging.approved` / `staging.rejected` | 管理员审核决定 |
| `access.change` | 管理员修改某个文档的访问组（记录 `access_groups`） |

每条记录包含**微秒精度的本地时间带时区偏移**（如 `2026-09-12T15:37:01.630276+08:00`，
既可排序又无需换算）、操作者、知识库、来源文件、`doc_ids`、文档/切片数、以及当时的
向量模型与结构化引擎。删除文档时还会在真正删除前快照 `deleted_documents`（标题、
`doc_id`、来源、切片数），所以审计页能显示「删的是哪篇文档」，而不是只显示操作者与
切片数。历史日志没有这份标题快照，文档又已经不存在，无法事后还原标题——页面会诚实
回退到当时已记录的 `doc_id`，不伪造历史信息：

```json
{"timestamp": "2026-09-12T15:37:01.630276+08:00", "event": "ingest.files",
 "status": "success", "actor": "alice", "knowledge_base_id": "default",
 "files": ["xxx.pdf"], "doc_ids": ["xxx_edcb415d"],
 "documents": 1, "chunks": 61, "structurize": true,
 "structurize_provider": "mineru", "embedding_model": "bge-small-zh:q8_0"}
```

新记录的 `status` 是 `success` / `failed`；历史日志没有该字段，按成功兼容。上传或删除
失败时还记录 `failure_stage`、异常类型与经过**安全清洗**的失败原因：不含 traceback，
并会遮掉 Bearer、API Key、Token、Secret 与预签名 URL 查询参数，避免把凭据写进永久
只追加的日志。原异常仍按原路径返回给上传任务/网页，审计不会吞掉它。

审计页同时区分两个维度：**上传蓝色、删除橙色；成功绿色徽章、失败红色徽章**，旁边
显示失败阶段与原因，并保留文字标签，不只靠颜色传达信息。

能记录的是**请求已经到达服务器**后的失败（请求过大、文件类型不支持、队列满、
MinerU/OCR/切片/向量写入失败、文档不存在、Chroma 删除异常等）。浏览器在请求到达
服务器之前就断网，后端根本没收到操作，无法写服务端审计——这种情况仍只在前端显示
「网络中断」。

日志是**只追加**的，从不清空重写；审计文件本身写入失败只会记一条 warning，不会反过来
让原本成功的入库/删除失败。读取时会跳过崩溃造成的半行（torn line）。

操作者来自 Web 会话的登录用户名；CLI 入库记为 `cli`；未启用权限策略时记为 `open-mode`。

日志文件路径可用 `VECTOR_AUDIT_LOG` 覆盖。

管理员也可以在顶栏「审计」页查看（最新 200 条），并按事件类型与操作者筛选。

「知识库」页的表格也把这份日志用起来了：每行显示该文档的**切片方式**和**最近入库**
（`操作者 · 月-日 时:分`，悬停看完整时间），后者只对管理员显示——审计页本身就是仅
管理员可见。两个细节：

- **「最近」是字面意思**：日志按 `doc_id` 取最新一条 `ingest.*` 记录，同名文档重新
  入库会覆盖它，它不代表这份文档最初是谁放进来的
- 日志建立**之前**入库的文档没有记录，显示「—」。这个信息事后找不回来：chunk 元数据
  里的 `created_at` 是从没填过的空占位，上传任务队列也只留在临时文件里且不带操作者。
  重新入库能生成新记录，但那记的是重入的时间，不是原件

## 写入权限与审核

启用权限策略后，上传分两条路径：

| 身份 | 行为 |
|---|---|
| 管理员 | 直接入库（原有行为） |
| 其他已认证用户 | 提交到**待审区**，不切片、不落库，等管理员决定 |

待审文件存在 `data/pending/<item_id>/`，元数据在 `.web_staging.json`。

管理员在「知识库」页的「待审核」面板逐项处理：选择一个**访问组**后点「通过」，或直接「拒绝」。

**访问组必须在入库之前写入策略**——因为切片时就把 `document_access_group` 写进了 chunk
metadata，事后改策略需要重新索引。所以「通过」的动作顺序是：

1. 按源文件路径算出 `doc_id`（`build_doc_id` 由路径确定性推导）
2. 把 `documents[doc_id]` 写入 `config/access_policy.json`
3. 把文件交给正常的上传队列入库

这样新文档对指定的组可见，其他组检索不到。

> 注意：待审区的文件是**明文暂存**在磁盘上的，提交者与审核者都能看到；涉密材料不要走这条路。

## Web API

- `GET /api/status`：运行配置、索引状态，以及建库模型和是否与当前配置不一致。
- `GET /api/knowledge-bases`：当前知识库、它的访问组、调用者是否属于该组，以及策略里
  定义了但**本实例不服务**的其它知识库。注意一个实例只服务一个知识库（`KNOWLEDGE_BASE_ID`）。
- `GET /api/audit`：向量库写入审计日志，`limit` 默认 100（1-1000），返回最新在前。仅管理员。
- `GET /api/eval`：评估问题集与标注（标注已重新锚定到当前索引，失效的会带出来）。仅管理员。
- `POST /api/eval/import`：从对话记录导入**首轮**问题（幂等，不覆盖已有标注）。仅管理员。
- `POST /api/eval/retrieve`：看一道题召回了什么，接受与问答相同的检索参数。仅管理员。
- `POST /api/eval/label`：写一条标注；`unresolved` 字段回传索引里找不到的 `chunk_id`。仅管理员。
- `POST /api/eval/run`：跑批并返回汇总指标，`ks` 为 K 列表。只回检索结果 id、不回正文。仅管理员。
- `GET /api/staging`：待审提交列表和可用的访问组。仅管理员。
- `POST /api/staging/:id/approve`：批准入库，`group` 指定访问组（须是策略里已有的组）。仅管理员。
- `POST /api/staging/:id/reject`：拒绝并删除暂存文件。仅管理员。
- `GET /api/users`：用户列表（策略用户 ∪ 认证用户），每项含 `username` / `groups` /
  `is_admin` / `has_password` / `in_policy`。仅管理员。
- `POST /api/users`：创建账号 `{username, password, groups[], is_admin}`，同时写入
  `access_policy.json` 与 `web_auth.json`；`web_auth.json` 缺失时自动生成 `secret`。仅管理员。
- `POST /api/users/:username`：更新 `{groups?, is_admin?, password?}`（按出现的键部分更新），
  或 `{clear_password: true}` 清除密码禁止登录；不能取消自己的管理员权限、不能移除最后一位
  管理员。仅管理员。
- `DELETE /api/users/:username`：删除账号（两个文件里同时移除）；不能删除自己、不能删除最后
  一位管理员。仅管理员。
- `GET /api/config`：当前模型接入配置（API Key 不回显，只给 `••••` 加尾 4 位）和索引建库
  模型。仅管理员。
- `POST /api/config`：更新 LLM / Embedding 的请求地址、API Key 与模型名，以及 MinerU 的
  `mineru_api_token` / `mineru_model_version` 和结构化引擎 `structurize_provider`
  （`markitdown` 或 `mineru`），写入 `config/runtime_config.json`。**省略**的字段保持不变；
  API Key 和模型名留空也保持不变；Embedding 的地址留空表示复用 LLM 的地址，LLM 的地址留空
  表示回退到 SDK 默认端点。仅管理员。
- `POST /api/models`：按 `target`（`llm` / `embedding`）列出该端点可用模型，返回
  `{scope, models, error}`；`scope` 为 `local` 时模型带磁盘体积。仅管理员。
- `GET /api/documents`：已入库文档、切片数与切片方式；管理员还会拿到该文档**最近一次**
  入库的人和时间。
- `GET /api/documents/:doc_id`：文档切片，支持 `offset`、`limit` 和 `focus`（定位到某个
  `chunk_id`）。
- `DELETE /api/documents/:doc_id`：删除该文档的全部切片，不影响源文件。仅管理员。
- `POST /api/documents/:doc_id/access`：改该文档的访问组（`{"groups": ["hr", "finance"]}`，
  兼容旧的 `{"group": "hr"}` 单组写法）。仅管理员，且只接受策略里已存在的组。
- `GET /api/upload-jobs`：上传索引任务。仅管理员。
- `POST /api/documents/upload`：上传 `files`，支持 PDF、Markdown、TXT 与视频/音频（视频/音频自动转写语音）。管理员直接入库；其他用户返回 `staged: true` 并进入待审区。
- `POST /api/answer-jobs`：创建异步问答任务，可带 `conversation_id`（会校验归属）。
- `GET /api/conversations`：当前用户的对话列表（标题、时间、轮数）。
- `POST /api/conversations`：新建空对话。
- `GET /api/conversations/:id`：含全部轮次；跨用户返回 403。
- `DELETE /api/conversations/:id`：删除；跨用户返回 403。
- `GET /api/answer-jobs/:id`：查看编码、检索、生成进度和向量命中。
- `POST /api/ask`：同步问答兼容接口。

三个检索方式：

- `vector`：Chroma 向量检索，直接取 Top-K，应用相似度阈值。
- `bm25`：jieba + SQLite FTS5 BM25 全库关键词检索，仅返回包含查询词的结果，展示原始 BM25 分数，不应用阈值。
- `hybrid`（默认）：向量先粗筛候选池（`candidate_pool`，20-500，默认 100，运行时不低于 Top-K），BM25 对候选重排，RRF（k=60）融合后取 Top-K。结果带相似度和 BM25 双来源分，阈值不参与过滤。

两个问答接口均接受 `retrieval_method`（`vector` / `bm25` / `hybrid`）、`top_k`（1-100）、`candidate_pool`（20-500，仅混合模式）及 `similarity_threshold`（0-1，仅向量模式）；省略时使用 `.env` 默认值。网页设置只作用于后续请求，不修改 `.env`。

BM25 使用 jieba 中文分词和 SQLite FTS5 内置评分，直接读取 Chroma 已入库切片，检索时不调用 Embedding 服务。仅返回包含查询词的结果，展示原始 BM25 分数（越高越相关，不是 0-1 相似度），不应用向量相似度阈值。答案生成仍使用 Chat 模型。Python 的 SQLite 需支持 FTS5。

当前每次 BM25 查询都会从最新切片构建临时全文索引，避免文档更新后命中旧内容；大规模知识库应改为持久化全文索引。

上传限制由 `MAX_UPLOAD_BYTES`、`MAX_UPLOAD_FILES`、`MAX_UPLOAD_JOBS` 控制。没有
`ACCESS_POLICY_FILE` 时保持单用户开发模式；启用策略后，Web 登录走 `rag_session` 会话
cookie（`config/web_auth.json` 校验密码并签发，12 小时有效），问答、命中文档列表和异步
任务都以会话里的登录用户做权限校验，上传只允许管理员。生产部署应由反向代理（见
`deploy/nginx-rag.conf`）终止 TLS 后再转发给本服务，账号和密码由「用户」页管理。

## 检索评估

「切片策略」那节里搁置的 contextual retrieval，当时给的理由是「项目还没有评估集，改完
测不出收益」。这套东西就是来解开它的：先能量出可比较的数字，之后任何检索侧改动
（切片参数、融合权重、上下文检索）才有资格谈「该不该做」。

### 用法

标注是唯一需要人力的部分，所以流程围绕它设计：

```bash
# 1. 从真实问过的问题起步（只取首轮；追问依赖上下文，原样导入等于测一个断章问题）
python -m app.main eval import-conversations

# 2. 看清一道题召回了什么——不生成答案，所以不花一次 LLM 调用的钱
python -m app.main retrieve "GloVe 和 word2vec 的区别" --method hybrid --top-k 10 --user-id alice

# 3. 把判相关的 chunk_id 粘过来（正文指纹由索引现取，不用手算）
python -m app.main eval label "GloVe 和 word2vec 的区别" --relevant "glove_..._897f2aa3#chunk_089"

# 4. 跑批看指标
python -m app.main eval run --k 1,3,5 --user-id alice
```

也可以直接在网页 `#/eval` 里做第 2、3 步：展开一道题就能看到召回卡片，勾选保存即可。

### 判断一个改动有没有用

要验的那类改动（比如「把章节路径拼进 embedding 文本」）是**代码改动**而不是参数，
所以用快照而不是运行时开关：

```bash
python -m app.main eval run --save reports/改前.json
# 改代码
python -m app.main eval run --save reports/改后.json
python -m app.main eval compare reports/改前.json reports/改后.json
```

`compare` 的**第一件事是比指纹**，因为两次跑之间索引重建过、或有人改了标注，指标差异
就跟你要验的改动毫无关系。语料指纹或标注指纹变了会**直接拒绝对比**（退出码 1）并说清
是哪一项；确认过原因仍要比才加 `--force`（仍会提醒谨慎）。配置不同只警告不阻断——
对比两套参数本来就是正常用法。

头条是**变好 / 变差 / 不变的题数**而不是均值：题目少的时候，一个均值会被单题带飞，
看不出真实分布。

### 指标怎么读

以 **`Hit@K` 与 `MRR`** 为主，`Recall@K` 只是**下界**：人工标注必然标不全，只从一套
系统的结果里标注（pooling）会让 Recall 系统性偏低——另一套召回了标注者没见过的好
切片，却会被记成漏，**对它系统性不利**。要做公平对比，标注池得取两套系统结果的并集。
没有做 nDCG：二值标签加上标注不完整，它相对 MRR 几乎没有增量。

所有指标一律只看**排名**：三种检索方式的分数根本不可比（向量是距离换算的相似度、
BM25 无上界、hybrid 是 RRF 融合分），而且 hybrid 分支压根忽略 `similarity_threshold`。

**未标注的题会被剔除，不是当成 0 分**——「没标」和「一条都没召回对」是两回事。

### 标注为什么会失效（以及怎么处理）

`chunk_id` 是按顺序编的号（`#chunk_089` 的意思是「第 89 片」，不是「这段文字」），
而 `doc_id` 由源文件路径派生。所以改 `CHUNK_SIZE` 重建索引、甚至只是重命名源文件之后，
同一个 `chunk_id` 就指向别的正文了——**标签会静静地失效、指标却照跑**。

所以每条标注都存了**正文指纹**，跑批前重新锚定：编号还在且指纹一致就用它；编号找
不到、或指纹对不上（编号还在但内容换了）就按 `(doc_id, 正文指纹)` 重新定位；还找不到
就标为失效、**从指标里剔除**（绝不当作「没召回」，否则索引一变指标就无故变差）。
报告里会列出是哪几道题的哪些 id 失效了，重新标一次即可。

### 一个现实限制

评估的区分度取决于语料规模。当前索引只有 3 个文档、20 多个切片，`top_k=20` 已经接近
全集，`Recall@20` 恒为 1——所以现阶段用 `--k 1,3,5`，而且重点是把口径与工具跑通，
真正有意义的用法是「同一批题、两份快照的 diff」。要让绝对数字有意义，得先有一个像样
的语料。

```bash
# 同一个问题换参数跑两次，就能看出差异
python -m app.main retrieve "问题" --method hybrid --top-k 10 --user-id alice
python -m app.main retrieve "问题" --method bm25   --top-k 10 --user-id alice
```

## 检查

```bash
pytest -q
```

测试覆盖 PDF/OCR 分流、切片元数据、向量 ID、安全重建、独立 Embedding 端点、并发问答、
上传边界、原始命中证据、Web 请求安全、运行时配置读写、模型目录（本地体积 / 云端列表）、
索引 manifest、切片定位、网关请求头、文档结构化（markitdown / MinerU 显式选择）、MinerU
API 流程、写入审计，以及检索评估（指标边界、标注的重新锚定、快照可比性、评估端点权限）。
测试不访问真实外部服务。

## 运行数据

以下内容不会提交到 Git：

- `.env`
- `config/runtime_config.json`（网页里保存的模型配置，**明文含 API Key**）
- `config/access_policy.json`
- `.chroma*/`（含索引目录内的 `index_manifest.json`）
- `data/raw/` 中的真实文档
- `.web_upload_jobs.json`、`.web_conversations.json`
- `.markdown-cache/`（转换后的 Markdown，删了会自动重建）
- `data/vector_audit.jsonl`（向量库写入审计日志）
- `data/retrieval_eval.jsonl`（评估问题集与人工标注——只在生成它的那份索引下有含义）
- `reports/`（评估跑批快照）
- `.web_staging.json`
- `data/pending/`（待审核上传的暂存文件）
- `output/`
- `.playwright-cli/`
- 虚拟环境和 Python 缓存
