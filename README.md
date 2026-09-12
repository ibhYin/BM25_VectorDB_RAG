# Local RAG Knowledge Workspace

面向中文 PDF、Markdown 和 TXT 的本地知识库问答应用。文档通过 OpenAI-compatible Embedding 服务写入 Chroma，问题经过向量检索后交给 OpenAI-compatible Chat 模型生成带 `chunk_id` 引用的回答。

## 核心能力

- 四个页面按职责划分：**问答**、**知识库**、**设置**、**审计**（哈希路由，切换不刷新，问答记录保留）。
- 浏览器拖拽上传，后台增量索引，同名文档安全替换旧切片。
- PDF 文本提取与扫描件 OCR 分流。
- **可选的文档结构化提取**：上传或 CLI 入库时勾选「结构化提取」，把文档转成 Markdown 后再切片，
  保留标题、列表、表格结构。支持两个引擎：**MinerU 在线解析**（真实版面分析，适用 PDF 等复杂版面）
  和 **markitdown 本地转换**（适用自带结构的 docx/pptx/xlsx/html）。引擎在配置弹窗选择。
- 稳定的 `doc_id`、`chunk_id`、页码和章节元数据。
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
| `#/ask` | 默认落地页。对话流 + 检索参数 + 右侧证据栏（问答过程 / 匹配内容） |
| `#/library` | 索引统计、上传、处理任务、待审核、已入库文档；点文档原地打开查看器 |
| `#/settings` | 模型接入配置（LLM / Embedding / MinerU）与索引状态、运行目录 |
| `#/audit` | 向量库写入记录，可按事件类型与操作者筛选（仅管理员） |

导航在顶栏，`知识库` 上会显示待审数量角标，`设置` 与 `审计` 对非管理员隐藏。

在问答页点答案里的引用会**跳到知识库页**并打开对应文档、定位并高亮那个切片——文档
查看器不再占据问答区。

前端用原生 ES 模块，无构建步骤；服务端已满足条件（`.js` 返回正确的 MIME 类型、
CSP `default-src 'self'` 允许同源模块）。

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

每条记录包含**微秒精度的本地时间带时区偏移**（如 `2026-09-12T15:37:01.630276+08:00`，
既可排序又无需换算）、操作者、知识库、来源文件、`doc_ids`、文档/切片数、以及当时的
向量模型与结构化引擎：

```json
{"timestamp": "2026-09-12T15:37:01.630276+08:00", "event": "ingest.files",
 "actor": "alice", "knowledge_base_id": "default",
 "files": ["xxx.pdf"], "doc_ids": ["xxx_edcb415d"],
 "documents": 1, "chunks": 61, "structurize": true,
 "structurize_provider": "mineru", "embedding_model": "bge-small-zh:q8_0"}
```

日志是**只追加**的，从不清空重写；写入失败只会记一条 warning，不会让入库失败。读取时
会跳过崩溃造成的半行（torn line）。

操作者来自 Web 请求的 `X-User-ID`；CLI 入库记为 `cli`；未启用权限策略时记为 `open-mode`。

日志文件路径可用 `VECTOR_AUDIT_LOG` 覆盖。

管理员也可以在顶栏「审计」页查看（最新 200 条），并按事件类型与操作者筛选。

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
- `GET /api/audit`：向量库写入审计日志，`limit` 默认 100（1-1000），返回最新在前。仅管理员。
- `GET /api/staging`：待审提交列表和可用的访问组。仅管理员。
- `POST /api/staging/:id/approve`：批准入库，`group` 指定访问组（须是策略里已有的组）。仅管理员。
- `POST /api/staging/:id/reject`：拒绝并删除暂存文件。仅管理员。
- `GET /api/config`：当前模型接入配置（API Key 不回显，只给 `••••` 加尾 4 位）和索引建库
  模型。仅管理员。
- `POST /api/config`：更新 LLM / Embedding 的请求地址、API Key 与模型名，以及 MinerU 的
  `mineru_api_token` / `mineru_model_version` 和结构化引擎 `structurize_provider`
  （`markitdown` 或 `mineru`），写入 `config/runtime_config.json`。**省略**的字段保持不变；
  API Key 和模型名留空也保持不变；Embedding 的地址留空表示复用 LLM 的地址，LLM 的地址留空
  表示回退到 SDK 默认端点。仅管理员。
- `POST /api/models`：按 `target`（`llm` / `embedding`）列出该端点可用模型，返回
  `{scope, models, error}`；`scope` 为 `local` 时模型带磁盘体积。仅管理员。
- `GET /api/documents`：已入库文档及切片数。
- `GET /api/documents/:doc_id`：文档切片，支持 `offset`、`limit` 和 `focus`（定位到某个
  `chunk_id`）。
- `DELETE /api/documents/:doc_id`：删除该文档的全部切片，不影响源文件。仅管理员。
- `GET /api/upload-jobs`：上传索引任务。仅管理员。
- `POST /api/documents/upload`：上传 `files`，支持 PDF、Markdown、TXT。管理员直接入库；其他用户返回 `staged: true` 并进入待审区。
- `POST /api/answer-jobs`：创建异步问答任务。
- `GET /api/answer-jobs/:id`：查看编码、检索、生成进度和向量命中。
- `POST /api/ask`：同步问答兼容接口。

开发用的 `scripts/dev_proxy.py` 另外提供 `GET /api/me`、`GET /login?user=ID` 和
`GET /logout`，用于在浏览器里切换身份并注入 `X-User-ID`。

三个检索方式：

- `vector`：Chroma 向量检索，直接取 Top-K，应用相似度阈值。
- `bm25`：jieba + SQLite FTS5 BM25 全库关键词检索，仅返回包含查询词的结果，展示原始 BM25 分数，不应用阈值。
- `hybrid`（默认）：向量先粗筛候选池（`candidate_pool`，20-500，默认 100，运行时不低于 Top-K），BM25 对候选重排，RRF（k=60）融合后取 Top-K。结果带相似度和 BM25 双来源分，阈值不参与过滤。

两个问答接口均接受 `retrieval_method`（`vector` / `bm25` / `hybrid`）、`top_k`（1-100）、`candidate_pool`（20-500，仅混合模式）及 `similarity_threshold`（0-1，仅向量模式）；省略时使用 `.env` 默认值。网页设置只作用于后续请求，不修改 `.env`。

BM25 使用 jieba 中文分词和 SQLite FTS5 内置评分，直接读取 Chroma 已入库切片，检索时不调用 Embedding 服务。仅返回包含查询词的结果，展示原始 BM25 分数（越高越相关，不是 0-1 相似度），不应用向量相似度阈值。答案生成仍使用 Chat 模型。Python 的 SQLite 需支持 FTS5。

当前每次 BM25 查询都会从最新切片构建临时全文索引，避免文档更新后命中旧内容；大规模知识库应改为持久化全文索引。

上传限制由 `MAX_UPLOAD_BYTES`、`MAX_UPLOAD_FILES`、`MAX_UPLOAD_JOBS` 控制。没有
`ACCESS_POLICY_FILE` 时保持单用户开发模式；启用策略后，Web API 从上游认证层提供的
`X-User-ID` 读取身份，问答、命中文档列表和异步任务都会做权限校验，上传只允许策略中的
管理员。`X-User-ID` 不是认证机制，暴露到网络前必须由可信反向代理覆盖该请求头，并增加
TLS、真实认证和限流。

## 切片

当前使用 `RecursiveCharacterTextSplitter`：

- `CHUNK_SIZE=800`
- `CHUNK_OVERLAP=120`
- 优先按 Markdown 标题、段落、换行和中文标点切分

不同文档场景应通过真实问题集评估 `Recall@K`、MRR、引用正确率和拒答准确率后再调整，不应只凭经验更换参数。

## 检查

```bash
pytest -q
```

测试覆盖 PDF/OCR 分流、切片元数据、向量 ID、安全重建、独立 Embedding 端点、并发问答、
上传边界、原始命中证据、Web 请求安全、运行时配置读写、模型目录（本地体积 / 云端列表）、
索引 manifest、切片定位、网关请求头、文档结构化（markitdown / MinerU 显式选择）、MinerU
API 流程与写入审计。测试不访问真实外部服务。

## 运行数据

以下内容不会提交到 Git：

- `.env`
- `config/runtime_config.json`（网页里保存的模型配置，**明文含 API Key**）
- `config/access_policy.json`
- `.chroma*/`（含索引目录内的 `index_manifest.json`）
- `data/raw/` 中的真实文档
- `.web_upload_jobs.json`
- `data/vector_audit.jsonl`（向量库写入审计日志）
- `.web_staging.json`
- `data/pending/`（待审核上传的暂存文件）
- `output/`
- `.playwright-cli/`
- 虚拟环境和 Python 缓存
