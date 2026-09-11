# Local RAG Knowledge Workspace

面向中文 PDF、Markdown 和 TXT 的本地知识库问答应用。文档通过 OpenAI-compatible Embedding 服务写入 Chroma，问题经过向量检索后交给 OpenAI-compatible Chat 模型生成带 `chunk_id` 引用的回答。

## 核心能力

- 浏览器拖拽上传，后台增量索引，同名文档安全替换旧切片。
- PDF 文本提取与扫描件 OCR 分流。
- 稳定的 `doc_id`、`chunk_id`、页码和章节元数据。
- 本地 Ollama Embedding 与远程 LLM 可独立配置。
- 真实展示问题编码、向量搜索、答案生成的状态和耗时。
- 展示 Top-K 的 Chroma 距离、来源和完整原始切片文本。
- 页面下拉框可切换向量检索与 BM25 关键词检索，两者共用已入库切片和 Top-K。
- Enter 发送，Shift+Enter 换行。
- 多问答并发；索引写入与问答读取通过进程内读写锁隔离。

## 目录

```text
app/                CLI、Web API、后台任务与业务服务
app/web_static/     原生 HTML/CSS/JavaScript 页面
chains/             LLM 和回答 Prompt
rag/                文档加载、OCR、切片、Chroma 与检索结果
data/raw/           本地知识库文件，不提交真实文档
scripts/            macOS 双击启动/停止脚本
tests/              不访问真实外部服务的测试
```

主入口是 `app/main.py`，公共业务逻辑集中在 `app/service.py`。

## 安装

要求 Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
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
```

切换 Embedding 模型后必须使用新的 Chroma 目录重新建库，不要混用不同模型的向量。

## 使用

构建完整索引：

```bash
python -m app.main ingest
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

## Web API

- `GET /api/status`：运行配置和索引状态。
- `GET /api/documents`：已入库文档及切片数。
- `GET /api/upload-jobs`：上传索引任务。
- `POST /api/documents/upload`：上传 `files`，支持 PDF、Markdown、TXT。
- `POST /api/answer-jobs`：创建异步问答任务。
- `GET /api/answer-jobs/:id`：查看编码、检索、生成进度和向量命中。
- `POST /api/ask`：同步问答兼容接口。

三个检索方式：

- `vector`：Chroma 向量检索，直接取 Top-K，应用相似度阈值。
- `bm25`：jieba + SQLite FTS5 BM25 全库关键词检索，仅返回包含查询词的结果，展示原始 BM25 分数，不应用阈值。
- `hybrid`（默认）：向量先粗筛候选池（`candidate_pool`，20-500，默认 100，运行时不低于 Top-K），BM25 对候选重排，RRF（k=60）融合后取 Top-K。结果带相似度和 BM25 双来源分，阈值不参与过滤。

两个问答接口均接受 `retrieval_method`（`vector` / `bm25` / `hybrid`）、`top_k`（1-100）、`candidate_pool`（20-500，仅混合模式）及 `similarity_threshold`（0-1，仅向量模式）；省略时使用 `.env` 默认值。网页设置只作用于后续请求，不修改 `.env`。

BM25 使用 jieba 中文分词和 SQLite FTS5 内置评分，直接读取 Chroma 已入库切片，检索时不调用 Embedding 服务。仅返回包含查询词的结果，展示原始 BM25 分数（越高越相关，不是 0-1 相似度），不应用向量相似度阈值。答案生成仍使用 Chat 模型。Python 的 SQLite 需支持 FTS5。

当前每次 BM25 查询都会从最新切片构建临时全文索引，避免文档更新后命中旧内容；大规模知识库应改为持久化全文索引。

上传限制由 `MAX_UPLOAD_BYTES`、`MAX_UPLOAD_FILES`、`MAX_UPLOAD_JOBS` 控制。Web 默认只监听本机且没有身份认证；暴露到网络前必须增加 TLS、认证和限流。

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

测试覆盖 PDF/OCR 分流、切片元数据、向量 ID、安全重建、独立 Embedding 端点、并发问答、上传边界、原始命中证据和 Web 请求安全。

## 运行数据

以下内容不会提交到 Git：

- `.env`
- `.chroma*/`
- `data/raw/` 中的真实文档
- `.web_upload_jobs.json`
- `output/`
- 虚拟环境和 Python 缓存
