# RAG + LangChain + LangGraph + Knowledge Graph

面向本地中文技术文档的知识库问答项目。主应用负责在线索引和问答；`KG_RAG` 负责离线批量生产、清洗三元组。两条流水线共享原始文档，但目前不会自动同步数据。

运行环境为 Python 3.10+。项目使用 OpenAI-compatible Chat/Embedding API；Neo4j 是可选增强，Chroma 是向量问答的必需存储。

## 项目地图

```text
app/                    CLI、Web、配置和 LangGraph 工作流
chains/                 回答 Prompt 与 LLM 调用
rag/                    文档加载、PDF/OCR、共享索引、Chroma 检索
kg/                     在线实体关系抽取与 Neo4j 读写
scripts/kg/             离线图谱生产脚本（正式入口）
KG_RAG/libriry/         旧路径兼容入口
data/raw/               原始 PDF、Markdown、TXT
data/kg/                离线抽取的原始图谱
data/kg_aligned/        实体对齐后的图谱
data/triples/           最终三元组导出
tests/                  不访问外部服务的单元测试
```

主应用唯一入口是 [app/main.py](app/main.py)，公共业务逻辑集中在 [app/service.py](app/service.py)。`rag/local_vector_qa` 是无需 Neo4j 的轻量子入口，适合只调试向量检索。

模块依赖保持单向：`app` 编排业务，`rag` 和 `kg` 实现领域能力，`chains` 只负责模型与 Prompt，`scripts/kg` 消费共享 schema 生成离线产物。旧脚本只做命令转发，不再维护第二套实现。

## 在线问答流程

```text
PDF / Markdown / TXT
  → 文本提取或 OCR
  → 带 doc_id/chunk_id 的文档切片
  → Chroma 向量索引
  → 向量检索（必需）
  → Neo4j 图谱检索（可用时增强）
  → LLM 生成带来源引用的答案
```

Neo4j 不可用时，问答会记录警告并退化为纯向量 RAG，不会让整个查询失败。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少配置模型服务。默认模型面向 OpenAI；使用 DashScope 等 OpenAI 兼容服务时，同时修改 `OPENAI_BASE_URL`、模型名和 Embedding 名。Embedding 可通过 `EMBEDDING_API_KEY` 和 `EMBEDDING_BASE_URL` 使用独立 OpenAI-compatible 服务；两者为空时沿用 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。

构建向量索引：

```bash
python -m app.main ingest
```

需要让索引严格对应当前数据目录时使用：

```bash
python -m app.main ingest --rebuild
```

`--rebuild` 不会先清空 Chroma。程序先写入当前文档的全部片段，写入调用成功后才删除已经不存在的旧片段；普通 `ingest` 只新增或更新稳定 `chunk_id`，不会删除其他片段。

同时调用 LLM 抽取图谱并写入 Neo4j：

```bash
docker compose up -d neo4j
python -m app.main ingest --kg
```

`--kg` 会逐个 chunk 调用模型，耗时和费用都明显高于纯向量索引。

问答及 Web：

```bash
python -m app.main ask "Transformer 为什么使用注意力机制？"
python -m app.main web
```

Web 默认地址为 <http://127.0.0.1:8000>。

macOS 开发阶段不需要打开命令行：双击桌面的 `知识库工作台.app` 会自动启动服务并用 Google Chrome 打开页面；双击 `关闭知识库工作台.app` 会停止本地 Web 服务。移动硬盘需要保持连接。

Web API 保持精简且与 GUI 共用同一服务层：

- `GET /api/status`：配置、文档和索引状态。
- `GET /api/documents`：从 Chroma 元数据聚合已入库文档及切片数。
- `GET /api/upload-jobs`：查看浏览器上传和索引任务。
- `POST /api/answer-jobs`：创建异步问答任务，返回真实 RAG 阶段状态。
- `GET /api/answer-jobs/:id`：轮询问题编码、向量搜索、图谱搜索和答案生成进度，以及结构化向量命中原文。
- `POST /api/documents/upload`：以 `multipart/form-data` 上传 `files`，文件进入单线程后台索引队列。
- `POST /api/ingest`：`data_dir`、`build_kg`、`rebuild`。
- `POST /api/ask`：`question`。

浏览器支持拖拽上传 PDF、Markdown 和 TXT。上传大小、单次数量和排队任务数分别由
`MAX_UPLOAD_BYTES`、`MAX_UPLOAD_FILES`、`MAX_UPLOAD_JOBS` 控制。上传文件保存到
`DATA_DIR`，同名文件会更新对应文档的向量；新切片写入成功后才删除该文档的旧切片。
任务状态保存在 `.web_upload_jobs.json`，服务重启后未完成任务会重新排队。

浏览器问答输入框使用 Enter 发送、Shift+Enter 换行。问答过程面板展示各阶段的真实执行状态和耗时；向量命中包含 Chroma 距离、`chunk_id`、页码、来源和未经摘要的原始切片文本。Chroma 距离越小表示越相近，该值不是概率。

问答请求可并发执行，数量由 `MAX_CONCURRENT_ANSWERS` 控制；索引构建之间、
构建与问答之间互斥，避免同时读写 Chroma。该互斥只作用于单个应用进程，
不要在多个进程写同一个 Chroma 目录。

Web 服务没有用户认证，默认只监听本机；如果通过 `--host 0.0.0.0` 暴露到网络，应由反向代理补充 TLS、身份认证和访问控制。

## 离线图谱流水线

这套流水线用于批量处理教材，默认规则模式无需 API Key：

```bash
python scripts/kg/extract.py --mode rules
python scripts/kg/align.py
python scripts/kg/export.py
```

旧命令 `KG_RAG/libriry/01_triple.py` 等仍然可用。各阶段输入输出、数据规模和限制见 [KG_RAG/README.md](KG_RAG/README.md)。导出完成后可批量导入在线 Neo4j：

```bash
python -m app.main import-kg
```

需要完全替换现有图谱时显式加 `--clear`；该操作会删除 Neo4j 中现有节点和关系。

导入器会先完整解析并校验 JSONL，再执行 `--clear`，避免坏文件导致图谱被提前删除。Neo4j 批量写入本身不是跨批次事务；生产环境导入超大文件时应保留导出文件，并在失败后安全重试。

## 检查

```bash
python check_openai_key.py
pytest -q
```

测试不要求真实 API 或 Neo4j。当前测试集覆盖 PDF/OCR 分流、切片元数据、稳定向量 ID 与安全重建、图谱抽取和端点消歧、离线产物保护、Neo4j 导入校验与降级，以及 Web 请求边界和静态路径保护。

## 兼容性与可重建数据

- 主 CLI 的 `ingest`、`ask`、`web` 保持可用，并新增 `import-kg`。
- `KG_RAG/libriry/01_triple.py`、`02_duiqi.py`、`03_create_triple.py` 保持旧命令兼容。
- `data/kg`、`data/kg_aligned`、`data/triples` 都是生成数据，可从 `data/raw` 依次重建；不要直接维护大型生成文件。
- `.chroma` 和 Neo4j 是运行时状态，不属于离线生成数据。执行 `--rebuild` 或 `import-kg --clear` 前应确认目标目录和数据库配置。
