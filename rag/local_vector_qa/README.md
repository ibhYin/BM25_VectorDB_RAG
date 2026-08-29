# Local Vector QA

只使用本地 Chroma 向量数据库的 RAG 问答子项目，不连接 Neo4j，也不使用知识图谱。

## 构建索引

```bash
conda run -n langchain python -m rag.local_vector_qa.cli ingest
```

需要删除当前数据目录中已经不存在的旧片段时加 `--rebuild`。新片段写入成功后才会删除旧片段，不会先清空向量库。

指定数据目录和向量库目录：

```bash
conda run -n langchain python -m rag.local_vector_qa.cli ingest \
  --data-dir data/raw \
  --persist-dir .chroma
```

## 检索片段

```bash
conda run -n langchain python -m rag.local_vector_qa.cli retrieve "你的问题" --top-k 4
```

## 直接问答

```bash
conda run -n langchain python -m rag.local_vector_qa.cli ask "你的问题" --top-k 4
```

## 查看状态

```bash
conda run -n langchain python -m rag.local_vector_qa.cli status
```

问答只基于召回的 Source Chunks，回答会要求引用 `chunk_id`。
