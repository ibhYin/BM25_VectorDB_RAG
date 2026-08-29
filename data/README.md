# 数据目录

```text
raw/          原始知识库文件；人工维护
kg/           scripts/kg/extract.py 生成；可重新构建
kg_aligned/   scripts/kg/align.py 生成；可重新构建
triples/      scripts/kg/export.py 生成；可重新构建
kg_test/      小规模图谱样例
```

不要手工修改生成目录中的大型 JSON/JSONL；需要调整结果时，修改对应脚本并重新生成。在线 Chroma 数据保存在项目根目录的 `.chroma/`，Neo4j 数据保存在 Docker volume 中。

旧的 `KG_RAG/libriry/01_triple.py`、`02_duiqi.py`、`03_create_triple.py` 仍可执行，但只负责转发到上述正式脚本。
