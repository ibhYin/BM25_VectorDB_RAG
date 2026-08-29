# 离线知识图谱流水线

这里是批量处理 `data/raw` 技术教材的离线流程，不是在线问答入口。正式脚本位于 `scripts/kg/`；`libriry` 目录只保留旧命令兼容入口。

## 阶段

1. `scripts/kg/extract.py`：加载和切分文档，使用规则、LLM 或混合模式抽取实体关系，输出到 `data/kg`。
2. `scripts/kg/align.py`：清理无效实体，执行别名和模糊对齐，输出到 `data/kg_aligned`。
3. `scripts/kg/export.py`：过滤低质量关系，输出 JSONL、CSV 和纯文本三元组到 `data/triples`。

## 常用命令

完全离线：

```bash
python scripts/kg/extract.py --mode rules
python scripts/kg/align.py
python scripts/kg/export.py
```

规则和 LLM 混合抽取：

```bash
python scripts/kg/extract.py --mode hybrid
```

先用 `--max-chunks 20` 做小样本检查，再对全部教材运行，可以避免错误配置产生大量无效调用。

## 当前产物

| 阶段 | 实体 | 关系/三元组 |
|---|---:|---:|
| 原始抽取 | 11,155 | 36,878 |
| 对齐清洗 | 8,493 | 29,110 |
| 最终导出 | — | 25,977 |

三套流程现在共享 `kg/schema.py` 和 `kg/records.py` 的类型契约。导出文件可以通过 `python -m app.main import-kg` 批量写入 Neo4j。
