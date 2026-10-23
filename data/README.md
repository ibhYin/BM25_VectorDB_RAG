# 运行数据

`data/raw/` 保存本机上传或复制的原文件，`data/pending/` 保存待审核文件。
`data/indexes/<知识库>/generations/<版本>/` 保存独立索引版本（Chroma、BM25、模型清单及解析快照）。
审计记录和评测标注也属于本地运行数据。这些数据不进入 Git。

可公开的自编材料在 `examples/documents/`；新环境按根目录 README 复制后首次入库。
不要把真实财报语料、简历、账号、对话、向量库或解析缓存放入公开示例。
