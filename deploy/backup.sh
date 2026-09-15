#!/bin/zsh
# 备份 RAG 服务里「丢了就没了」的东西。
#
#   ./deploy/backup.sh              备份到 ~/rag-backups/<时间戳>/
#   ./deploy/backup.sh /path/to/dir 备份到指定目录下
#   ./deploy/backup.sh --with-index 连向量索引一起备（体积大，但重建要花时间）
#
# 备份里含明文密钥（.env 与 runtime_config.json 存着 LLM API Key 和 MinerU
# Token），所以目录权限设成 700。别把它丢进同步盘或 Git。

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
STAMP="$(date +%Y%m%d-%H%M%S)"
WITH_INDEX=0
DEST_ROOT="${HOME}/rag-backups"

for arg in "$@"; do
  case "$arg" in
    --with-index) WITH_INDEX=1 ;;
    -*) printf '未知参数: %s\n' "$arg" >&2; exit 2 ;;
    *) DEST_ROOT="$arg" ;;
  esac
done

TARGET="${DEST_ROOT}/${STAMP}"
mkdir -p "$TARGET"
chmod 700 "$DEST_ROOT" "$TARGET"

cd "$PROJECT_DIR" || exit 1

# --- 不可再生的：丢了就永久失去，必须每次备 ---------------------------------
#   .env / runtime_config.json  密钥与模型配置
#   access_policy.json          谁是谁、谁能看什么
#   vector_audit.jsonl          只追加的审计日志，重建不出来
#   .web_conversations.json     用户的对话记录
IRREPLACEABLE=(
  .env
  config/runtime_config.json
  config/access_policy.json
  data/vector_audit.jsonl
  .web_conversations.json
)

# --- 可重建但代价高的：默认一起备，除非明确不要 -----------------------------
#   向量索引重建要重新 embedding（本地算力），走 MinerU 的还要花额度
#   .web_staging.json / data/pending 是待审队列，重建会丢待审内容
REBUILDABLE_BUT_PAINFUL=(
  .web_upload_jobs.json
  .web_staging.json
  data/pending
)

copied=0
skipped=0
copy_one() {
  local src="$1"
  if [[ -e "$src" ]]; then
    # 保留相对目录结构，-p 保留权限位
    mkdir -p "$TARGET/${src:h}"
    cp -Rp "$src" "$TARGET/$src"
    printf '  ✅ %-32s %s\n' "$src" "$(du -sh "$src" 2>/dev/null | cut -f1)"
    copied=$((copied + 1))
  else
    printf '  ⏭️  %-32s （不存在，跳过）\n' "$src"
    skipped=$((skipped + 1))
  fi
}

printf '备份到 %s\n\n' "$TARGET"
printf '不可再生的文件：\n'
for item in "${IRREPLACEABLE[@]}"; do
  copy_one "$item"
done

printf '\n队列与任务记录：\n'
for item in "${REBUILDABLE_BUT_PAINFUL[@]}"; do
  copy_one "$item"
done

if [[ "$WITH_INDEX" -eq 1 ]]; then
  printf '\n向量索引与 Markdown 缓存（体积大）：\n'
  for item in .chroma* .markdown-cache; do
    # 没有匹配到时 glob 会原样保留，用 -e 挡掉
    copy_one "$item"
  done
fi

printf '\n共 %d 项，跳过 %d 项，总计 %s\n' \
  "$copied" "$skipped" "$(du -sh "$TARGET" | cut -f1)"

if [[ "$WITH_INDEX" -eq 0 ]]; then
  printf '\n提示：本次未包含向量索引。重建索引需要重新入库全部文档（走 MinerU 的还要花额度），\n'
  printf '      想连索引一起备就加 --with-index。\n'
fi
