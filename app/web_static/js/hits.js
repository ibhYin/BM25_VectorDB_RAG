// 召回卡片的共享渲染：问答页的检索链路与评估页的标注都用它。
//
// options（缺省时保持旧行为，评估页不受影响）：
//   compact       正文默认收成 3 行预览，提供「展开全文/收起」切换
//   used          "used" | "unused" | "candidate"，标题旁加对应徽标；不传不显示
//   onOpenOriginal(hit)  提供时卡片加「在原文查看」按钮
//
// 混合检索的卡片会显示「向量 → BM25 → 融合」名次轨迹；缺少新字段的旧数据
// （历史对话）只显示已有的分数。

const USED_BADGE_LABELS = { used: "已引用", unused: "未引用", candidate: "候选" };
const SOURCE_LABELS = { vector: "向量", bm25: "BM25" };

export function setUsedBadge(card, used) {
  const badge = card.querySelector(".used-badge");
  if (!badge) return;
  badge.className = `used-badge ${used}`;
  badge.textContent = USED_BADGE_LABELS[used] || "";
  badge.title = "按答案中的切片引用标记判断；未引用不代表模型未读取或未使用";
}

function fmt(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function sourceLabel(sources) {
  if (!Array.isArray(sources) || !sources.length) return "";
  if (sources.length > 1) return "两路";
  return SOURCE_LABELS[sources[0]] || String(sources[0]);
}

function scoreParts(hit) {
  const parts = [];
  const hasVectorRank = hit.vector_rank !== undefined && hit.vector_rank !== null;
  const hasBm25Rank = hit.bm25_rank !== undefined && hit.bm25_rank !== null;

  if (hit.retrieval_method === "hybrid") {
    if (hasVectorRank) {
      parts.push({
        label: "向量",
        text: `#${hit.vector_rank} · ${fmt(hit.similarity)}`,
        title: "向量召回名次 · 由 L2 距离换算的相似度（0–1，非概率，不跨查询可比）",
      });
    } else if (hit.similarity !== undefined) {
      parts.push({
        label: "向量",
        text: fmt(hit.similarity),
        title: "由 L2 距离换算的相似度（0–1，非概率）",
      });
    }
    if (hasBm25Rank) {
      parts.push({
        label: "BM25",
        text: `#${hit.bm25_rank} · ${fmt(hit.bm25_score)}`,
        title: "BM25 关键词召回名次 · 原始分数（无固定上限，只与同次检索比较）",
      });
    } else if (hit.bm25_score !== undefined) {
      parts.push({
        label: "BM25",
        text: fmt(hit.bm25_score),
        title: "BM25 关键词匹配分，越高越相关，不是概率",
      });
    }
    const fusionValue = hit.rrf_score !== undefined
      ? ` · RRF ${Number(hit.rrf_score).toFixed(4)}`
      : hit.weighted_score !== undefined
        ? ` · 加权 ${Number(hit.weighted_score).toFixed(3)}`
        : "";
    parts.push({
      label: "融合",
      text: `#${hit.rank}${fusionValue}`,
      title: "最终名次 · 融合分只反映排序，不能与相似度或 BM25 比大小",
    });
    return parts;
  }

  if (hit.retrieval_method === "bm25") {
    parts.push({
      label: "BM25",
      text: fmt(hit.bm25_score),
      title: "BM25 关键词匹配分，越高越相关，不是概率",
    });
    return parts;
  }

  parts.push({
    label: "相似度",
    text: fmt(hit.similarity),
    title: "由 Chroma L2 距离换算的相似度（0–1，非概率），越大越接近",
  });
  if (hit.distance !== undefined) {
    parts.push({
      label: "距离",
      text: fmt(hit.distance, 4),
      title: "Chroma 返回的原始 L2 距离（越小越近）",
    });
  }
  return parts;
}

function movement(hit) {
  if (hit.retrieval_method !== "hybrid") return "";
  if (hit.vector_rank === undefined || hit.vector_rank === null) return "";
  const delta = Number(hit.vector_rank) - Number(hit.rank);
  if (!Number.isFinite(delta) || delta === 0) return "";
  return delta > 0 ? `↑${delta}` : `↓${Math.abs(delta)}`;
}

export function createVectorHitCard(hit, options = {}) {
  const { compact = false, used, onOpenOriginal } = options;
  const card = document.createElement("article");
  card.className = "vector-hit";
  card.dataset.chunkId = hit.chunk_id || "";

  const header = document.createElement("div");
  header.className = "vector-hit-header";
  const titleRow = document.createElement("span");
  titleRow.className = "vector-hit-title-row";
  const title = document.createElement("strong");
  title.textContent = `${hit.rank}. ${hit.title || "未命名文档"}`;
  titleRow.append(title);
  const badgeLabel = USED_BADGE_LABELS[used];
  if (badgeLabel) {
    const badge = document.createElement("span");
    badge.className = `used-badge ${used}`;
    badge.textContent = badgeLabel;
    badge.title = "按答案中的切片引用标记判断；未引用不代表模型未读取或未使用";
    titleRow.append(badge);
  }
  const source = sourceLabel(hit.sources);
  if (source) {
    const sourceChip = document.createElement("span");
    sourceChip.className = "source-chip";
    sourceChip.textContent = source;
    sourceChip.title = "这一条由哪一路或哪两路召回";
    titleRow.append(sourceChip);
  }
  header.append(titleRow);

  const trail = document.createElement("div");
  trail.className = "hit-trail";
  scoreParts(hit).forEach((part, index) => {
    if (index) {
      const arrow = document.createElement("span");
      arrow.className = "trail-arrow";
      arrow.textContent = "→";
      trail.append(arrow);
    }
    const chip = document.createElement("span");
    chip.className = "trail-chip";
    chip.title = part.title;
    const label = document.createElement("span");
    label.className = "trail-label";
    label.textContent = part.label;
    chip.append(label, document.createTextNode(part.text));
    trail.append(chip);
  });
  const move = movement(hit);
  if (move) {
    const moveChip = document.createElement("span");
    moveChip.className = `trail-move ${move.startsWith("↑") ? "up" : "down"}`;
    moveChip.textContent = move;
    moveChip.title = "最终名次相对向量召回名次的升降";
    trail.append(moveChip);
  }

  const metadata = document.createElement("div");
  metadata.className = "vector-hit-meta";
  const page = hit.page_label
    || (hit.page !== null && hit.page !== undefined ? Number(hit.page) + 1 : "");
  metadata.textContent = [hit.chunk_id, page ? `第 ${page} 页` : "", hit.section_path]
    .filter(Boolean)
    .join(" · ");

  const sourceLine = document.createElement("div");
  sourceLine.className = "vector-hit-source";
  sourceLine.textContent = hit.source || "未知来源";

  const text = document.createElement("div");
  text.className = "vector-hit-text";
  // 服务端存的只是切片引用，回看历史对话时没有正文。
  text.textContent = hit.text || "（历史记录只保留切片引用，正文请在知识库页查看）";
  if (compact) text.classList.add("clamped");

  const actions = document.createElement("div");
  actions.className = "vector-hit-actions";
  if (compact) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "text-button";
    toggle.textContent = "展开全文";
    toggle.addEventListener("click", () => {
      const clamped = text.classList.toggle("clamped");
      toggle.textContent = clamped ? "展开全文" : "收起";
    });
    actions.append(toggle);
  }
  if (onOpenOriginal) {
    const open = document.createElement("button");
    open.type = "button";
    open.className = "ghost-button";
    open.textContent = "在原文查看";
    open.addEventListener("click", (event) => {
      event.stopPropagation();
      onOpenOriginal(hit);
    });
    actions.append(open);
  }

  card.append(header, trail, metadata, sourceLine, actions, text);
  return card;
}
