// 召回卡片的共享渲染：问答页的「匹配内容」与评估页的标注都用它。
//
// 放在这里而不是留在 ask.js 里，是因为评估页也要用同一张卡片——视图之间互相 import
// 会把两页耦在一起，而且以后卡片样式一改，两处显示就分叉了。
//
// options（缺省时保持旧行为，评估页不受影响）：
//   compact       默认收起全文，提供「展开全文/收起」切换
//   used          boolean，标题旁加「已用/未用」徽标
//   onOpenOriginal(hit)  提供时卡片加「在原文查看」按钮

export function createVectorHitCard(hit, options = {}) {
  const { compact = false, used, onOpenOriginal } = options;
  const card = document.createElement("article");
  card.className = "vector-hit";

  const header = document.createElement("div");
  header.className = "vector-hit-header";
  const titleRow = document.createElement("span");
  titleRow.className = "vector-hit-title-row";
  const title = document.createElement("strong");
  title.textContent = `${hit.rank}. ${hit.title || "未命名文档"}`;
  titleRow.append(title);
  if (typeof used === "boolean") {
    const badge = document.createElement("span");
    badge.className = `used-badge ${used ? "used" : "unused"}`;
    badge.textContent = used ? "已用" : "未用";
    titleRow.append(badge);
  }
  const distance = document.createElement("span");
  distance.className = "distance-value";
  if (hit.retrieval_method === "hybrid") {
    distance.title = "混合检索：相似度为向量语义相关性，BM25 为关键词匹配分";
    const similarity = Number(hit.similarity) || 0;
    const bm25 = Number(hit.bm25_score) || 0;
    distance.textContent = [
      similarity > 0 ? `相似度 ${similarity.toFixed(2)}` : "",
      bm25 > 0 ? `BM25 ${bm25.toPrecision(4)}` : "",
    ].filter(Boolean).join(" · ") || "混合";
  } else if (hit.retrieval_method === "bm25") {
    distance.title = "BM25 关键词匹配分数，越高越相关，不是概率";
    distance.textContent = `BM25 ${Number(hit.bm25_score).toPrecision(4)}`;
  } else {
    distance.title = "Chroma L2 距离（越小越近）· 相似度分数（越大越近）";
    const similarity = hit.similarity !== undefined ? Number(hit.similarity) : null;
    distance.textContent = [
      `距离 ${Number(hit.distance).toFixed(4)}`,
      similarity !== null ? `相似度 ${similarity.toFixed(2)}` : "",
    ].filter(Boolean).join(" · ");
  }
  header.append(titleRow, distance);

  const metadata = document.createElement("div");
  metadata.className = "vector-hit-meta";
  const page = hit.page_label
    || (hit.page !== null && hit.page !== undefined ? Number(hit.page) + 1 : "");
  metadata.textContent = [hit.chunk_id, page ? `第 ${page} 页` : "", hit.section_path]
    .filter(Boolean)
    .join(" · ");

  const source = document.createElement("div");
  source.className = "vector-hit-source";
  source.textContent = hit.source || "未知来源";

  const text = document.createElement("div");
  text.className = "vector-hit-text";
  // 服务端存的只是切片引用，回看历史对话时没有正文。
  text.textContent = hit.text || "（历史记录只保留切片引用，正文请在知识库页查看）";
  if (compact) text.hidden = true;

  const actions = document.createElement("div");
  actions.className = "vector-hit-actions";
  if (compact) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "text-button";
    toggle.textContent = "展开全文";
    toggle.addEventListener("click", () => {
      const isHidden = text.hidden;
      text.hidden = !isHidden;
      toggle.textContent = isHidden ? "收起" : "展开全文";
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

  card.append(header, metadata, source, actions, text);
  return card;
}
