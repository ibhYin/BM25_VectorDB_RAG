// 问答页：对话流、检索参数、过程与命中证据。

import {
  createSpinner,
  requestJson,
  router,
  shortDocLabel,
  state,
} from "../core.js";

const el = {
  chatStream: document.querySelector("#chat-stream"),
  runStatus: document.querySelector("#run-status"),
  clearChatButton: document.querySelector("#clear-chat-button"),
  askForm: document.querySelector("#ask-form"),
  askButton: document.querySelector("#ask-button"),
  question: document.querySelector("#question"),
  retrievalMethod: document.querySelector("#retrieval-method"),
  retrievalTopK: document.querySelector("#retrieval-top-k"),
  candidatePool: document.querySelector("#candidate-pool"),
  similarityThreshold: document.querySelector("#similarity-threshold"),
  similarityThresholdValue: document.querySelector("#similarity-threshold-value"),
  processList: document.querySelector("#process-list"),
  contextList: document.querySelector("#context-list"),
};

let mounted = false;

// --- retrieval controls ---------------------------------------------------

function getRetrievalMethod() {
  const checked = document.querySelector('input[name="retrieval-method"]:checked');
  return checked ? checked.value : "vector";
}

function setRetrievalMethod(value) {
  const radio = document.querySelector(
    `input[name="retrieval-method"][value="${value}"]`,
  );
  if (radio) radio.checked = true;
}

function updateRetrievalControls() {
  const method = getRetrievalMethod();
  const thresholdOff = method !== "vector";
  el.similarityThreshold.disabled = thresholdOff;
  el.similarityThresholdValue.textContent = thresholdOff
    ? "不适用"
    : Number(el.similarityThreshold.value).toFixed(2);
  el.candidatePool.disabled = method !== "hybrid";
}

// --- answer rendering -----------------------------------------------------

// A citation may or may not carry a chunk id; bold and inline code ride along in
// the same pass so the whole answer can be rendered without touching innerHTML.
const INLINE_PATTERN =
  /\[source:\s*([^\]#\s]+)(?:#(chunk_\d+))?\]|\*\*([^*]+)\*\*|`([^`]+)`/g;

function createCitationChip(docId, chunkId) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "citation";
  const label = shortDocLabel(docId);
  chip.textContent = chunkId ? `${label} ¶${chunkId.replace(/^chunk_/, "")}` : label;
  chip.title = chunkId
    ? `在原文中查看 ${docId} · ${chunkId}`
    : `打开文档 ${docId}`;
  // The document viewer lives on the library page now.
  chip.addEventListener("click", () => router.navigate("library", { docId, chunkId }));
  return chip;
}

function renderInline(text) {
  const fragment = document.createDocumentFragment();
  let cursor = 0;
  INLINE_PATTERN.lastIndex = 0;
  let match;
  while ((match = INLINE_PATTERN.exec(text)) !== null) {
    if (match.index > cursor) {
      fragment.append(document.createTextNode(text.slice(cursor, match.index)));
    }
    if (match[1]) {
      fragment.append(createCitationChip(match[1], match[2] || ""));
    } else if (match[3] !== undefined) {
      const strong = document.createElement("strong");
      strong.textContent = match[3];
      fragment.append(strong);
    } else {
      const code = document.createElement("code");
      code.textContent = match[4];
      fragment.append(code);
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    fragment.append(document.createTextNode(text.slice(cursor)));
  }
  return fragment;
}

// The model answers in light markdown; render just the subset it actually emits
// (headings, lists, bold, inline code, citations) as real DOM nodes.
function renderAnswer(text) {
  const fragment = document.createDocumentFragment();
  let paragraph = [];
  let list = null;
  let listTag = "";

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const block = document.createElement("p");
    block.append(renderInline(paragraph.join(" ")));
    fragment.append(block);
    paragraph = [];
  };
  const flushList = () => {
    if (!list) return;
    fragment.append(list);
    list = null;
    listTag = "";
  };

  for (const rawLine of String(text || "").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }
    if (/^[-*_]{3,}$/.test(line)) {
      flushParagraph();
      flushList();
      fragment.append(document.createElement("hr"));
      continue;
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      const node = document.createElement(heading[1].length <= 2 ? "h3" : "h4");
      node.append(renderInline(heading[2]));
      fragment.append(node);
      continue;
    }

    const bullet = /^[-*+]\s+(.*)$/.exec(line);
    const numbered = bullet ? null : /^\d+[.)]\s+(.*)$/.exec(line);
    if (bullet || numbered) {
      flushParagraph();
      const tag = bullet ? "ul" : "ol";
      if (listTag !== tag) {
        flushList();
        list = document.createElement(tag);
        listTag = tag;
      }
      const item = document.createElement("li");
      item.append(renderInline((bullet || numbered)[1]));
      list.append(item);
      continue;
    }

    flushList();
    paragraph.push(line);
  }
  flushParagraph();
  flushList();
  return fragment;
}

function addChatMessage(type, text) {
  const empty = el.chatStream.querySelector(".empty-chat");
  if (empty) empty.remove();

  const message = document.createElement("article");
  message.className = `chat-message ${type}`;
  if (type === "answer") {
    message.append(renderAnswer(text));
  } else {
    message.textContent = text;
  }
  el.chatStream.append(message);
  el.chatStream.scrollTop = el.chatStream.scrollHeight;
}

// --- evidence -------------------------------------------------------------

function renderContext() {
  const items = state.vectorHits || [];
  el.contextList.replaceChildren();

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无上下文";
    el.contextList.append(empty);
    return;
  }
  for (const item of items) el.contextList.append(createVectorHitCard(item));
}

function createVectorHitCard(hit) {
  const card = document.createElement("article");
  card.className = "vector-hit";

  const header = document.createElement("div");
  header.className = "vector-hit-header";
  const title = document.createElement("strong");
  title.textContent = `${hit.rank}. ${hit.title || "未命名文档"}`;
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
  header.append(title, distance);

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
  text.textContent = hit.text || "";

  card.append(header, metadata, source, text);
  return card;
}

function renderContextLoading() {
  el.contextList.replaceChildren();
  const loading = document.createElement("div");
  loading.className = "loading-state";
  loading.append(createSpinner(), document.createTextNode("检索中"));
  el.contextList.append(loading);
}

// --- process rail ---------------------------------------------------------

const processDefinitions = [
  ["encode", "问题编码"],
  ["vector_search", "向量搜索"],
  ["answer", "生成答案"],
];

function renderProcess() {
  el.processList.replaceChildren();
  const stages = state.answerJob ? state.answerJob.stages || {} : {};
  const method = state.answerJob?.retrieval_method || getRetrievalMethod();

  for (const [key, label] of processDefinitions) {
    const stage = stages[key] || { status: "pending", detail: {} };
    const item = document.createElement("article");
    item.className = `process-step ${stage.status}`;

    const indicator = document.createElement("span");
    indicator.className = "process-indicator";
    if (stage.status === "running") {
      indicator.append(createSpinner());
    } else {
      indicator.textContent = stageSymbol(stage.status);
    }

    const body = document.createElement("div");
    const title = document.createElement("strong");
    const labels = { bm25: "BM25 关键词搜索", hybrid: "混合检索" };
    title.textContent =
      key === "vector_search" && method !== "vector" ? labels[method] : label;
    const detail = document.createElement("span");
    detail.textContent = stageDetail(key, stage);
    body.append(title, detail);
    item.append(indicator, body);
    el.processList.append(item);
  }
}

function stageSymbol(status) {
  if (status === "completed") return "✓";
  if (status === "degraded") return "!";
  if (status === "failed") return "×";
  if (status === "skipped") return "-";
  return "·";
}

function stageDetail(key, stage) {
  const detail = stage.detail || {};
  const duration = detail.duration_ms !== undefined ? ` · ${detail.duration_ms} ms` : "";
  if (stage.status === "pending") return "等待执行";
  if (stage.status === "running") return "正在执行";
  if (stage.status === "failed") return "执行失败";
  if (stage.status === "skipped") return "已跳过";
  if (key === "encode") {
    const preview = Array.isArray(detail.vector_preview)
      ? ` · 前 8 维 [${detail.vector_preview.join(", ")}]`
      : "";
    return `${detail.model || "Embedding"} · ${detail.dimensions || 0} 维${duration}${preview}`;
  }
  if (key === "vector_search") {
    const pool = detail.pool ? ` · 池 ${detail.pool}` : "";
    const recall = detail.recall != null
      && detail.recall !== detail.matches && detail.recall !== detail.pool
      ? ` · 召回 ${detail.recall}` : "";
    return `${detail.matches || 0} 个结果 · Top ${detail.top_k || 0}${pool}${recall}${duration}`;
  }
  if (key === "answer") return `${detail.model || "LLM"}${duration}`;
  return "完成";
}

function setRunStatusLoading(loading, message) {
  el.runStatus.replaceChildren();
  if (loading) {
    el.runStatus.append(createSpinner(), document.createTextNode(` ${message}`));
  } else {
    el.runStatus.textContent = message;
  }
}

function updateRunStatusFromJob(job) {
  const stages = job.stages || {};
  if (stages.answer && stages.answer.status === "running") {
    setRunStatusLoading(true, "正在生成答案...");
  } else if (stages.vector_search && stages.vector_search.status === "running") {
    const messages = {
      bm25: "正在进行 BM25 关键词搜索...",
      hybrid: "正在进行混合检索...",
      vector: "正在搜索向量数据库...",
    };
    setRunStatusLoading(true, messages[job.retrieval_method] || messages.vector);
  } else if (stages.encode && stages.encode.status === "running") {
    setRunStatusLoading(true, "正在编码问题...");
  } else if (job.status === "queued") {
    setRunStatusLoading(true, "等待处理...");
  }
}

// --- asking ---------------------------------------------------------------

async function waitForAnswer(jobId) {
  while (true) {
    const job = await requestJson(`/api/answer-jobs/${jobId}`);
    state.answerJob = job;
    if (job.stages?.vector_search?.status === "completed") {
      state.vectorHits = job.vector_hits || [];
      renderContext();
    }
    renderProcess();
    updateRunStatusFromJob(job);
    if (job.status === "completed") return job;
    if (job.status === "failed") {
      throw new Error(job.error || "问答处理失败");
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

async function ask(event) {
  event.preventDefault();
  if (el.askButton.disabled || !el.askForm.reportValidity()) return;
  const question = el.question.value.trim();
  if (!question) {
    el.question.focus();
    return;
  }

  addChatMessage("question", question);
  el.question.value = "";
  el.askButton.disabled = true;
  setRunStatusLoading(true, "正在处理问题...");
  state.answerJob = { retrieval_method: getRetrievalMethod() };
  state.vectorHits = [];
  renderProcess();
  renderContextLoading();

  try {
    const job = await requestJson("/api/answer-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        retrieval_method: getRetrievalMethod(),
        top_k: Number(el.retrievalTopK.value),
        candidate_pool: el.candidatePool.disabled
          ? undefined
          : Number(el.candidatePool.value),
        similarity_threshold: el.similarityThreshold.disabled
          ? undefined
          : Number(el.similarityThreshold.value),
      }),
    });
    state.answerJob = job;
    renderProcess();
    const result = await waitForAnswer(job.id);
    state.vectorHits = result.vector_hits || [];
    addChatMessage("answer", result.answer || "未生成答案");
    renderContext();
    setRunStatusLoading(false, "回答完成");
  } catch (error) {
    addChatMessage("error", error.message);
    renderContext();
    setRunStatusLoading(false, "回答失败");
  } finally {
    el.askButton.disabled = false;
    el.question.focus();
  }
}

function clearChat() {
  el.chatStream.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "empty-chat";
  const mark = document.createElement("span");
  mark.className = "empty-mark";
  mark.setAttribute("aria-hidden", "true");
  mark.textContent = "RAG";
  const title = document.createElement("strong");
  title.textContent = "等待提问";
  const detail = document.createElement("span");
  detail.textContent = "答案将引用知识库中的 chunk_id";
  empty.append(mark, title, detail);
  el.chatStream.append(empty);
  state.vectorHits = [];
  state.answerJob = null;
  renderProcess();
  renderContext();
  el.runStatus.textContent = "就绪";
}

// --- view contract --------------------------------------------------------

// Evidence rail width, persisted so the layout survives a reload.
const RAIL_STORAGE_KEY = "rag-rail-widths";
const RAIL_LIMITS = { min: 260, max: 620, default: 360 };
const layout = document.querySelector(".ask-layout");
const resizer = document.querySelector("#resizer-right");

function setRailWidth(width) {
  const clamped = Math.round(
    Math.min(RAIL_LIMITS.max, Math.max(RAIL_LIMITS.min, width)),
  );
  layout.style.setProperty("--rail-right", `${clamped}px`);
  return clamped;
}

function currentRailWidth() {
  const rail = document.querySelector(".ask-rail");
  return rail.getBoundingClientRect().width || RAIL_LIMITS.default;
}

function saveRailWidth() {
  try {
    localStorage.setItem(
      RAIL_STORAGE_KEY,
      JSON.stringify({ right: Math.round(currentRailWidth()) }),
    );
  } catch (_) {
    /* localStorage unavailable (private mode); width stays session-only */
  }
}

function restoreRailWidth() {
  try {
    const saved = JSON.parse(localStorage.getItem(RAIL_STORAGE_KEY) || "{}");
    if (Number.isFinite(saved.right)) setRailWidth(saved.right);
  } catch (_) {
    /* ignore malformed storage */
  }
}

function enableRailResize() {
  let startX = 0;
  let startWidth = 0;
  let dragging = false;

  resizer.addEventListener("pointerdown", (event) => {
    dragging = true;
    startX = event.clientX;
    startWidth = currentRailWidth();
    resizer.classList.add("dragging");
    resizer.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  resizer.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    setRailWidth(startWidth - (event.clientX - startX));
  });

  const stop = (event) => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove("dragging");
    if (resizer.hasPointerCapture(event.pointerId)) {
      resizer.releasePointerCapture(event.pointerId);
    }
    saveRailWidth();
  };
  resizer.addEventListener("pointerup", stop);
  resizer.addEventListener("pointercancel", stop);

  resizer.addEventListener("dblclick", () => {
    setRailWidth(RAIL_LIMITS.default);
    saveRailWidth();
  });

  resizer.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const step = (event.shiftKey ? 40 : 12) * (event.key === "ArrowRight" ? -1 : 1);
    setRailWidth(currentRailWidth() + step);
    saveRailWidth();
  });
}

export default {
  name: "ask",

  mount() {
    if (mounted) return;
    mounted = true;

    restoreRailWidth();
    enableRailResize();

    // Empty states for the evidence rail before the first question.
    renderProcess();
    renderContext();

    el.askForm.addEventListener("submit", ask);
    el.clearChatButton.addEventListener("click", clearChat);
    el.question.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        el.askForm.requestSubmit();
      }
    });

    for (const element of [el.retrievalTopK, el.similarityThreshold]) {
      element.addEventListener("input", () => {
        state.retrievalInitialized = true;
        updateRetrievalControls();
        if (!state.answerJob) renderProcess();
      });
    }
    el.retrievalMethod.addEventListener("change", () => {
      state.retrievalInitialized = true;
      updateRetrievalControls();
      if (!state.answerJob) renderProcess();
    });
  },

  show() {
    el.chatStream.scrollTop = el.chatStream.scrollHeight;
  },

  // Retrieval defaults come from the server, but only until the user touches
  // them — re-entering the page must not clobber their edits.
  onStatus(status) {
    if (!status || state.retrievalInitialized) return;
    setRetrievalMethod(status.retrieval_method || "vector");
    el.retrievalTopK.value = status.retrieval_top_k ?? 20;
    el.candidatePool.value = status.candidate_pool_size ?? 100;
    el.similarityThreshold.value = status.similarity_threshold ?? 0.35;
    state.retrievalInitialized = true;
    updateRetrievalControls();
  },
};
