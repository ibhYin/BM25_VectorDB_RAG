// 问答页：对话流、检索参数、过程与命中证据。

import {
  canConfigure,
  createSpinner,
  formatTime,
  on,
  requestJson,
  router,
  shortDocLabel,
  showToast,
  state,
} from "../core.js";
import { createVectorHitCard, setUsedBadge } from "../hits.js";
import { createModelPicker } from "../model-picker.js";
import { enableResize } from "../resizer.js";
import { resetSelectSearch } from "../select-search.js";
import { submitAggregate } from "./aggregate.js";

const el = {
  chatStream: document.querySelector("#chat-stream"),
  runStatus: document.querySelector("#run-status"),
  clearChatButton: document.querySelector("#clear-chat-button"),
  askForm: document.querySelector("#ask-form"),
  askButton: document.querySelector("#ask-button"),
  question: document.querySelector("#question"),
  retrievalMethod: document.querySelector("#retrieval-method"),
  fusionMethod: document.querySelector("#fusion-method"),
  fusionK: document.querySelector("#fusion-k"),
  fusionKField: document.querySelector("#fusion-k-field"),
  fusionAlpha: document.querySelector("#fusion-alpha"),
  fusionAlphaField: document.querySelector("#fusion-alpha-field"),
  retrievalTopK: document.querySelector("#retrieval-top-k"),
  candidatePool: document.querySelector("#candidate-pool"),
  similarityThreshold: document.querySelector("#similarity-threshold"),
  similarityThresholdValue: document.querySelector("#similarity-threshold-value"),
  rerankEnabled: document.querySelector("#rerank-enabled"),
  rerankModelLabel: document.querySelector("#rerank-model-label"),
  rerankPoolSize: document.querySelector("#rerank-pool-size"),
  rerankScoreFilterEnabled: document.querySelector("#rerank-score-filter-enabled"),
  rerankScoreThreshold: document.querySelector("#rerank-score-threshold"),
  rerankPreferencesStatus: document.querySelector("#rerank-preferences-status"),
  rerankPreferencesRetry: document.querySelector("#rerank-preferences-retry"),
  processList: document.querySelector("#process-list"),
  processToggle: document.querySelector("#process-toggle"),
  retrievalStages: document.querySelector("#retrieval-stages"),
  retrievalCount: document.querySelector("#retrieval-count"),
  retrievalToggle: document.querySelector("#retrieval-toggle"),
  retrievalWarning: document.querySelector("#retrieval-warning"),
  retrievalWarningTitle: document.querySelector("#retrieval-warning-title"),
  retrievalWarningMessage: document.querySelector("#retrieval-warning-message"),
  retrievalDiagnosticToggle: document.querySelector("#retrieval-diagnostic-toggle"),
  retrievalDiagnosticCopy: document.querySelector("#retrieval-diagnostic-copy"),
  retrievalDiagnostic: document.querySelector("#retrieval-diagnostic"),
  kbPicker: document.querySelector(".kb-picker"),
  kbStatus: document.querySelector("#kb-status"),
  kbMenu: document.querySelector("#kb-menu"),
  kbOthers: document.querySelector("#kb-others"),
  newConversation: document.querySelector("#new-conversation"),
  conversationList: document.querySelector("#conversation-list"),

  llmModel: document.querySelector("#ask-llm-model"),
  llmReadonly: document.querySelector("#ask-llm-readonly"),
  llmModelCustom: document.querySelector("#ask-llm-model-custom"),
  llmHint: document.querySelector("#ask-llm-hint"),

  resizerLeft: document.querySelector("#resizer-left"),
  resizerRight: document.querySelector("#resizer-right"),
  resizerBar: document.querySelector("#resizer-bar"),
  resizerForm: document.querySelector("#resizer-form"),
};

let mounted = false;
let workspaceMode = "ask";
let conversationModel = "";
let retrievalPreferences = null;
let retrievalPreferencesEpoch = 0;
let preferencesSaving = false;
let preferencesDirty = false;
let retrievalWarningKey = "";
let retrievalDiagnosticValue = "";

// 流式答案：轮询期间不断把 job 里的增量答案重渲染进同一个 article。
let streamingAnswerEl = null;
let lastStreamedAnswer = "";
// 问答过程默认折叠成一行摘要，点「展开」看完整阶段。
let processExpanded = false;
// 点聊天消息 pin 到某一轮：非 null 时证据栏显示该轮的切片，否则跟最新。
let selectedTurnHits = null;
let selectedTurnAnswer = "";
let selectedTurnRetrieval = null;
let selectedTurnKey = "";
let selectedTurnEl = null;
// 最新一轮的 question 消息元素，答案完成后给它也挂上该轮数据。
let lastQuestionEl = null;

const convState = {
  items: [],
  currentId: "",
  /**
   * Bumped whenever the user switches conversation. The polling loop compares
   * it each tick so a late answer cannot paint into a different conversation.
   */
  epoch: 0,
};

// --- retrieval controls ---------------------------------------------------

function updateRerankControls() {
  const configured = Boolean(retrievalPreferences?.rerank_configured);
  const enabled = configured && el.rerankEnabled.checked;
  const filtering = enabled && el.rerankScoreFilterEnabled.checked;
  el.rerankEnabled.disabled = !configured;
  el.rerankPoolSize.disabled = !enabled;
  el.rerankScoreFilterEnabled.disabled = !enabled;
  el.rerankScoreThreshold.disabled = !filtering;
  el.rerankScoreThreshold.required = filtering;
  if (retrievalPreferences) {
    el.rerankModelLabel.textContent = configured
      ? `模型 · ${retrievalPreferences.rerank_model || "未命名"}`
      : "未配置重排服务，请联系管理员";
  }
}

function applyRetrievalPreferences(preferences) {
  retrievalPreferences = preferences;
  el.retrievalTopK.value = preferences.top_k ?? el.retrievalTopK.value;
  el.rerankPoolSize.value = preferences.rerank_pool_size ?? el.rerankPoolSize.value;
  el.rerankEnabled.checked = Boolean(preferences.rerank_enabled && preferences.rerank_configured);
  el.rerankScoreFilterEnabled.checked = Boolean(preferences.rerank_score_filter_enabled);
  el.rerankScoreThreshold.value = preferences.rerank_score_threshold ?? "";
  el.rerankPreferencesRetry.hidden = true;
  el.rerankPreferencesStatus.textContent = "偏好已按当前账号恢复";
  el.rerankPreferencesStatus.classList.remove("error");
  updateRerankControls();
}

async function loadRetrievalPreferences() {
  const epoch = ++retrievalPreferencesEpoch;
  el.rerankPreferencesStatus.textContent = "读取当前账号偏好…";
  el.rerankPreferencesStatus.classList.remove("error");
  el.rerankPreferencesRetry.hidden = true;
  try {
    const preferences = await requestJson("/api/preferences");
    if (epoch !== retrievalPreferencesEpoch) return;
    applyRetrievalPreferences(preferences);
  } catch (error) {
    if (epoch !== retrievalPreferencesEpoch) return;
    el.rerankPreferencesStatus.textContent = `偏好读取失败：${error.message}`;
    el.rerankPreferencesStatus.classList.add("error");
    el.rerankPreferencesRetry.hidden = false;
  }
}

function retrievalPreferencePayload() {
  const threshold = String(el.rerankScoreThreshold.value).trim();
  return {
    rerank_enabled: el.rerankEnabled.checked,
    rerank_pool_size: Number(el.rerankPoolSize.value),
    rerank_score_filter_enabled: el.rerankScoreFilterEnabled.checked,
    rerank_score_threshold: threshold === "" ? null : Number(threshold),
    top_k: Number(el.retrievalTopK.value),
  };
}

async function saveRetrievalPreferences() {
  if (!retrievalPreferences) return;
  if (preferencesSaving) {
    preferencesDirty = true;
    return;
  }
  const payload = retrievalPreferencePayload();
  if (payload.rerank_score_filter_enabled && payload.rerank_score_threshold === null) {
    el.rerankPreferencesStatus.textContent = "启用分数过滤后请填写最低相关性分";
    el.rerankPreferencesStatus.classList.add("error");
    return;
  }
  preferencesSaving = true;
  el.rerankPreferencesStatus.textContent = "保存当前账号偏好…";
  el.rerankPreferencesStatus.classList.remove("error");
  try {
    const saved = await requestJson("/api/preferences", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    retrievalPreferences = { ...retrievalPreferences, ...saved };
    el.rerankPreferencesStatus.textContent = "偏好已保存，下次进入自动恢复";
  } catch (error) {
    el.rerankPreferencesStatus.textContent = `偏好保存失败：${error.message}`;
    el.rerankPreferencesStatus.classList.add("error");
  } finally {
    preferencesSaving = false;
    if (preferencesDirty) {
      preferencesDirty = false;
      saveRetrievalPreferences();
    }
  }
}

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

function getFusionMethod() {
  const checked = document.querySelector('input[name="fusion-method"]:checked');
  return checked ? checked.value : "rrf";
}

function setFusionMethod(value) {
  const radio = document.querySelector(
    `input[name="fusion-method"][value="${value}"]`,
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

  // 融合方式只在 hybrid 下有意义（vector/bm25 是单路检索，没有融合）。
  const fusionOff = method !== "hybrid";
  for (const radio of el.fusionMethod.querySelectorAll("input")) {
    radio.disabled = fusionOff;
  }
  el.fusionMethod.classList.toggle("disabled", fusionOff);
  const weighted = getFusionMethod() === "weighted";
  el.fusionKField.hidden = fusionOff || weighted;
  el.fusionAlphaField.hidden = fusionOff || !weighted;
  updateRerankControls();
}

function setWorkspaceMode(mode) {
  workspaceMode = mode === "aggregate" ? "aggregate" : "ask";
  const aggregate = workspaceMode === "aggregate";
  document.querySelector("#page-ask").classList.toggle("aggregate-mode", aggregate);
  document.querySelector("#mode-ask").setAttribute("aria-pressed", String(!aggregate));
  document.querySelector("#mode-aggregate").setAttribute("aria-pressed", String(aggregate));
  document.querySelector("#aggregate-workspace").hidden = !aggregate;
  el.chatStream.hidden = aggregate;
  if (aggregate) {
    el.retrievalWarning.hidden = true;
    el.retrievalDiagnostic.hidden = true;
    retrievalWarningKey = "";
  } else if (mounted) {
    renderRetrieval();
  }
  document.querySelector(".ask-head h2").textContent = aggregate ? "样本归类统计" : "问答";
  el.runStatus.hidden = aggregate;
  el.clearChatButton.hidden = aggregate;
  el.retrievalTopK.disabled = aggregate;
  el.retrievalTopK.closest("label").hidden = aggregate;
  for (const id of ["aggregate-sample-size", "aggregate-top-categories"]) {
    document.getElementById(id).disabled = !aggregate;
  }
  el.question.placeholder = aggregate
    ? "输入归类主题，例如：检索样本中的违约条款有哪些类别？"
    : "输入问题，Enter 发送 · Shift+Enter 换行";
  el.askButton.title = aggregate ? "开始归类统计" : "发送";
  el.askButton.setAttribute("aria-label", el.askButton.title);
}

function requestParams() {
  const preferences = retrievalPreferences ? retrievalPreferencePayload() : {};
  return {
    llm_model: llmPicker.value() || state.status?.llm_model,
    retrieval_method: getRetrievalMethod(),
    top_k: Number(el.retrievalTopK.value),
    candidate_pool: el.candidatePool.disabled ? undefined : Number(el.candidatePool.value),
    similarity_threshold: el.similarityThreshold.disabled ? undefined : Number(el.similarityThreshold.value),
    fusion_method: el.candidatePool.disabled ? undefined : getFusionMethod(),
    fusion_k: el.candidatePool.disabled || getFusionMethod() !== "rrf" ? undefined : Number(el.fusionK.value),
    fusion_alpha: el.candidatePool.disabled || getFusionMethod() !== "weighted" ? undefined : Number(el.fusionAlpha.value),
    ...preferences,
  };
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

// 从答案文本收集被引用的完整切片 id（doc_id#chunk_N），用来给匹配内容标「已引用」。
function extractCitedChunkIds(answerText) {
  const cited = new Set();
  if (!answerText) return cited;
  for (const match of String(answerText).matchAll(INLINE_PATTERN)) {
    const doc = match[1];
    const chunk = match[2];
    if (doc && chunk) cited.add(`${doc}#${chunk}`);
  }
  return cited;
}

// 把一轮的命中切片按文档聚合：{doc_id → {title, total, used, firstChunk}}。
function aggregateDocs(hits, answerText) {
  const cited = answerText ? extractCitedChunkIds(answerText) : null;
  const docs = new Map();
  for (const hit of hits || []) {
    const docId = hit.doc_id;
    if (!docId) continue;
    let entry = docs.get(docId);
    if (!entry) {
      entry = { title: hit.title || docId, total: 0, used: 0, firstChunk: "" };
      docs.set(docId, entry);
    }
    entry.total += 1;
    if (cited && cited.has(hit.chunk_id)) entry.used += 1;
    if (!entry.firstChunk) entry.firstChunk = hit.chunk_id || "";
  }
  return docs;
}

// 从「这一轮」的 hits + 答案算某文档的已引用/未引用切片，跳文档查看器并双色高亮。
function navigateToDoc(hits, answerText, docId, chunkId) {
  const cited = answerText ? extractCitedChunkIds(answerText) : null;
  const sameDoc = (hits || []).filter((item) => item.doc_id === docId);
  const used = cited
    ? sameDoc.filter((item) => cited.has(item.chunk_id)).map((item) => item.chunk_id)
    : [];
  const unused = cited
    ? sameDoc.filter((item) => !cited.has(item.chunk_id)).map((item) => item.chunk_id)
    : [];
  router.navigate("library", { docId, chunkId, used, unused });
}

// 在答案消息下方追加一行「检索文档：…」，每篇文档一个可点击 chip。
function renderRetrievedDocs(answerEl, hits, answerText) {
  if (answerEl.querySelector(".answer-docs")) return;
  const docs = aggregateDocs(hits, answerText);
  if (!docs.size) return;

  const row = document.createElement("div");
  row.className = "answer-docs";
  const label = document.createElement("span");
  label.className = "answer-docs-label";
  label.textContent = "检索文档";
  row.append(label);

  for (const [docId, doc] of docs) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "doc-chip";
    chip.textContent = doc.used
      ? `${doc.title}（已引用 ${doc.used}/${doc.total}）`
      : `${doc.title}（${doc.total} 片）`;
    chip.title = "在文档查看器中查看该文档的命中切片";
    chip.addEventListener("click", () =>
      navigateToDoc(hits, answerText, docId, doc.firstChunk),
    );
    row.append(chip);
  }
  answerEl.append(row);
}

// 跳到知识库文档查看器，并把「同一文档下」的命中切片按已引用/未引用分色高亮。
// 答案还没生成时（候选期）不传高亮集合，只定位点击的切片。
function openHitInOriginal(hit) {
  const hits = selectedTurnHits ?? (state.vectorHits || []);
  const answerText = selectedTurnHits
    ? selectedTurnAnswer
    : state.answerJob?.answer || "";
  const cited = answerText ? extractCitedChunkIds(answerText) : null;
  const sameDoc = hits.filter((item) => item.doc_id === hit.doc_id);
  const used = cited
    ? sameDoc.filter((item) => cited.has(item.chunk_id)).map((item) => item.chunk_id)
    : [];
  const unused = cited
    ? sameDoc.filter((item) => !cited.has(item.chunk_id)).map((item) => item.chunk_id)
    : [];
  router.navigate("library", {
    docId: hit.doc_id,
    chunkId: hit.chunk_id,
    used,
    unused,
  });
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

function addChatMessage(type, text, turn) {
  const empty = el.chatStream.querySelector(".empty-chat");
  if (empty) empty.remove();

  const message = document.createElement("article");
  message.className = `chat-message ${type}`;
  if (turn) message._turn = turn;
  if (type === "answer") {
    message.append(renderAnswer(text));
  } else {
    message.textContent = text;
  }
  el.chatStream.append(message);
  el.chatStream.scrollTop = el.chatStream.scrollHeight;
  return message;
}

function elapsedSeconds(job) {
  const start = new Date(job?.started_at);
  const end = new Date(job?.finished_at);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return null;
  return ((end - start) / 1000).toFixed(1);
}

// 轮询期间把增量答案重渲染进同一个 article；没新内容就不重渲染。
function renderStreamingAnswer(text) {
  if (!streamingAnswerEl) {
    streamingAnswerEl = addChatMessage("answer", text);
    lastStreamedAnswer = text;
    return;
  }
  if (text === lastStreamedAnswer) return;
  streamingAnswerEl.replaceChildren(renderAnswer(text));
  el.chatStream.scrollTop = el.chatStream.scrollHeight;
  lastStreamedAnswer = text;
}

function finalizeAnswer(job) {
  const text = job.answer || "未生成答案";
  if (!streamingAnswerEl) streamingAnswerEl = addChatMessage("answer", text);
  else streamingAnswerEl.replaceChildren(renderAnswer(text));

  // 把该轮数据挂到答案与对应的问题消息上，点击任一可查看这轮的检索详情。
  // 这一份留在内存里，所以还带着完整候选轨迹；写进对话文件的是轻量摘要。
  const detail = job.stages?.vector_search?.detail || {};
  const rerankStage = job.stages?.rerank || {};
  const turn = {
    key: `job:${job.id || ""}`,
    hits: job.vector_hits || [],
    answer: job.answer || "",
    retrieval: {
      ...detail,
      retrieval_method: job.retrieval_method,
      fusion_method: job.fusion_method,
      llm_model: job.llm_model,
      rerank: { ...(rerankStage.detail || {}), status: rerankStage.status || "skipped" },
    },
  };
  streamingAnswerEl._turn = turn;
  if (lastQuestionEl) lastQuestionEl._turn = turn;

  const seconds = elapsedSeconds(job);
  if (seconds !== null) {
    const time = document.createElement("div");
    time.className = "answer-time";
    time.textContent = `⏱ 共 ${seconds} 秒`;
    streamingAnswerEl.append(time);
  }
  renderRetrievedDocs(streamingAnswerEl, turn.hits, turn.answer);
  streamingAnswerEl = null;
  lastStreamedAnswer = "";
}

// --- retrieval chain ------------------------------------------------------

// 当前要展示的是「钉住的那一轮」还是最新任务。
function retrievalSource() {
  if (selectedTurnHits) {
    return {
      key: selectedTurnKey || "pinned",
      hits: selectedTurnHits,
      answer: selectedTurnAnswer,
      retrieval: selectedTurnRetrieval || {},
      pinned: true,
    };
  }
  const job = state.answerJob;
  const detail = job?.stages?.vector_search?.detail || {};
  const rerankStage = job?.stages?.rerank || {};
  return {
    key: job?.id || "live",
    hits: state.vectorHits || [],
    answer: job?.answer || "",
    status: job?.stages?.vector_search?.status || "pending",
    error: job?.error || "",
    retrieval: {
      ...detail,
      retrieval_method: job?.retrieval_method,
      fusion_method: job?.fusion_method,
      rerank: { ...(rerankStage.detail || {}), status: rerankStage.status || "skipped" },
    },
    pinned: false,
  };
}

let retrievalStructureKey = "";
// chunk_id → { card, used }：结构不变时只更新已引用/未引用徽标，不重建 DOM，
// 免得 300ms 一次的轮询把用户展开的 <details> 收掉。
let retrievalCards = new Map();

function fmt(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function chunkLabel(chunkId) {
  const match = /chunk_(\d+)$/.exec(String(chunkId || ""));
  return match ? `¶${match[1]}` : String(chunkId || "");
}

function pageLabel(hit) {
  if (hit.page_label) return `第 ${hit.page_label} 页`;
  if (hit.page !== null && hit.page !== undefined) return `第 ${Number(hit.page) + 1} 页`;
  return "";
}

function fusionNote(retrieval) {
  const fusion = retrieval.retrieval_trace?.fusion || {};
  const method = fusion.method || retrieval.fusion_method || "rrf";
  const next = retrieval.rerank?.enabled ? "后续交给独立重排模型。" : "当前按融合结果直接取最终 Top-K。";
  if (method === "weighted") {
    const alpha = fusion.alpha ?? retrieval.fusion_alpha;
    return `加权融合 α=${fmt(alpha)}：α·相似度 + (1−α)·归一化 BM25；相似度未校准，属近似排序。${next}`;
  }
  const k = fusion.k ?? retrieval.fusion_k;
  return `RRF k=${k ?? "—"}：只按两路名次组合，缺席路贡献为零，分数不能与相似度/BM25 比较。${next}`;
}

const RECALL_PATH_LABELS = { vector: "向量", bm25: "BM25" };

function retrievalRecallSummary(retrieval) {
  const recall = retrieval?.recall_detail;
  if (recall?.status !== "degraded") return null;
  const failedPath = (recall.failed_paths || [])[0];
  const succeededPath = (recall.succeeded_paths || [])[0];
  if (!failedPath || !succeededPath) return null;
  const failedLabel = RECALL_PATH_LABELS[failedPath] || failedPath;
  const succeededLabel = RECALL_PATH_LABELS[succeededPath] || succeededPath;
  const routeError = recall.routes?.[failedPath]?.error || {};
  const impact = failedPath === "vector"
    ? "回答可能缺少语义相关内容"
    : "回答可能缺少关键词精确匹配结果";
  const hint = routeError.hint ? `排查建议：${routeError.hint}。` : "";
  return {
    failedPath,
    succeededPath,
    title: `检索降级：仅 ${succeededLabel}`,
    message: `${failedLabel}召回失败，本次仅使用 ${succeededLabel}结果，${impact}。${hint}`,
  };
}

function retrievalDiagnosticText(retrieval) {
  const recall = retrieval?.recall_detail || {};
  const routeFields = ["status", "elapsed_ms", "count"];
  const errorFields = [
    "stage", "error_type", "error_message", "error_code", "http_status",
    "provider_request_id", "retryable", "hint",
  ];
  const routes = {};
  for (const path of ["vector", "bm25"]) {
    const route = recall.routes?.[path];
    if (!route) continue;
    const safeRoute = Object.fromEntries(routeFields
      .filter(key => route[key] !== undefined && route[key] !== null && route[key] !== "")
      .map(key => [key, route[key]]));
    if (route.error) {
      safeRoute.error = Object.fromEntries(errorFields
        .filter(key => route.error[key] !== undefined && route.error[key] !== null && route.error[key] !== "")
        .map(key => [key, route.error[key]]));
    }
    routes[path] = safeRoute;
  }
  const diagnostic = {
    stage: "retrieval",
    task_id: retrieval?.task_id || "",
    recall_id: recall.recall_id || "",
    status: recall.status || "",
    parallel: recall.parallel === true,
    wall_ms: recall.wall_ms ?? null,
    succeeded_paths: recall.succeeded_paths || [],
    failed_paths: recall.failed_paths || [],
    routes,
  };
  return JSON.stringify(diagnostic, null, 2);
}

function renderRetrievalWarning(source) {
  const summary = workspaceMode === "ask" ? retrievalRecallSummary(source.retrieval) : null;
  const key = summary
    ? JSON.stringify([source.key, source.retrieval.recall_detail?.recall_id,
      source.retrieval.recall_detail?.status, summary.failedPath])
    : "";
  if (!summary) {
    retrievalWarningKey = "";
    retrievalDiagnosticValue = "";
    el.retrievalWarning.hidden = true;
    el.retrievalDiagnostic.hidden = true;
    el.retrievalDiagnosticToggle.setAttribute("aria-expanded", "false");
    return;
  }
  el.retrievalWarning.hidden = false;
  el.retrievalWarningTitle.textContent = summary.title;
  el.retrievalWarningMessage.textContent = summary.message;
  if (key !== retrievalWarningKey) {
    retrievalWarningKey = key;
    retrievalDiagnosticValue = retrievalDiagnosticText(source.retrieval);
    el.retrievalDiagnostic.textContent = retrievalDiagnosticValue;
    el.retrievalDiagnostic.hidden = true;
    el.retrievalDiagnosticToggle.textContent = "查看诊断";
    el.retrievalDiagnosticToggle.setAttribute("aria-expanded", "false");
  }
}

const RERANK_DIAGNOSTIC_FIELDS = [
  "status", "reason", "model", "pool_size", "input_count", "output_count",
  "filtered_count", "score_filter_enabled", "score_threshold", "elapsed_ms",
  "http_status", "error_type", "error_code", "error_message", "request_id",
  "provider_request_id", "created_at", "task_id",
];

function rerankDiagnosticText(detail) {
  const safe = { stage: "rerank" };
  for (const key of RERANK_DIAGNOSTIC_FIELDS) {
    const value = detail?.[key];
    if (value !== undefined && value !== null && value !== "") safe[key] = value;
  }
  return JSON.stringify(safe, null, 2);
}

function rerankStage(retrieval, hits, finalRankById, index) {
  const detail = retrieval.rerank || {};
  if (!detail.enabled) return null;
  const status = detail.status || "pending";
  const reason = {
    disabled: "功能关闭",
    no_candidates: "没有候选",
    single_candidate: "只有一个候选",
    fallback: "调用失败，已回退原排序",
  }[detail.reason] || detail.reason || "";
  const counts = detail.input_count != null || detail.output_count != null
    ? `${detail.input_count ?? "—"} → ${detail.output_count ?? hits.length}`
    : `${hits.length} 条`;
  let note = `模型 ${detail.model || "—"} · ${counts}`;
  if (detail.elapsed_ms != null) note += ` · ${detail.elapsed_ms} ms`;
  if (status === "degraded" || status === "failed") {
    note = `${reason || "重排调用失败"} · ${detail.error_message || detail.error_type || "未提供错误详情"} · ${note}`;
  } else if (status === "skipped") {
    note = `未调用重排模型 · ${reason || "已跳过"} · ${note}`;
  } else if (status === "running") {
    note = `正在调用重排模型 · ${note}`;
  } else if (detail.score_filter_enabled) {
    note += ` · 阈值 ≥ ${fmt(detail.score_threshold)} · 过滤 ${detail.filtered_count ?? 0} 条`;
  }
  return {
    key: "rerank",
    index,
    title: status === "degraded" || status === "failed" ? "模型重排（已回退）" : "模型重排",
    stat: status === "completed" ? `${detail.input_count ?? hits.length} → ${hits.length} 条` : counts,
    empty: status === "completed" && detail.score_filter_enabled
      ? "没有达到阈值的证据"
      : status === "degraded" ? "重排失败，最终使用原融合排序" : "没有重排结果",
    note,
    rows: hits,
    cards: true,
    open: true,
    diagnostic: status === "degraded" || status === "failed" ? { ...detail, status } : null,
    score: (row) => row.rerank_score != null ? `相关度 ${fmt(row.rerank_score)}` : "回退排序",
    rankOf: (row) => row.rerank_rank ?? row.rank,
    finalRankById,
  };
}

function buildStages(method, source, trace, hits) {
  const retrieval = source.retrieval;
  const finalRankById = new Map(hits.map((hit) => [hit.chunk_id, hit.rank]));
  const reranking = Boolean(retrieval.rerank?.enabled);

  if (method === "hybrid") {
    const candidates = trace && Array.isArray(trace.candidates) ? trace.candidates : null;
    const recall = retrieval.recall_detail || {};
    const routes = recall.routes || {};
    const degraded = recall.status === "degraded";
    const parallelTiming = recall.parallel
      ? `两路并行：向量 ${routes.vector?.elapsed_ms ?? "—"} ms / BM25 ${routes.bm25?.elapsed_ms ?? "—"} ms，整体 ${recall.wall_ms ?? "—"} ms。`
      : "";
    const routeIssue = (path) => {
      const error = routes[path]?.error || {};
      return [error.error_message || "未提供错误详情", error.hint ? `建议：${error.hint}` : ""]
        .filter(Boolean).join("；");
    };
    const routeDiagnostic = (path) => {
      const error = routes[path]?.error;
      return error ? { ...error, recall_id: recall.recall_id, task_id: retrieval.task_id } : null;
    };
    const vectorRows = (candidates
      ? candidates.filter((item) => item.vector_rank != null)
      : hits.filter((hit) => hit.vector_rank != null)
    ).slice().sort((a, b) => a.vector_rank - b.vector_rank);
    const bm25Rows = (candidates
      ? candidates.filter((item) => item.bm25_rank != null)
      : hits.filter((hit) => hit.bm25_rank != null)
    ).slice().sort((a, b) => a.bm25_rank - b.bm25_rank);
    const vectorCount = trace
      ? trace.vector_count
      : retrieval.vector_candidates;
    const bm25Count = trace
      ? trace.bm25_count
      : retrieval.bm25_candidates;
    const unionCount = trace ? trace.union_count : retrieval.union_candidates;
    const historyNote = candidates
      ? ""
      : " 历史候选明细未保存；下方仅展示最终结果中可追溯的条目，不代表完整召回池。";
    const fusionRows = reranking && candidates ? candidates : hits;
    const vectorFailed = routes.vector?.status === "failed";
    const bm25Failed = routes.bm25?.status === "failed";
    const fusionScore = (row) => {
      const method = trace?.fusion?.method || retrieval.fusion_method;
      if (method === "weighted") return `加权 ${fmt(row.weighted_score)}`;
      return `RRF ${fmt(row.rrf_score)}`;
    };
    const stages = [
      {
        key: "vector",
        index: "①",
        title: vectorFailed ? "向量召回（失败）" : "向量召回",
        status: routes.vector?.status,
        stat: vectorFailed ? "失败" : vectorCount != null ? `${vectorCount} 条` : vectorRows.length ? "总数未记录" : "未记录",
        empty: vectorFailed ? "向量召回失败，本次未参与融合" : vectorCount === 0 ? "向量这一路没有命中" : "历史召回明细未保存，无法还原候选列表",
        note: vectorFailed
          ? `向量路失败：${routeIssue("vector")}。${parallelTiming}`
          : `按 L2 距离换算分排序，是候选来源之一。${parallelTiming}${degraded ? "本次为唯一有效召回。" : ""}${historyNote}`,
        rows: vectorRows,
        diagnostic: routeDiagnostic("vector"),
        diagnosticLabel: "复制向量诊断",
        score: (row) => `相似度 ${fmt(row.similarity)}`,
        rankOf: (row) => row.vector_rank,
        finalRankById,
      },
      {
        key: "bm25",
        index: "②",
        title: bm25Failed ? "BM25 召回（失败）" : "BM25 召回",
        status: routes.bm25?.status,
        stat: bm25Failed ? "失败" : bm25Count != null ? `${bm25Count} 条` : bm25Rows.length ? "总数未记录" : "未记录",
        empty: bm25Failed ? "BM25 召回失败，本次未参与融合" : bm25Count === 0 ? "BM25 这一路没有命中" : "历史召回明细未保存，无法还原候选列表",
        note: bm25Failed
          ? `BM25 路失败：${routeIssue("bm25")}。${parallelTiming}`
          : `在持久化 FTS5 索引上检索全库，与向量路并行且相互独立，不是对①的重排。${parallelTiming}${degraded ? "本次为唯一有效召回。" : ""}${historyNote}`,
        rows: bm25Rows,
        diagnostic: routeDiagnostic("bm25"),
        diagnosticLabel: "复制 BM25 诊断",
        score: (row) => `BM25 ${fmt(row.bm25_score)}`,
        rankOf: (row) => row.bm25_rank,
        finalRankById,
      },
      {
        key: "fusion",
        index: "③",
        title: "融合",
        stat: reranking
          ? `并集 ${unionCount ?? "—"} → 重排 ${retrieval.rerank.input_count ?? fusionRows.length}`
          : unionCount != null ? `并集 ${unionCount} → 最终 ${hits.length}` : `最终 ${hits.length}`,
        status: recall.status,
        note: `${fusionNote(retrieval)}${degraded ? " 本次为降级融合，仅成功路径参与。" : ""}`,
        rows: fusionRows,
        cards: !reranking,
        open: !reranking,
        score: fusionScore,
        rankOf: (row) => row.final_rank ?? row.rank,
        finalRankById,
      },
    ];
    const finalRerank = rerankStage(retrieval, hits, finalRankById, "④");
    if (finalRerank) stages.push(finalRerank);
    return stages;
  }

  if (method === "bm25") {
    const stage = {
      key: "single",
      index: "①",
      title: "BM25 关键词召回",
      stat: reranking ? `${retrieval.rerank.input_count ?? hits.length} 条输入` : `${hits.length} 条`,
      note: `在持久化 FTS5 索引上检索全库。${reranking ? "下方仅保留最终入选条目。" : ""}`,
      rows: hits,
      cards: !reranking,
      open: !reranking,
      score: (row) => `BM25 ${fmt(row.bm25_score)}`,
      rankOf: (row) => row.retrieval_rank ?? row.bm25_rank ?? row.rank,
      finalRankById,
    };
    const finalRerank = rerankStage(retrieval, hits, finalRankById, "②");
    return finalRerank ? [stage, finalRerank] : [stage];
  }

  if (method === "vector") {
    const threshold = retrieval.similarity_threshold;
    const stage = {
      key: "single",
      index: "①",
      title: "向量召回",
      stat: reranking ? `${retrieval.rerank.input_count ?? hits.length} 条输入` : `${hits.length} 条`,
      note: `按距离换算分取 Top-K${threshold != null ? `，阈值 ≥ ${fmt(threshold)}` : ""}。${reranking ? "下方仅保留最终入选条目。" : ""}`,
      rows: hits,
      cards: !reranking,
      open: !reranking,
      score: (row) => `相似度 ${fmt(row.similarity)}`,
      rankOf: (row) => row.retrieval_rank ?? row.vector_rank ?? row.rank,
      finalRankById,
    };
    const finalRerank = rerankStage(retrieval, hits, finalRankById, "②");
    return finalRerank ? [stage, finalRerank] : [stage];
  }

  return [];
}

function createStageRow(row, stage) {
  const item = document.createElement("details");
  item.className = "stage-row";
  const head = document.createElement("summary");
  head.className = "stage-row-head";

  const rank = document.createElement("span");
  rank.className = "stage-rank";
  rank.textContent = `#${stage.rankOf(row) ?? "—"}`;
  const title = document.createElement("span");
  title.className = "stage-row-title";
  title.textContent = `${row.title || "未命名文档"} · ${chunkLabel(row.chunk_id)}`;
  const score = document.createElement("span");
  score.className = "stage-score";
  score.textContent = stage.score(row);
  const final = document.createElement("span");
  final.className = "stage-final";
  const finalRank = stage.finalRankById.get(row.chunk_id);
  if (finalRank) {
    final.textContent = `→ 最终 #${finalRank}`;
  } else {
    final.textContent = "未入选";
    final.classList.add("dropped");
  }
  head.append(rank, title, score, final);

  const body = document.createElement("div");
  body.className = "stage-row-body";
  const meta = document.createElement("div");
  meta.className = "vector-hit-meta";
  meta.textContent = [row.doc_id, pageLabel(row), row.section_path]
    .filter(Boolean)
    .join(" · ");
  const open = document.createElement("button");
  open.type = "button";
  open.className = "ghost-button";
  open.textContent = "在原文查看";
  open.addEventListener("click", (event) => {
    event.preventDefault();
    openHitInOriginal(row);
  });
  body.append(meta, open);
  item.append(head, body);
  return item;
}

function createStageCard(hit, cited) {
  const used = cited === null ? "candidate" : cited.has(hit.chunk_id) ? "used" : "unused";
  const card = createVectorHitCard(hit, {
    compact: true,
    used,
    onOpenOriginal: openHitInOriginal,
  });
  retrievalCards.set(hit.chunk_id, { card, used });
  return card;
}

async function copyDiagnosticText(text, button) {
  const original = button.textContent;
  try {
    if (!navigator.clipboard?.writeText) throw new Error("当前浏览器不支持剪贴板");
    await navigator.clipboard.writeText(text);
    button.textContent = "已复制";
    setTimeout(() => { button.textContent = original; }, 1500);
  } catch (error) {
    showToast(`复制诊断失败：${error.message}`, "error");
  }
}

function copyRerankDiagnostic(detail, button) {
  return copyDiagnosticText(rerankDiagnosticText(detail), button);
}

function copyRetrievalDiagnostic(button) {
  return copyDiagnosticText(retrievalDiagnosticValue, button);
}

function toggleRetrievalDiagnostic() {
  const expand = el.retrievalDiagnostic.hidden;
  el.retrievalDiagnostic.hidden = !expand;
  el.retrievalDiagnosticToggle.textContent = expand ? "收起诊断" : "查看诊断";
  el.retrievalDiagnosticToggle.setAttribute("aria-expanded", String(expand));
}

function renderStage(stage, cited) {
  const details = document.createElement("details");
  details.className = "retrieval-stage";
  details.dataset.stage = stage.key;
  if (stage.status) details.dataset.status = stage.status;
  details.open = stage.open ?? (stage.key !== "vector" && stage.key !== "bm25");

  const head = document.createElement("summary");
  head.className = "stage-head";
  const index = document.createElement("span");
  index.className = "stage-index";
  index.textContent = stage.index;
  const title = document.createElement("strong");
  title.textContent = stage.title;
  const stat = document.createElement("span");
  stat.className = "stage-stat";
  stat.textContent = stage.stat;
  const chevron = document.createElement("span");
  chevron.className = "stage-chevron";
  chevron.setAttribute("aria-hidden", "true");
  chevron.textContent = "▾";
  head.append(index, title, stat, chevron);

  const note = document.createElement("p");
  note.className = "stage-note";
  note.textContent = stage.note;

  const list = document.createElement("ol");
  list.className = "stage-list";
  if (!stage.rows.length) {
    const empty = document.createElement("li");
    empty.className = "stage-empty";
    empty.textContent = stage.empty || "这一路没有结果";
    list.append(empty);
  }
  for (const row of stage.rows) {
    const item = document.createElement("li");
    item.className = "stage-item";
    item.append(stage.cards ? createStageCard(row, cited) : createStageRow(row, stage));
    list.append(item);
  }

  details.append(head, note);
  if (stage.diagnostic) {
    const actions = document.createElement("div");
    actions.className = "stage-diagnostic-actions";
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "text-button";
    copy.textContent = stage.diagnosticLabel || "复制失败诊断";
    copy.addEventListener("click", () => copyRerankDiagnostic(stage.diagnostic, copy));
    actions.append(copy);
    details.append(actions);
  }
  details.append(list);
  return details;
}

function updateUsedBadges(cited) {
  for (const [chunkId, entry] of retrievalCards) {
    const used = cited === null
      ? "candidate"
      : cited.has(chunkId) ? "used" : "unused";
    if (entry.used === used) continue;
    entry.used = used;
    setUsedBadge(entry.card, used);
  }
}

function renderRetrieval() {
  const source = retrievalSource();
  renderRetrievalWarning(source);
  const method = source.retrieval.retrieval_method || getRetrievalMethod();
  const hits = source.hits || [];
  const trace = source.retrieval.retrieval_trace || null;
  const cited = source.answer ? extractCitedChunkIds(source.answer) : null;

  el.retrievalCount.textContent = hits.length ? ` · ${hits.length} 条` : "";

  const key = [
    source.key,
    method,
    hits.map((hit) => hit.chunk_id).join("␟"),
    trace?.candidates?.length || 0,
    source.retrieval.vector_candidates,
    source.retrieval.bm25_candidates,
    source.retrieval.rerank?.status,
    source.retrieval.rerank?.output_count,
    source.retrieval.recall_detail?.status,
    source.retrieval.recall_detail?.failed_paths?.join("+") || "",
    source.status,
    source.pinned ? "pinned" : "live",
  ].join("|");

  if (key === retrievalStructureKey) {
    updateUsedBadges(cited);
    return;
  }
  retrievalStructureKey = key;
  retrievalCards = new Map();
  el.retrievalStages.replaceChildren();

  if (method === "system" || (!hits.length && !trace && source.status !== "completed"
      && source.retrieval.vector_candidates == null && source.retrieval.bm25_candidates == null)) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = method === "system"
      ? "本次由系统状态直接回答，未检索知识库。"
      : source.pinned ? "历史检索明细未保存"
        : source.status === "failed" ? `检索失败：${source.error || "请查看问答过程"}`
          : source.status === "running" ? "检索中…"
            : source.status === "skipped" ? "本次未执行检索"
              : "尚未执行检索";
    el.retrievalStages.append(empty);
    return;
  }

  const stages = buildStages(method, source, trace, hits);
  if (!stages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无上下文";
    el.retrievalStages.append(empty);
    return;
  }
  for (const stage of stages) {
    el.retrievalStages.append(renderStage(stage, cited));
  }
  updateUsedBadges(cited);
  el.retrievalToggle.textContent = "展开全部";
  el.retrievalToggle.setAttribute("aria-expanded", "false");
}

function renderRetrievalLoading() {
  renderRetrievalWarning({ key: "loading", retrieval: {} });
  retrievalStructureKey = "";
  retrievalCards = new Map();
  el.retrievalCount.textContent = "";
  el.retrievalStages.replaceChildren();
  const loading = document.createElement("div");
  loading.className = "loading-state";
  loading.append(createSpinner(), document.createTextNode("检索中"));
  el.retrievalStages.append(loading);
}

function toggleRetrievalStages() {
  const stages = [...el.retrievalStages.querySelectorAll("details.retrieval-stage")];
  if (!stages.length) return;
  const expand = stages.some((stage) => !stage.open);
  for (const stage of stages) stage.open = expand;
  el.retrievalToggle.textContent = expand ? "收起全部" : "展开全部";
  el.retrievalToggle.setAttribute("aria-expanded", String(expand));
}

function clearSelectedTurn() {
  selectedTurnHits = null;
  selectedTurnAnswer = "";
  selectedTurnRetrieval = null;
  selectedTurnKey = "";
  if (selectedTurnEl) {
    selectedTurnEl.classList.remove("selected");
    selectedTurnEl = null;
  }
}

// --- process rail ---------------------------------------------------------

const processDefinitions = [
  ["condense", "问题改写"],
  ["system", "系统问答"],
  ["encode", "问题编码"],
  ["vector_search", "向量搜索"],
  ["rerank", "模型重排"],
  ["answer", "生成答案"],
];

function renderProcess() {
  el.processList.replaceChildren();
  const stages = state.answerJob ? state.answerJob.stages || {} : {};
  const method = state.answerJob?.retrieval_method || getRetrievalMethod();

  if (!processExpanded) {
    // 折叠：一行摘要，把空间让给匹配内容。
    const summary = document.createElement("div");
    summary.className = "process-summary";
    summary.textContent = processSummary(stages);
    el.processList.append(summary);
    return;
  }

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
    const labels = { bm25: "BM25 关键词搜索", hybrid: "混合检索（双路并行）" };
    title.textContent =
      key === "vector_search" && method !== "vector" ? labels[method] : label;
    const detail = document.createElement("span");
    detail.textContent = stageDetail(key, stage);
    body.append(title, detail);
    item.append(indicator, body);
    el.processList.append(item);
  }
}

function processSummary(stages) {
  const job = state.answerJob;
  const entries = processDefinitions.map(([key, label]) => ({
    key,
    label,
    stage: (stages || {})[key] || {},
  }));
  const failed = entries.find((entry) => entry.stage.status === "failed");
  const degraded = entries.filter((entry) => entry.stage.status === "degraded");
  const running = entries.find((entry) => entry.stage.status === "running");
  const terminal = entries.filter((entry) => ["completed", "degraded", "skipped"].includes(entry.stage.status)).length;
  if (failed) return `× ${failed.label} 失败`;
  if (running) return `${running.label}…`;
  if (terminal === entries.length) {
    const seconds = elapsedSeconds(job);
    if (degraded.length) {
      const retrievalDegraded = degraded.find((entry) => entry.key === "vector_search");
      const rerankDegraded = degraded.some((entry) => entry.key === "rerank");
      const recall = retrievalDegraded?.stage.detail?.recall_detail;
      if (recall?.status === "degraded") {
        const successPath = recall.succeeded_paths?.[0];
        const label = RECALL_PATH_LABELS[successPath] || successPath || "单路";
        const rerankNote = rerankDegraded ? "；重排回退" : "";
        return `! 检索降级：仅 ${label}${rerankNote}${seconds !== null ? ` · 共 ${seconds} 秒` : ""}`;
      }
      return `! ${degraded[0].label}失败，已回退${seconds !== null ? ` · 共 ${seconds} 秒` : ""}`;
    }
    return `✓ 完成${seconds !== null ? ` · 共 ${seconds} 秒` : ""}`;
  }
  if (terminal > 0) return `进行中 · 已完成 ${terminal}/${entries.length}`;
  return "等待执行";
}

function toggleProcess() {
  processExpanded = !processExpanded;
  el.processToggle.setAttribute("aria-expanded", String(processExpanded));
  el.processToggle.textContent = processExpanded ? "收起" : "展开";
  renderProcess();
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
  if (stage.status === "running") {
    return key === "rerank"
      ? `正在调用 ${detail.model || "重排模型"} · 输入 ${detail.input_count ?? "—"} 条`
      : "正在执行";
  }
  if (stage.status === "failed") return "执行失败";
  if (key === "rerank") {
    const counts = `${detail.input_count ?? "—"} → ${detail.output_count ?? "—"}`;
    if (stage.status === "degraded") {
      return `失败回退 · ${detail.error_message || detail.error_type || "重排服务不可用"} · ${counts}${duration}`;
    }
    if (stage.status === "skipped") {
      const reasons = { disabled: "功能关闭", no_candidates: "没有候选", single_candidate: "只有一个候选" };
      return `已跳过 · ${reasons[detail.reason] || detail.reason || "无需重排"}${duration}`;
    }
    return `${detail.model || "重排模型"} · ${counts}${duration}`;
  }
  if (stage.status === "skipped") return "已跳过";
  if (key === "encode") {
    if (stage.status === "degraded") {
      return `向量编码失败 · ${detail.error_message || detail.error_type || "Embedding 不可用"}${duration}`;
    }
    const preview = Array.isArray(detail.vector_preview)
      ? ` · 前 8 维 [${detail.vector_preview.join(", ")}]`
      : "";
    return `${detail.model || "Embedding"} · ${detail.dimensions || 0} 维${duration}${preview}`;
  }
  if (key === "vector_search") {
    const fusionLabels = { rrf: "RRF", weighted: "加权" };
    const fusion = fusionLabels[detail.fusion_method]
      ? ` · ${fusionLabels[detail.fusion_method]}`
      : "";
    if (detail.retrieval_method === "hybrid") {
      const recall = detail.recall_detail;
      if (recall?.status === "degraded") {
        const successPath = recall.succeeded_paths?.[0];
        const successLabel = RECALL_PATH_LABELS[successPath] || successPath || "单路";
        return `双路并行 → 降级为仅 ${successLabel} · 最终 ${detail.matches || 0} 条 · ${recall.wall_ms ?? "—"} ms`;
      }
      const vector = detail.vector_candidates ?? detail.recall ?? 0;
      const bm25 = detail.bm25_candidates ?? 0;
      const union = detail.union_candidates != null
        ? ` → 去重 ${detail.union_candidates}`
        : "";
      const parallel = recall?.parallel ? ` · 并行 ${recall.wall_ms ?? "—"} ms` : "";
      return `向量 ${vector} + BM25 ${bm25}${union} → 最终 ${detail.matches || 0}${fusion}${parallel}`;
    }
    const pool = detail.pool ? ` · 池 ${detail.pool}` : "";
    const recall = detail.recall != null
      && detail.recall !== detail.matches && detail.recall !== detail.pool
      ? ` · 召回 ${detail.recall}` : "";
    return `${detail.matches || 0} 个结果 · Top ${detail.top_k || 0}${pool}${recall}${duration}`;
  }
  if (key === "answer") return `${detail.model || "LLM"}${duration}`;
  if (key === "condense") {
    // 有历史才走这一步，展示改写前后的对照。
    return detail.standalone
      ? `“${detail.original}” → “${detail.standalone}”${duration}`
      : `已改写为独立问题${duration}`;
  }
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
  } else if (stages.rerank && stages.rerank.status === "running") {
    setRunStatusLoading(true, "正在重排检索结果...");
  } else if (stages.encode && stages.encode.status === "running") {
    setRunStatusLoading(true, "正在编码问题...");
  } else if (job.status === "queued") {
    setRunStatusLoading(true, "等待处理...");
  }
}

// --- asking ---------------------------------------------------------------

// Thrown when the user switched conversation while an answer was in flight.
class StaleAnswer extends Error {}

async function waitForAnswer(jobId, epoch) {
  while (true) {
    if (convState.epoch !== epoch) throw new StaleAnswer();
    const job = await requestJson(`/api/answer-jobs/${jobId}`);
    // Re-check after the await: the user may have switched mid-request.
    if (convState.epoch !== epoch) throw new StaleAnswer();

    state.answerJob = job;
    if (["completed", "degraded"].includes(job.stages?.vector_search?.status)) {
      state.vectorHits = job.vector_hits || [];
    }
    renderRetrieval();
    // 答案还在生成就边生成边显示（流式）。
    if (job.status !== "completed" && job.answer) renderStreamingAnswer(job.answer);
    renderProcess();
    updateRunStatusFromJob(job);
    if (job.status === "completed") {
      finalizeAnswer(job);
      return job;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "问答处理失败");
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
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

  const params = requestParams();
  conversationModel ||= params.llm_model || "";
  if (workspaceMode === "aggregate") {
    await submitAggregate(question, params);
    return;
  }

  lastQuestionEl = addChatMessage("question", question);
  el.question.value = "";
  el.askButton.disabled = true;
  streamingAnswerEl = null;
  lastStreamedAnswer = "";
  processExpanded = false;
  el.processToggle.setAttribute("aria-expanded", "false");
  el.processToggle.textContent = "展开";
  // 新提问回到最新：清掉 pin 的历史轮。
  clearSelectedTurn();
  setRunStatusLoading(true, "正在处理问题...");
  state.answerJob = { retrieval_method: getRetrievalMethod() };
  state.vectorHits = [];
  renderProcess();
  renderRetrievalLoading();

  let epoch = convState.epoch;
  try {
    // 首问时才真正建对话，避免点开页面就留下一堆空对话。
    const conversationId = await ensureConversation();
    epoch = convState.epoch;
    renderConversations();

    const job = await requestJson("/api/answer-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
        ...params,
      }),
    });
    state.answerJob = job;
    renderProcess();
    const result = await waitForAnswer(job.id, epoch);
    if (convState.epoch !== epoch) return;

    state.vectorHits = result.vector_hits || [];
    renderRetrieval();
    const recall = result.stages?.vector_search?.detail?.recall_detail;
    const successPath = recall?.status === "degraded" ? recall.succeeded_paths?.[0] : "";
    const degradedLabel = RECALL_PATH_LABELS[successPath] || "";
    setRunStatusLoading(false, degradedLabel
      ? `回答完成（检索降级：仅 ${degradedLabel}）`
      : "回答完成");
  } catch (error) {
    if (error instanceof StaleAnswer) return;
    addChatMessage("error", error.message);
    renderRetrieval();
    setRunStatusLoading(false, "回答失败");
  } finally {
    el.askButton.disabled = false;
    el.question.focus();
    // 标题与轮数由服务端生成。即使这轮答复已过期（用户切走了），列表也要刷新，
    // 否则刚建的对话要等到下次刷新才出现在列表里。
    loadConversations();
  }
}

function clearChat() {
  // 只清空当前对话的显示，不动服务端记录。
  convState.epoch += 1;
  resetChatStream();
}

// --- conversations --------------------------------------------------------

async function loadConversations() {
  try {
    const payload = await requestJson("/api/conversations");
    convState.items = payload.conversations || [];
  } catch (_) {
    convState.items = [];
  }
  renderConversations();
}

function renderConversations() {
  el.conversationList.replaceChildren();
  if (!convState.items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无对话";
    el.conversationList.append(empty);
    return;
  }

  for (const conversation of convState.items) {
    const item = document.createElement("article");
    item.className = `conv-item${conversation.id === convState.currentId ? " current" : ""}`;

    const open = document.createElement("button");
    open.type = "button";
    open.className = "conv-open";
    const title = document.createElement("strong");
    title.textContent = conversation.title || "新对话";
    const meta = document.createElement("span");
    meta.textContent = `${formatTime(conversation.updated_at)} · ${conversation.turn_count} 轮`;
    open.append(title, meta);
    open.addEventListener("click", () => openConversation(conversation.id));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "row-action";
    remove.title = "删除对话";
    remove.setAttribute("aria-label", `删除 ${conversation.title}`);
    remove.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>';
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteConversation(conversation.id);
    });

    item.append(open, remove);
    el.conversationList.append(item);
  }
}

function resetChatStream() {
  el.chatStream.replaceChildren();
  streamingAnswerEl = null;
  lastStreamedAnswer = "";
  lastQuestionEl = null;
  clearSelectedTurn();
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
  renderRetrieval();
  el.runStatus.textContent = "就绪";
}

function selectConversation(conversationId) {
  convState.currentId = conversationId;
  // Anything still polling for the previous conversation is now stale.
  convState.epoch += 1;
  renderConversations();
}

function newConversation() {
  selectConversation("");
  conversationModel = "";
  setupModelSwitcher();
  resetChatStream();
  el.question.focus();
}

async function deleteConversation(conversationId) {
  if (!confirm("删除这个对话及其全部记录？")) return;
  try {
    await requestJson(`/api/conversations/${encodeURIComponent(conversationId)}`, {
      method: "DELETE",
    });
    if (convState.currentId === conversationId) newConversation();
    await loadConversations();
    showToast("已删除对话");
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function openConversation(conversationId) {
  if (conversationId === convState.currentId) return;
  selectConversation(conversationId);
  conversationModel = "";
  setupModelSwitcher();
  resetChatStream();

  try {
    const conversation = await requestJson(
      `/api/conversations/${encodeURIComponent(conversationId)}`,
    );
    if (convState.currentId !== conversationId) return;
    conversationModel = conversation.turns?.at(-1)?.retrieval?.llm_model || "";
    setupModelSwitcher();
    const empty = el.chatStream.querySelector(".empty-chat");
    if (empty) empty.remove();

    const turns = conversation.turns || [];
    turns.forEach((turn, index) => {
      const turnData = {
        key: `history:${conversationId}:${index}`,
        hits: turn.hits || [],
        answer: turn.answer || "",
        retrieval: {
          ...(turn.retrieval || {}),
          retrieval_method: turn.retrieval_method
            || turn.retrieval?.retrieval_method,
        },
      };
      const questionEl = addChatMessage("question", turn.question, turnData);
      const answerEl = addChatMessage("answer", turn.answer || "（未记录答案）", turnData);
      renderRetrievedDocs(answerEl, turnData.hits, turnData.answer);
      questionEl._turn = turnData;
    });
    const last = turns.at(-1);
    state.vectorHits = last?.hits || [];
    // pin 到最后一轮，让 used 标记基于该轮答案（而不是空的 answerJob）。
    clearSelectedTurn();
    if (last) {
      selectedTurnHits = last.hits || [];
      selectedTurnAnswer = last.answer || "";
      selectedTurnRetrieval = {
        ...(last.retrieval || {}),
        retrieval_method: last.retrieval_method || last.retrieval?.retrieval_method,
      };
      selectedTurnKey = `history:${conversationId}:${turns.length - 1}`;
      const lastAnswerEl = el.chatStream.lastElementChild;
      if (lastAnswerEl) {
        selectedTurnEl = lastAnswerEl;
        lastAnswerEl.classList.add("selected");
      }
    }
    renderRetrieval();
    el.chatStream.scrollTop = el.chatStream.scrollHeight;
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function ensureConversation() {
  if (convState.currentId) return convState.currentId;
  const conversation = await requestJson("/api/conversations", { method: "POST" });
  convState.currentId = conversation.id;
  return conversation.id;
}

// --- knowledge base status ------------------------------------------------

// 点知识库 chip 展开这一库里有哪些文档。数据就是 state.documents——它对非管理员
// 已经按访问组过滤过了，所以这里不会漏出看不到的文档。
function renderKnowledgeBaseMenu() {
  const documents = state.documents || [];
  el.kbMenu.replaceChildren();

  const title = document.createElement("div");
  title.className = "menu-title";
  title.textContent = documents.length
    ? `${documents.length} 个文档 · 点击打开`
    : "这个知识库里还没有文档";
  el.kbMenu.append(title);

  if (!documents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "去「知识库」页上传文档后就能在这里看到";
    el.kbMenu.append(empty);
    return;
  }

  for (const doc of documents) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "menu-item";
    item.setAttribute("role", "menuitem");
    item.title = doc.source || doc.doc_id || "";
    const name = document.createElement("span");
    name.textContent = doc.title || doc.doc_id || "未命名文档";
    const chunks = document.createElement("span");
    chunks.className = "kb-chunks";
    chunks.textContent = `${doc.chunk_count || 0} 片`;
    item.append(name, chunks);
    item.addEventListener("click", () => {
      closeKnowledgeBaseMenu();
      // 与引用标签走同一条路：跳到知识库页并原地打开该文档
      router.navigate("library", { docId: doc.doc_id });
    });
    el.kbMenu.append(item);
  }
}

function toggleKnowledgeBaseMenu() {
  const open = el.kbMenu.hidden;
  if (open) renderKnowledgeBaseMenu();
  el.kbMenu.hidden = !open;
  el.kbStatus.setAttribute("aria-expanded", String(open));
}

function closeKnowledgeBaseMenu() {
  el.kbMenu.hidden = true;
  el.kbStatus.setAttribute("aria-expanded", "false");
}

function renderKnowledgeBaseStatus() {
  const documents = state.documents.length;
  const chunks = state.documents.reduce(
    (total, item) => total + Number(item.chunk_count || 0),
    0,
  );
  const current = knowledgeBases ? String(knowledgeBases.current || "") : "";
  const label = current ? `知识库 ${current}` : "知识库";

  // 策略里定义了、但本实例不服务的库。用一个虚线 chip 明说，而不是列成「可用」——
  // 一个实例只服务一个知识库，切过去是查不到东西的。
  const others = (knowledgeBases && knowledgeBases.other_knowledge_bases) || [];
  el.kbOthers.hidden = !others.length;
  el.kbOthers.textContent = others.length
    ? `策略另有 ${others.length} 个库（${others.join("、")}），本实例只服务 ${current}`
    : "";

  if (knowledgeBases && knowledgeBases.accessible === false) {
    // 访问组对不上时检索会返回空。这是策略配错，不能让表现成「莫名其妙搜不到」。
    el.kbStatus.className = "kb-status warn";
    el.kbStatus.textContent = `${label}：你不属于它的访问组，检索会返回空`;
    return;
  }
  if (!documents) {
    el.kbStatus.className = "kb-status warn";
    el.kbStatus.textContent = `${label} 还没有文档，答案会缺少依据`;
    return;
  }
  el.kbStatus.className = "kb-status";
  el.kbStatus.textContent = `${label} · ${documents} 个文档 · ${chunks} 个切片`;

  // 菜单开着的时候文档变了（比如刚上传完），跟着刷新，免得看到过期列表
  if (el.kbMenu && !el.kbMenu.hidden) renderKnowledgeBaseMenu();
}

// --- view contract --------------------------------------------------------

// --- 面板尺寸 --------------------------------------------------------------

// 四个手柄：左栏、右栏、检索参数栏、输入区。拖动、双击复位、方向键、持久化都在
// js/resizer.js 里 —— 四处交互一样，写四份迟早分叉（尤其方向键的符号：右栏往右
// 按是变窄，左栏往右按是变宽）。
const CONV_LIMITS = { min: 180, max: 460, default: 232 };
const RIGHT_LIMITS = { min: 260, max: 620, default: 360 };

function setupResizers() {
  const layout = document.querySelector(".ask-layout");
  const measure = (selector, prop) => () =>
    document.querySelector(selector)?.getBoundingClientRect()[prop] || 0;

  enableResize({
    handle: el.resizerLeft,
    axis: "x",
    growsWithDrag: 1, // 左栏在左边：往右拖变宽
    limits: CONV_LIMITS,
    read: measure(".conv-rail", "width"),
    apply: (px) => layout.style.setProperty("--rail-conv", `${px}px`),
    field: "conv",
  });

  enableResize({
    handle: el.resizerRight,
    axis: "x",
    growsWithDrag: -1, // 右栏在右边：往右拖变窄
    limits: RIGHT_LIMITS,
    read: measure(".ask-rail", "width"),
    apply: (px) => layout.style.setProperty("--rail-right", `${px}px`),
    field: "right",
  });

  // 竖向手柄拖的是「下方那个区块」的高度，对话区继续用 flex:1 吸收剩余空间——
  // 这样怎么拖都不会在底部留出空隙。复位值取装手柄前量到的自然高度：那才是
  // 「没有被调过」的样子，猜一个像素数只会让复位看起来没反应。
  const vertical = [
    [el.resizerBar, ".settings-bar", "--bar-height", 44, 240, "bar"],
    [el.resizerForm, ".ask-form", "--form-height", 64, 320, "form"],
  ];
  for (const [handle, selector, variable, min, max, field] of vertical) {
    const element = document.querySelector(selector);
    const natural = element.getBoundingClientRect().height;
    enableResize({
      handle,
      axis: "y",
      growsWithDrag: -1, // 往上拖变高
      limits: { min, max, default: Math.max(min, Math.round(natural)) },
      read: measure(selector, "height"),
      apply: (px) => layout.style.setProperty(variable, `${px}px`),
      field,
    });
  }
}

// --- 知识库 ---------------------------------------------------------------
// --- 知识库 ---------------------------------------------------------------

// **一个实例只服务一个知识库**（knowledge_base_id 是进程级设置，检索时当 where
// 过滤）。所以策略里的其它库只能如实说成「定义了、但本实例不服务」——列成
// 「可用知识库」会让人以为切过去就能查到东西。
let knowledgeBases = null;

async function loadKnowledgeBases() {
  try {
    knowledgeBases = await requestJson("/api/knowledge-bases");
  } catch (_) {
    knowledgeBases = null;
  }
  renderKnowledgeBaseStatus();
}

// --- 对话模型 -------------------------------------------------------------

// 沿用管理员模型选择权限；会话选择只随请求发送，不写全局配置。
const llmPicker = createModelPicker({
  select: el.llmModel,
  custom: el.llmModelCustom,
  hint: el.llmHint,
  target: "llm",
});
el.llmModelCustom.placeholder = "搜索模型名称或 ID；自定义按 Enter 确认";

let modelSwitcherReady = false;

async function setupModelSwitcher(status = state.status) {
  const allowed = canConfigure();
  if (allowed && document.querySelector("#ask-model-dialog").open) return;
  const model = (allowed && conversationModel) || status?.llm_model || "";
  // 开机时身份还没到（state.identity.user 为 null），canConfigure() 按开放模式算成
  // 允许，于是选择器会先显示出来；等拿到身份发现其实是普通用户，靠 onIdentity 收回。
  el.llmModel.hidden = !allowed;
  const trigger = document.querySelector("#ask-model-open");
  trigger.hidden = !allowed;
  trigger.textContent = model || "选择模型";
  trigger.title = `当前会话模型：${model || "未配置"}`;
  // 不能改的人也要知道现在用的是哪个模型——它直接决定答案质量。所以给一个只读的
  // chip，而不是干脆什么都不显示。
  el.llmReadonly.hidden = allowed;
  el.llmReadonly.textContent = model ? `模型 ${model}` : "";
  el.llmReadonly.title = model ? `当前对话模型：${model}（由管理员配置）` : "";
  if (!allowed) {
    llmPicker.setValue(model);
    // 下拉、自定义输入框、提示都是「能改」才该出现的东西，一个都不能留。此前只收
    // 了下拉：开机时身份还没到，canConfigure() 按开放模式算成允许，于是去拉模型
    // 列表（非管理员必然 403），而错误分支会把自定义框显示出来——等身份到了就
    // 留下一个没人能用的输入框和一句「可在自定义里手动输入」。
    document.querySelector("#ask-model-dialog").close();
    el.llmHint.textContent = "";
    el.llmHint.classList.remove("error");
    return;
  }
  llmPicker.setValue(model);
  if (modelSwitcherReady) {
    reflectModel(model);
    return;
  }
  modelSwitcherReady = true;
  await llmPicker.load();
  if (!document.querySelector("#ask-model-dialog").open) {
    reflectModel(conversationModel || status?.llm_model || "");
  }
}

/** 把下拉同步到某个模型：列表里有就选中，没有就保持自定义输入。 */
function reflectModel(model) {
  if (!model) return;
  const known = [...el.llmModel.options].some((option) => option.value === model);
  if (!known) return;
  el.llmModel.value = model;
  resetSelectSearch(el.llmModel);
  llmPicker.syncFromSelect();
}

function bindModelSwitcher() {
  const trigger = document.querySelector("#ask-model-open");
  const dialog = document.querySelector("#ask-model-dialog");
  trigger.addEventListener("click", async () => {
    dialog.showModal();
    el.llmModelCustom.focus();
    el.llmModelCustom.select();
    await llmPicker.load();
  });
  dialog.addEventListener("close", () => {
    resetSelectSearch(el.llmModel);
    trigger.focus();
  });
  dialog.addEventListener("keydown", event => {
    if (event.key === "Escape" && !event.isComposing) {
      event.preventDefault();
      dialog.close();
    }
  });
  el.llmModel.addEventListener("change", () => {
    llmPicker.syncFromSelect();
    conversationModel = llmPicker.value();
    trigger.textContent = conversationModel;
    trigger.title = `当前会话模型：${conversationModel}`;
    if (dialog.open) dialog.close();
    showToast(`当前会话后续请求使用 ${conversationModel}`);
  });
}

export default {
  name: "ask",

  mount() {
    if (mounted) return;
    mounted = true;

    setupResizers();
    setWorkspaceMode("ask");
    document.querySelector("#mode-ask").addEventListener("click", () => setWorkspaceMode("ask"));
    document.querySelector("#mode-aggregate").addEventListener("click", () => setWorkspaceMode("aggregate"));
    bindModelSwitcher();
    document.addEventListener("invalid", event => {
      if (event.target.form === el.askForm) {
        const details = event.target.closest("details");
        if (details) details.open = true;
      }
    }, true);
    loadKnowledgeBases();

    el.kbStatus.addEventListener("click", toggleKnowledgeBaseMenu);
    document.addEventListener("click", (event) => {
      if (!el.kbMenu.hidden && !el.kbPicker.contains(event.target)) {
        closeKnowledgeBaseMenu();
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !el.kbMenu.hidden) closeKnowledgeBaseMenu();
    });

    // Empty states for the evidence rail before the first question.
    renderProcess();
    renderRetrieval();

    el.askForm.addEventListener("submit", ask);
    el.processToggle.addEventListener("click", toggleProcess);
    el.retrievalToggle.addEventListener("click", toggleRetrievalStages);
    el.retrievalDiagnosticToggle.addEventListener("click", toggleRetrievalDiagnostic);
    el.retrievalDiagnosticCopy.addEventListener("click", () => copyRetrievalDiagnostic(el.retrievalDiagnosticCopy));
    el.rerankPreferencesRetry.addEventListener("click", loadRetrievalPreferences);
    el.clearChatButton.addEventListener("click", clearChat);
    el.newConversation.addEventListener("click", newConversation);
    // 点某条问答：右侧证据栏切到那一轮的检索切片（再点同一条取消）。
    el.chatStream.addEventListener("click", (event) => {
      const message = event.target.closest(".chat-message");
      if (!message || !message._turn) return;
      if (selectedTurnEl === message) {
        clearSelectedTurn();
        renderRetrieval();
        return;
      }
      if (selectedTurnEl) selectedTurnEl.classList.remove("selected");
      selectedTurnEl = message;
      message.classList.add("selected");
      selectedTurnHits = message._turn.hits || [];
      selectedTurnAnswer = message._turn.answer || "";
      selectedTurnRetrieval = message._turn.retrieval || {};
      selectedTurnKey = message._turn.key || "";
      renderRetrieval();
    });
    el.question.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        el.askForm.requestSubmit();
      }
    });

    on("documents", renderKnowledgeBaseStatus);
    on("identity", () => {
      // Conversations are per-owner, so switching user means a different list.
      newConversation();
      loadConversations();
      // 知识库的可见性随身份而变，也要重取
      loadKnowledgeBases();
    });

    for (const element of [
      el.retrievalTopK,
      el.similarityThreshold,
      el.fusionK,
      el.fusionAlpha,
    ]) {
      element.addEventListener("input", () => {
        state.retrievalInitialized = true;
        updateRetrievalControls();
        if (!state.answerJob) renderProcess();
      });
    }
    for (const element of [el.retrievalTopK, el.rerankPoolSize, el.rerankScoreThreshold]) {
      element.addEventListener("change", saveRetrievalPreferences);
    }
    for (const element of [el.rerankEnabled, el.rerankScoreFilterEnabled]) {
      element.addEventListener("change", () => {
        state.retrievalInitialized = true;
        updateRetrievalControls();
        saveRetrievalPreferences();
      });
    }
    el.retrievalMethod.addEventListener("change", () => {
      state.retrievalInitialized = true;
      updateRetrievalControls();
      if (!state.answerJob) renderProcess();
    });
    el.fusionMethod.addEventListener("change", () => {
      state.retrievalInitialized = true;
      updateRetrievalControls();
      if (!state.answerJob) renderProcess();
    });

    // 先按 HTML 默认值定一次显隐/禁用，别等服务端状态回来才对齐。
    updateRetrievalControls();
  },

  show(params) {
    if (params?.mode) setWorkspaceMode(params.mode);
    renderKnowledgeBaseStatus();
    // 管理员可能在设置页刚改过重排服务；重新进入工作区时刷新账号偏好与配置状态。
    if (retrievalPreferences) loadRetrievalPreferences();
    // 状态还没到就别装：此时身份也未知，初始化只会拿着空模型去拉一次列表
    // （非管理员必被 403），等 onStatus / onIdentity 到了自然会装。
    if (state.status) setupModelSwitcher(state.status);
    loadConversations();
    el.chatStream.scrollTop = el.chatStream.scrollHeight;
    el.question.focus();
  },

  // Retrieval defaults come from the server, but only until the user touches
  // them — re-entering the page must not clobber their edits.
  onIdentity() {
    // 身份到了才谈得上判断该不该显示模型选择器（开机时它按开放模式先显示了出来）。
    setupModelSwitcher();
    retrievalPreferences = null;
    el.rerankEnabled.checked = false;
    el.rerankScoreFilterEnabled.checked = false;
    updateRerankControls();
    loadRetrievalPreferences();
  },

  onStatus(status) {
    setupModelSwitcher(status);
    if (!status || state.retrievalInitialized) return;
    setRetrievalMethod(status.retrieval_method || "vector");
    el.retrievalTopK.value = retrievalPreferences?.top_k ?? status.retrieval_top_k ?? 20;
    el.candidatePool.value = status.candidate_pool_size ?? 100;
    el.similarityThreshold.value = status.similarity_threshold ?? 0.35;
    setFusionMethod(status.fusion_method || "rrf");
    el.fusionK.value = status.fusion_k ?? 60;
    el.fusionAlpha.value = status.fusion_alpha ?? 0.5;
    state.retrievalInitialized = true;
    updateRetrievalControls();
  },
};
