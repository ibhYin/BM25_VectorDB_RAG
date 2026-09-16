// 问答页：对话流、检索参数、过程与命中证据。

import {
  canConfigure,
  createSpinner,
  formatTime,
  on,
  refreshStatus,
  requestJson,
  router,
  shortDocLabel,
  showToast,
  state,
} from "../core.js";
import { createVectorHitCard } from "../hits.js";
import { CUSTOM_MODEL, createModelPicker } from "../model-picker.js";
import { enableResize } from "../resizer.js";

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
  kbStatus: document.querySelector("#kb-status"),
  kbOthers: document.querySelector("#kb-others"),
  newConversation: document.querySelector("#new-conversation"),
  conversationList: document.querySelector("#conversation-list"),

  llmModel: document.querySelector("#ask-llm-model"),
  llmModelCustom: document.querySelector("#ask-llm-model-custom"),
  llmHint: document.querySelector("#ask-llm-hint"),

  resizerLeft: document.querySelector("#resizer-left"),
  resizerRight: document.querySelector("#resizer-right"),
  resizerBar: document.querySelector("#resizer-bar"),
  resizerForm: document.querySelector("#resizer-form"),
};

let mounted = false;

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

function renderContextLoading() {
  el.contextList.replaceChildren();
  const loading = document.createElement("div");
  loading.className = "loading-state";
  loading.append(createSpinner(), document.createTextNode("检索中"));
  el.contextList.append(loading);
}

// --- process rail ---------------------------------------------------------

const processDefinitions = [
  ["condense", "问题改写"],
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
    const result = await waitForAnswer(job.id, epoch);
    if (convState.epoch !== epoch) return;

    state.vectorHits = result.vector_hits || [];
    addChatMessage("answer", result.answer || "未生成答案");
    renderContext();
    setRunStatusLoading(false, "回答完成");
  } catch (error) {
    if (error instanceof StaleAnswer) return;
    addChatMessage("error", error.message);
    renderContext();
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

function selectConversation(conversationId) {
  convState.currentId = conversationId;
  // Anything still polling for the previous conversation is now stale.
  convState.epoch += 1;
  renderConversations();
}

function newConversation() {
  selectConversation("");
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
  resetChatStream();

  try {
    const conversation = await requestJson(
      `/api/conversations/${encodeURIComponent(conversationId)}`,
    );
    const empty = el.chatStream.querySelector(".empty-chat");
    if (empty) empty.remove();

    for (const turn of conversation.turns || []) {
      addChatMessage("question", turn.question);
      addChatMessage("answer", turn.answer || "（未记录答案）");
    }
    const last = (conversation.turns || []).at(-1);
    state.vectorHits = last?.hits || [];
    renderContext();
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

// 门禁用 canConfigure() 而不是 state.identity.isAdmin：本地直连后端（不经过
// scripts/dev_proxy.py）时没有 /api/me，isAdmin 会是 false，而开放模式下后端是
// 允许改配置的 —— 用 isAdmin 会让这个选择器永远不显示。
const llmPicker = createModelPicker({
  select: el.llmModel,
  custom: el.llmModelCustom,
  hint: el.llmHint,
  target: "llm",
});

let modelSwitcherReady = false;

async function setupModelSwitcher(status = state.status) {
  // 开机时身份还没到（state.identity.user 为 null），canConfigure() 按开放模式算成
  // 允许，于是选择器会先显示出来；等拿到身份发现其实是普通用户，靠 onIdentity 收回。
  const allowed = canConfigure();
  el.llmModel.hidden = !allowed;
  if (!allowed) return;
  const model = status?.llm_model || "";
  llmPicker.setValue(model);
  if (modelSwitcherReady) {
    reflectModel(model);
    return;
  }
  modelSwitcherReady = true;
  await llmPicker.load();
  reflectModel(model);
}

/** 把下拉同步到某个模型：列表里有就选中，没有就保持自定义输入。 */
function reflectModel(model) {
  if (!model) return;
  const known = [...el.llmModel.options].some((option) => option.value === model);
  if (!known) return;
  el.llmModel.value = model;
  llmPicker.syncFromSelect();
}

async function switchModel() {
  const model = llmPicker.value();
  if (!model) return;
  try {
    await requestJson("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ llm_model: model }),
    });
    // 说「保存在服务器上」而不是「对后续问答生效」：answer_question 是在生成那一
    // 步才读 llm_model 的，正在跑的问答也可能用上新模型。
    showToast(`已切换到 ${model}，保存在服务器上，对所有人生效`);
    refreshStatus();
  } catch (error) {
    showToast(error.message, "error");
  }
}

function bindModelSwitcher() {
  el.llmModel.addEventListener("change", () => {
    llmPicker.syncFromSelect();
    if (el.llmModel.value !== CUSTOM_MODEL) switchModel();
  });
  el.llmModelCustom.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    switchModel();
  });
}

export default {
  name: "ask",

  mount() {
    if (mounted) return;
    mounted = true;

    setupResizers();
    bindModelSwitcher();
    loadKnowledgeBases();

    // Empty states for the evidence rail before the first question.
    renderProcess();
    renderContext();

    el.askForm.addEventListener("submit", ask);
    el.clearChatButton.addEventListener("click", clearChat);
    el.newConversation.addEventListener("click", newConversation);
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
    renderKnowledgeBaseStatus();
    setupModelSwitcher(state.status);
    loadConversations();
    el.chatStream.scrollTop = el.chatStream.scrollHeight;
    el.question.focus();
  },

  // Retrieval defaults come from the server, but only until the user touches
  // them — re-entering the page must not clobber their edits.
  onIdentity() {
    // 身份到了才谈得上判断该不该显示模型选择器（开机时它按开放模式先显示了出来）。
    setupModelSwitcher();
  },

  onStatus(status) {
    if (!status || state.retrievalInitialized) return;
    setRetrievalMethod(status.retrieval_method || "vector");
    el.retrievalTopK.value = status.retrieval_top_k ?? 20;
    el.candidatePool.value = status.candidate_pool_size ?? 100;
    el.similarityThreshold.value = status.similarity_threshold ?? 0.35;
    state.retrievalInitialized = true;
    updateRetrievalControls();
    // 模型名来自 status，所以选择器要等它到了再装
    setupModelSwitcher(status);
  },
};
