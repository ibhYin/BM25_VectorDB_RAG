// 评估页：标出「本该被召回的切片」，再用指标判断检索改动到底有没有用。
//
// 标注是这件事里唯一需要人力的部分，也是整条链路的瓶颈——所以这个页面的重点是把
// 标注做得尽量省事：展开一道题就能看到它召回了什么，勾选即可，正文指纹由后端从
// 索引里现取，不用手算。
//
// 跑批与对比仍在 CLI（`eval run --save` / `eval compare`）。页面只负责看清召回与
// 标注，那两件事用命令行做更顺手，也能写进脚本。

import { requestJson, showToast, state } from "../core.js";
import { createVectorHitCard } from "../hits.js";

const el = {
  list: document.querySelector("#eval-list"),
  summary: document.querySelector("#eval-summary"),
  status: document.querySelector("#eval-status"),
  method: document.querySelector("#eval-method"),
  topK: document.querySelector("#eval-top-k"),
  ks: document.querySelector("#eval-ks"),
  importButton: document.querySelector("#eval-import"),
  runButton: document.querySelector("#eval-run"),
  refreshButton: document.querySelector("#eval-refresh"),
};

let mounted = false;
let questions = [];
/** 展开中的题目 id → 该题的召回结果，避免重复请求。 */
const expanded = new Map();

function retrievalBody(extra) {
  const body = { ...extra };
  // 空字符串会被后端判成非法方法，所以「跟随配置」时干脆不发这个字段
  if (el.method.value) body.retrieval_method = el.method.value;
  const topK = Number(el.topK.value);
  if (Number.isInteger(topK) && topK > 0) body.top_k = topK;
  return body;
}

// requestJson 把 options 直接交给 fetch，所以 body 必须自己序列化——
// 传对象会被 fetch 转成 "[object Object]"。
function postJson(url, payload) {
  return requestJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
}

function labelText(entry) {
  const count = (entry.relevant_chunk_ids || []).length;
  if (entry.stale_chunk_ids && entry.stale_chunk_ids.length) {
    return `⚠ ${entry.stale_chunk_ids.length} 条标注失效`;
  }
  return count ? `已标 ${count} 片` : "未标注";
}

// --- 渲染 -----------------------------------------------------------------

function renderList() {
  el.list.replaceChildren();
  if (!questions.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "还没有题目。点「从对话记录导入」，或先用 CLI 的 eval label 加题。";
    el.list.append(empty);
    return;
  }
  for (const entry of questions) el.list.append(createQuestionCard(entry));
  el.status.textContent = `${questions.length} 道题`;
}

function createQuestionCard(entry) {
  const card = document.createElement("article");
  card.className = "eval-card";

  const head = document.createElement("div");
  head.className = "eval-card-head";
  const question = document.createElement("strong");
  question.textContent = entry.question;
  const badge = document.createElement("span");
  badge.className = "eval-badge";
  badge.textContent = labelText(entry);
  if (entry.stale_chunk_ids && entry.stale_chunk_ids.length) {
    badge.classList.add("warning");
    badge.title =
      "这些标注在索引里找不到了（重建索引或源文件改名会让编号指向别的正文）。"
      + "它们已从指标里剔除，不会算作「没召回」。重新标一次即可：\n"
      + entry.stale_chunk_ids.join("\n");
  }
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "ghost-button";
  toggle.textContent = expanded.has(entry.id) ? "收起" : "看召回";
  head.append(question, badge, toggle);

  const body = document.createElement("div");
  body.className = "eval-card-body";

  toggle.addEventListener("click", () => {
    if (expanded.has(entry.id)) {
      expanded.delete(entry.id);
    } else {
      expanded.set(entry.id, null);
      loadHits(entry);
    }
    renderList();
  });

  card.append(head);
  if (expanded.has(entry.id)) {
    card.append(body);
    renderHitsInto(body, entry);
  }
  return card;
}

function renderHitsInto(body, entry) {
  const hits = expanded.get(entry.id);
  if (hits === null) {
    body.textContent = "检索中…";
    return;
  }
  if (!hits.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "这道题一条都没召回——检索可能确实有问题，也可能是语料里没有。";
    body.append(empty);
    return;
  }

  const labelled = new Set(entry.relevant_chunk_ids || []);
  const boxes = [];
  for (const hit of hits) {
    const wrapper = document.createElement("div");
    wrapper.className = "eval-hit";

    const box = document.createElement("input");
    box.type = "checkbox";
    box.dataset.chunkId = hit.chunk_id;
    box.checked = labelled.has(hit.chunk_id);
    box.setAttribute("aria-label", `把 ${hit.chunk_id} 标为相关`);
    boxes.push(box);

    wrapper.append(box, createVectorHitCard(hit));
    body.append(wrapper);
  }

  const actions = document.createElement("div");
  actions.className = "eval-actions";
  const save = document.createElement("button");
  save.type = "button";
  save.className = "ghost-button";
  save.textContent = "保存这道题的标注";
  save.addEventListener("click", () => saveLabel(entry, boxes, save));
  const hint = document.createElement("span");
  hint.className = "section-hint";
  hint.textContent = "只勾「本该被召回」的。没勾的部分不会被当成不相关，只是没标注。";
  actions.append(save, hint);
  body.append(actions);
}

async function saveLabel(entry, boxes, button) {
  const chunkIds = boxes.filter((box) => box.checked).map((box) => box.dataset.chunkId);
  button.disabled = true;
  try {
    const payload = await postJson(
      "/api/eval/label",
      retrievalBody({ question: entry.question, relevant_chunk_ids: chunkIds }),
    );
    if (payload.unresolved && payload.unresolved.length) {
      showToast(`有 ${payload.unresolved.length} 个切片在索引里找不到，已跳过`, "error");
    } else {
      showToast(`已保存：${payload.relevant_chunk_ids.length} 片`);
    }
    await load();
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function loadHits(entry) {
  try {
    const payload = await postJson("/api/eval/retrieve", retrievalBody({ question: entry.question }));
    expanded.set(entry.id, payload.hits || []);
  } catch (error) {
    expanded.set(entry.id, []);
    showToast(error.message, "error");
  }
  renderList();
}

// --- 汇总 -----------------------------------------------------------------

function renderSummary(payload) {
  el.summary.replaceChildren();
  if (!payload) return;

  if (!payload.labelled) {
    const hint = document.createElement("p");
    hint.className = "section-hint";
    hint.textContent = "一道题都没标注，没有可报告的数字。展开题目、勾选相关切片再保存。";
    el.summary.append(hint);
    return;
  }

  const table = document.createElement("table");
  table.className = "eval-summary-table";
  const head = document.createElement("tr");
  head.append(Object.assign(document.createElement("th"), { textContent: "指标" }));
  head.append(Object.assign(document.createElement("th"), { textContent: "数值" }));
  table.append(head);
  for (const [name, value] of Object.entries(payload.summary || {})) {
    const row = document.createElement("tr");
    row.append(Object.assign(document.createElement("td"), { textContent: formatMetric(name) }));
    row.append(
      Object.assign(document.createElement("td"), {
        textContent: value === null ? "—" : Number(value).toFixed(3),
      }),
    );
    table.append(row);
  }

  const note = document.createElement("p");
  note.className = "section-hint";
  note.textContent =
    `已标注 ${payload.labelled} / ${payload.total} 道题（未标注的不计入指标）。`
    + (payload.stale_total ? ` 另有 ${payload.stale_total} 条失效标注已剔除。` : "")
    + " Recall@K 是下界——人工标注必然标不全，判断好坏以 Hit@K 与 MRR 为准。";
  el.summary.append(table, note);
}

function formatMetric(name) {
  if (name === "mrr") return "MRR";
  const [kind, value] = name.split("@");
  if (kind === "hit") return `Hit@${value}`;
  return `Recall@${value}（下界）`;
}

// --- 动作 -----------------------------------------------------------------

async function load() {
  el.status.textContent = "加载中…";
  try {
    const payload = await requestJson("/api/eval");
    questions = payload.questions || [];
    // 收掉已经不在问题集里的展开状态
    for (const id of [...expanded.keys()]) {
      if (!questions.some((entry) => entry.id === id)) expanded.delete(id);
    }
    renderList();
  } catch (error) {
    questions = [];
    el.list.replaceChildren();
    const failed = document.createElement("div");
    failed.className = "empty-list error";
    failed.textContent = error.message;
    el.list.append(failed);
    el.status.textContent = "";
  }
}

async function importFromConversations() {
  el.importButton.disabled = true;
  try {
    const payload = await postJson("/api/eval/import");
    showToast(
      payload.imported
        ? `新增 ${payload.imported} 道题（已有的不重复添加）`
        : "没有新题（对话记录里的首轮问题都已经在问题集里）",
    );
    await load();
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    el.importButton.disabled = false;
  }
}

async function runBatch() {
  const ks = parseKs(el.ks.value);
  if (!ks.length) {
    showToast("K 要写成逗号分隔的正整数，例如 1,3,5", "error");
    return;
  }
  el.runButton.disabled = true;
  el.status.textContent = "跑批中…";
  try {
    const payload = await postJson("/api/eval/run", retrievalBody({ ks }));
    renderSummary(payload);
    el.status.textContent = `${payload.total} 道题`;
  } catch (error) {
    showToast(error.message, "error");
    el.status.textContent = "";
  } finally {
    el.runButton.disabled = false;
  }
}

function parseKs(value) {
  const parts = String(value || "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  const ks = [];
  for (const part of parts) {
    const value = Number(part);
    if (!Number.isInteger(value) || value <= 0) return [];
    ks.push(value);
  }
  return [...new Set(ks)].sort((a, b) => a - b);
}

export default {
  name: "eval",

  mount() {
    if (mounted) return;
    mounted = true;
    el.refreshButton.addEventListener("click", load);
    el.importButton.addEventListener("click", importFromConversations);
    el.runButton.addEventListener("click", runBatch);
  },

  show() {
    load();
  },

  onIdentity() {
    // Switching to a non-admin while on this page bounces away in core.
    if (!state.identity.user && !state.identity.isAdmin) {
      questions = [];
      expanded.clear();
    }
  },
};
