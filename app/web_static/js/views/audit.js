// 审计页：向量库写入记录，支持按事件类型与操作者筛选（筛选在客户端做，
// 因为 /api/audit 只接受 limit）。

import { formatTime, requestJson, state } from "../core.js";

const el = {
  list: document.querySelector("#audit-list"),
  note: document.querySelector("#audit-note"),
  status: document.querySelector("#audit-status"),
  refresh: document.querySelector("#audit-refresh"),
  eventFilter: document.querySelector("#audit-event-filter"),
  actorFilter: document.querySelector("#audit-actor-filter"),
};

const EVENT_LABELS = {
  "ingest.directory": "目录入库",
  "ingest.files": "上传入库",
  "delete.documents": "删除文档",
  "staging.submitted": "提交待审",
  "staging.approved": "审核通过",
  "staging.rejected": "审核拒绝",
};

const FAILURE_STAGE_LABELS = {
  request: "请求校验",
  multipart_parse: "解析上传请求",
  queue: "加入上传队列",
  staging: "暂存待审",
  load: "文档解析",
  ocr: "OCR",
  split: "切片",
  vector_write: "向量写入",
  metadata_lookup: "读取文档元数据",
  vector_delete: "向量删除",
};

let mounted = false;
let entries = [];

function renderEntries() {
  const wantedEvent = el.eventFilter.value;
  const wantedActor = el.actorFilter.value;
  const visible = entries.filter(
    (entry) =>
      (!wantedEvent || entry.event === wantedEvent) &&
      (!wantedActor || entry.actor === wantedActor),
  );

  el.list.replaceChildren();
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = entries.length ? "没有符合筛选条件的记录" : "暂无写入记录";
    el.list.append(empty);
    return;
  }
  for (const entry of visible) el.list.append(createRow(entry));
  el.status.textContent = `显示 ${visible.length} / ${entries.length} 条`;
}

function deletedDocumentLabels(entry) {
  if (Array.isArray(entry.deleted_documents) && entry.deleted_documents.length) {
    return entry.deleted_documents.map((document) => {
      const docId = String(document.doc_id || "");
      const title = String(document.title || "");
      return title && title !== docId ? `${title}（${docId}）` : docId;
    }).filter(Boolean);
  }
  // 老日志只记了 doc_ids；文档已经删掉，无法再补回标题。至少把真实 id 展示出来，
  // 不伪造一份看似完整的历史。
  return Array.isArray(entry.doc_ids) ? entry.doc_ids.filter(Boolean) : [];
}

function operationType(entry) {
  if (entry.event === "delete.documents") return "delete";
  if (String(entry.event || "").startsWith("ingest.")
      || String(entry.event || "").startsWith("staging.")) return "upload";
  return "other";
}

const STATUS_LABELS = {
  success: "成功",
  failed: "失败",
  partial: "部分完成",
  cancelled: "已取消",
};

function entryStatus(entry) {
  // 历史日志没有 status，按成功兼容；只有显式异常状态才改变展示。
  return ["failed", "partial", "cancelled"].includes(entry.status)
    ? entry.status
    : "success";
}

function createRow(entry) {
  const status = entryStatus(entry);
  const operation = operationType(entry);
  const row = document.createElement("article");
  row.className = `audit-row operation-${operation} status-${status}`;

  const head = document.createElement("div");
  head.className = "audit-head";
  const headMain = document.createElement("div");
  headMain.className = "audit-head-main";
  const event = document.createElement("strong");
  event.className = "audit-event";
  event.textContent = EVENT_LABELS[entry.event] || String(entry.event || "");
  const result = document.createElement("span");
  result.className = `audit-result ${status}`;
  result.textContent = STATUS_LABELS[status] || status;
  headMain.append(event, result);
  const time = document.createElement("span");
  time.className = "audit-time";
  time.textContent = formatTime(entry.timestamp);
  head.append(headMain, time);

  const meta = document.createElement("div");
  meta.className = "audit-meta";
  const deleted = entry.event === "delete.documents"
    ? deletedDocumentLabels(entry)
    : [];
  meta.textContent = [
    `操作者 ${entry.actor}`,
    deleted.length ? `文档 ${deleted.join("、")}` : "",
    Array.isArray(entry.files) && entry.files.length ? entry.files.join("、") : "",
  ]
    .filter(Boolean)
    .join(" · ");

  const stats = document.createElement("div");
  stats.className = "audit-stats";
  const bits = [];
  if (entry.uploader) bits.push(`提交者 ${entry.uploader}`);
  if (entry.chunks !== undefined) bits.push(`${entry.chunks} 切片`);
  if (entry.documents !== undefined) bits.push(`${entry.documents} 页`);
  if (entry.deleted_chunks !== undefined) bits.push(`删除 ${entry.deleted_chunks} 切片`);
  if (Array.isArray(entry.access_groups) && entry.access_groups.length) {
    bits.push(`访问组 ${entry.access_groups.join("、")}`);
  } else if (entry.access_group) {
    bits.push(`访问组 ${entry.access_group}`);
  }
  if (entry.structurize_provider) bits.push(entry.structurize_provider);
  if (entry.embedding_model) bits.push(entry.embedding_model);
  stats.textContent = bits.join(" · ");

  row.append(head, meta, stats);
  if (status === "failed") {
    const failure = document.createElement("div");
    failure.className = "audit-error";
    const stage = FAILURE_STAGE_LABELS[entry.failure_stage] || entry.failure_stage || "";
    failure.textContent = [
      stage ? `失败阶段 ${stage}` : "",
      `原因 ${entry.error || "未记录原因"}`,
    ]
      .filter(Boolean)
      .join(" · ");
    row.append(failure);
  }
  return row;
}

// Rebuild the filter options from whatever the log actually contains, keeping
// the current selection when it is still valid.
function renderFilters() {
  const keepEvent = el.eventFilter.value;
  const keepActor = el.actorFilter.value;

  const events = [...new Set(entries.map((entry) => entry.event).filter(Boolean))].sort();
  const actors = [...new Set(entries.map((entry) => entry.actor).filter(Boolean))].sort();

  fillSelect(el.eventFilter, events, keepEvent, (value) => EVENT_LABELS[value] || value);
  fillSelect(el.actorFilter, actors, keepActor, (value) => value);
}

function fillSelect(select, values, keep, label) {
  select.replaceChildren();
  select.append(new Option("全部", ""));
  for (const value of values) select.append(new Option(label(value), value));
  if (values.includes(keep)) select.value = keep;
}

async function loadAudit() {
  el.status.textContent = "加载中…";
  try {
    const payload = await requestJson("/api/audit?limit=200");
    entries = payload.entries || [];
    el.note.textContent = `日志文件：${payload.path}`;
    renderFilters();
    renderEntries();
  } catch (error) {
    entries = [];
    el.note.textContent = "";
    el.list.replaceChildren();
    const failed = document.createElement("div");
    failed.className = "empty-list error";
    failed.textContent = error.message;
    el.list.append(failed);
    el.status.textContent = "";
  }
}

export default {
  name: "audit",

  mount() {
    if (mounted) return;
    mounted = true;
    el.refresh.addEventListener("click", loadAudit);
    el.eventFilter.addEventListener("change", renderEntries);
    el.actorFilter.addEventListener("change", renderEntries);
  },

  show() {
    loadAudit();
  },

  onIdentity() {
    // Switching to a non-admin while on this page bounces away in core.
    if (!state.identity.user && !state.identity.isAdmin) entries = [];
  },
};
