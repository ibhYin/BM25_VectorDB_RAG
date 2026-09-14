// 知识库页：统计、上传、任务队列、待审核、已入库文档与文档查看器。

import {
  clearListLoading,
  createSpinner,
  el as shell,
  emit,
  formatBytes,
  formatTime,
  on,
  refreshDocuments,
  renderListError,
  requestJson,
  setListLoading,
  showToast,
  state,
  uploadsToStaging,
} from "../core.js";

const el = {
  listView: document.querySelector("#library-list-view"),
  docView: document.querySelector("#library-doc-view"),

  documentCount: document.querySelector("#document-count"),
  chunkCount: document.querySelector("#chunk-count"),
  pendingCount: document.querySelector("#pending-count"),

  dropZone: document.querySelector("#drop-zone"),
  fileInput: document.querySelector("#file-input"),
  structurizeCheckbox: document.querySelector("#structurize-checkbox"),
  structurizeLabel: document.querySelector("#structurize-label"),
  chooseFilesButton: document.querySelector("#choose-files-button"),
  uploadPermission: document.querySelector("#upload-permission"),
  uploadLimit: document.querySelector("#upload-limit"),

  jobList: document.querySelector("#job-list"),
  queueSummary: document.querySelector("#queue-summary"),

  stagingSection: document.querySelector("#staging-section"),
  stagingList: document.querySelector("#staging-list"),
  stagingSummary: document.querySelector("#staging-summary"),

  documentList: document.querySelector("#document-list"),
  reloadDocumentsButton: document.querySelector("#reload-documents-button"),

  docBackButton: document.querySelector("#doc-back-button"),
  docTitle: document.querySelector("#doc-title"),
  docMeta: document.querySelector("#doc-meta"),
  docBody: document.querySelector("#doc-body"),
  docProgress: document.querySelector("#doc-progress"),
  docLoadMore: document.querySelector("#doc-load-more"),
  docViewMode: document.querySelector("#doc-view-mode"),
};

let mounted = false;
const stagingState = { items: [], groups: [] };

// --- stats ----------------------------------------------------------------

function renderStats() {
  const chunks = state.documents.reduce(
    (total, item) => total + Number(item.chunk_count || 0),
    0,
  );
  el.documentCount.textContent = String(state.documents.length);
  el.chunkCount.textContent = String(chunks);

  const pending = stagingState.items.filter((item) => item.status === "pending").length;
  el.pendingCount.textContent = String(pending);
  shell.navLibraryBadge.textContent = String(pending);
  shell.navLibraryBadge.hidden = !pending;
}

// --- upload ---------------------------------------------------------------

function renderUploadPermission() {
  const staged = uploadsToStaging();
  el.uploadPermission.hidden = !staged;
  el.uploadPermission.classList.toggle("info", staged);
  el.uploadPermission.textContent = staged
    ? "当前身份无直接入库权限，提交后需管理员审核"
    : "";
}

function openFilePicker() {
  el.fileInput.click();
}

function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;

  const invalid = files.filter((file) => !/\.(pdf|md|txt)$/i.test(file.name));
  const valid = files.filter((file) => /\.(pdf|md|txt)$/i.test(file.name));
  if (invalid.length) {
    showJobError(`不支持的文件：${invalid.map((file) => file.name).join("、")}`);
  }
  if (!valid.length) return;

  const form = new FormData();
  for (const file of valid) {
    form.append("files", file, file.name);
  }
  if (el.structurizeCheckbox.checked) form.append("structurize", "on");

  const uploadJob = {
    id: `upload-${Date.now()}`,
    filenames: valid.map((file) => file.name),
    status: "uploading",
    progress: 0,
    files: valid.length,
    documents: 0,
    chunks: 0,
    error: "",
  };
  state.jobs.unshift(uploadJob);
  renderJobs();

  const request = new XMLHttpRequest();
  request.open("POST", "/api/documents/upload");
  request.upload.addEventListener("progress", (event) => {
    if (event.lengthComputable) {
      uploadJob.progress = Math.round((event.loaded / event.total) * 100);
      renderJobs();
    }
  });

  request.addEventListener("load", () => {
    let payload = {};
    try {
      payload = JSON.parse(request.responseText);
    } catch (_) {
      payload = { error: "服务返回无效响应" };
    }

    if (request.status >= 200 && request.status < 300) {
      if (payload.staged) {
        // Non-admin submission: parked for review, so it is not an index job.
        const index = state.jobs.indexOf(uploadJob);
        if (index >= 0) state.jobs.splice(index, 1);
        renderJobs();
        showToast("已提交，等待管理员审核");
        loadStaging();
        return;
      }
      const index = state.jobs.indexOf(uploadJob);
      if (index >= 0) state.jobs.splice(index, 1, payload);
      startPolling();
    } else {
      uploadJob.status = "failed";
      uploadJob.error = payload.error || `上传失败 (${request.status})`;
      renderJobs();
    }
  });

  request.addEventListener("error", () => {
    uploadJob.status = "failed";
    uploadJob.error = "网络中断，文件未入队";
    renderJobs();
  });

  request.send(form);
}

// --- jobs -----------------------------------------------------------------

async function pollJobs() {
  try {
    const payload = await requestJson("/api/upload-jobs");
    const localUploads = state.jobs.filter((job) => job.id.startsWith("upload-"));
    const previous = new Map(state.jobs.map((job) => [job.id, job.status]));
    state.jobs = [...payload.jobs, ...localUploads];

    const completed = state.jobs.filter((job) => job.status === "completed");
    const changed = completed.some((job) => previous.get(job.id) !== "completed");
    renderJobs();
    if (changed) {
      emit("documents:refresh");
    }

    const active = state.jobs.some(
      (job) => job.status === "queued" || job.status === "indexing",
    );
    if (!active) stopPolling();
  } catch (_) {
    renderListError(el.jobList, "任务状态不可用");
    el.queueSummary.textContent = "读取失败";
  }
}

function startPolling() {
  if (!state.polling) {
    state.polling = setInterval(pollJobs, 1500);
    pollJobs();
  }
}

function stopPolling() {
  if (state.polling) {
    clearInterval(state.polling);
    state.polling = null;
    el.queueSummary.textContent = "空闲";
  }
}

function jobStatusText(status) {
  return {
    uploading: "上传中",
    queued: "排队",
    indexing: "索引中",
    completed: "完成",
    failed: "失败",
  }[status] || status;
}

function renderJobs() {
  el.jobList.replaceChildren();
  clearListLoading(el.jobList);
  if (!state.jobs.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无任务";
    el.jobList.append(empty);
    el.queueSummary.textContent = "空闲";
    return;
  }

  const active = state.jobs.filter(
    (job) =>
      job.status === "queued" || job.status === "indexing" || job.status === "uploading",
  );
  el.queueSummary.textContent = active.length ? `${active.length} 个进行中` : "空闲";

  for (const job of state.jobs.slice(0, 12)) {
    const item = document.createElement("article");
    item.className = `job-item ${job.status}`;

    const header = document.createElement("div");
    header.className = "job-header";
    const title = document.createElement("strong");
    title.textContent =
      job.filenames.length > 1
        ? `${job.filenames[0]} 等 ${job.filenames.length} 个文件`
        : job.filenames[0];
    const badge = document.createElement("span");
    badge.className = "job-badge";
    badge.textContent = jobStatusText(job.status);
    header.append(title, badge);

    const detail = document.createElement("div");
    detail.className = "job-detail";
    if (job.status === "uploading") {
      const progress = document.createElement("div");
      progress.className = "progress";
      const bar = document.createElement("span");
      bar.style.width = `${job.progress || 0}%`;
      progress.append(bar);
      detail.append(progress);
    } else if (job.status === "failed") {
      detail.textContent = job.error || "处理失败";
    } else if (job.status === "completed") {
      detail.textContent = `${job.files} 文件 · ${job.documents} 文档页 · ${job.chunks} 切片`;
    } else {
      detail.textContent = job.status === "indexing" ? "正在解析并生成向量" : "等待处理";
    }

    item.append(header, detail);
    el.jobList.append(item);
  }
}

function showJobError(message) {
  state.jobs.unshift({
    id: `upload-${Date.now()}`,
    filenames: [],
    status: "failed",
    error: message,
    files: 0,
    documents: 0,
    chunks: 0,
  });
  renderJobs();
}

// --- staging review -------------------------------------------------------

async function loadStaging() {
  try {
    const payload = await requestJson("/api/staging");
    stagingState.items = payload.items || [];
    stagingState.groups = payload.groups || [];
  } catch (_) {
    stagingState.items = [];
  }
  renderStaging();
  renderStats();
}

function renderStaging() {
  const pending = stagingState.items.filter((item) => item.status === "pending");
  el.stagingSection.hidden = !pending.length;
  el.stagingList.replaceChildren();
  el.stagingSummary.textContent = pending.length ? `${pending.length} 项待审核` : "空闲";
  for (const item of pending) el.stagingList.append(createStagingCard(item));
}

function createStagingCard(item) {
  const card = document.createElement("article");
  card.className = "job-item staging-item";

  const header = document.createElement("div");
  header.className = "job-header";
  const title = document.createElement("strong");
  title.textContent =
    item.filenames.length > 1
      ? `${item.filenames[0]} 等 ${item.filenames.length} 个文件`
      : item.filenames[0];
  const badge = document.createElement("span");
  badge.className = "job-badge";
  badge.textContent = "待审核";
  header.append(title, badge);

  const detail = document.createElement("div");
  detail.className = "job-detail";
  detail.textContent = [item.uploader, formatBytes(item.size_bytes), formatTime(item.created_at)]
    .filter(Boolean)
    .join(" · ");

  const actions = document.createElement("div");
  actions.className = "staging-actions";

  const groupSelect = document.createElement("select");
  groupSelect.className = "config-select";
  groupSelect.setAttribute("aria-label", "入库后的访问组");
  for (const group of stagingState.groups.length ? stagingState.groups : ["public"]) {
    const option = document.createElement("option");
    option.value = group;
    option.textContent = group === "public" ? "public（所有人可见）" : group;
    groupSelect.append(option);
  }

  const reject = document.createElement("button");
  reject.type = "button";
  reject.className = "ghost-button";
  reject.textContent = "拒绝";
  reject.addEventListener("click", () => decideStaging(item.id, "reject", "", reject));

  const approve = document.createElement("button");
  approve.type = "button";
  approve.className = "primary-button";
  approve.textContent = "通过";
  approve.addEventListener("click", () =>
    decideStaging(item.id, "approve", groupSelect.value, approve),
  );

  actions.append(groupSelect, reject, approve);
  card.append(header, detail, actions);
  return card;
}

async function decideStaging(itemId, action, group, button) {
  button.disabled = true;
  try {
    await requestJson(`/api/staging/${encodeURIComponent(itemId)}/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(action === "approve" ? { group } : {}),
    });
    showToast(action === "approve" ? "已通过，开始入库" : "已拒绝");
    await loadStaging();
    if (action === "approve") startPolling();
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

// --- indexed documents ----------------------------------------------------

const CHUNK_STRATEGY_LABELS = { heading: "标题对齐", fixed: "固定长度" };

// 切片方式来自入库时写进元数据的 structured_by：有它才会走标题对齐切分
// （先按 Markdown 标题分节，节内再按字数切），没有就是固定长度切分。
function chunkStrategyLabel(indexedDocument) {
  const kind = indexedDocument.chunk_strategy || "fixed";
  const provider = indexedDocument.chunk_strategy_provider || "";
  const label = CHUNK_STRATEGY_LABELS[kind] || kind;
  return provider ? `${label}·${provider}` : label;
}

function chunkStrategyDetail(indexedDocument) {
  const provider = indexedDocument.chunk_strategy_provider || "";
  return provider
    ? `先按 Markdown 标题层级分节，节内再按字数切（${provider} 结构化提取）`
    : "固定长度切分（未做结构化提取）";
}

// 「09-14 15:39」：年月日放在 title 里，列窄，放不下完整时间。
function formatIngestClock(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (number) => String(number).padStart(2, "0");
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function ingestLabel(indexedDocument) {
  const actor = indexedDocument.last_ingested_by || "";
  const clock = formatIngestClock(indexedDocument.last_ingested_at);
  if (!actor && !clock) return "—";
  return [actor, clock].filter(Boolean).join(" · ");
}

// 只代表**最近一次**入库（同名文档重新入库会覆盖），不是这份文档最初被谁放进来的。
function ingestDetail(indexedDocument) {
  if (!indexedDocument.last_ingested_at) {
    return "没有入库记录：该文档早于审计日志建立（2026-09-12）";
  }
  return `最近一次入库：${indexedDocument.last_ingested_by}，${formatTime(indexedDocument.last_ingested_at)}`;
}

function renderDocuments() {
  el.documentList.replaceChildren();
  clearListLoading(el.documentList);
  if (!state.documents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无入库文档";
    el.documentList.append(empty);
    return;
  }

  const table = document.createElement("div");
  table.className = "doc-table";

  const canManageAccess = Boolean(state.access.manageable);
  // 入库人与时间来自审计日志，服务端只回给管理员（审计页本身也是仅管理员可见）。
  const isAdmin = Boolean(state.identity.isAdmin);
  const headers = ["标题", "来源", "切片", "切片方式"];
  if (isAdmin) headers.push("最近入库");
  if (canManageAccess) headers.push("访问组");
  headers.push("");
  // 列模板按这两个 class 切换（见 styles.css 的 --doc-cols）。
  table.classList.toggle("with-ingest", isAdmin);
  table.classList.toggle("with-access", canManageAccess);

  const head = document.createElement("div");
  head.className = "doc-row doc-row-head";
  for (const label of headers) {
    const cell = document.createElement("span");
    cell.textContent = label;
    head.append(cell);
  }
  table.append(head);

  for (const indexedDocument of state.documents) {
    const row = document.createElement("div");
    row.className = "doc-row";

    const title = document.createElement("button");
    title.type = "button";
    title.className = "document-open";
    title.title = "查看文档内容";
    title.textContent = indexedDocument.title || "未命名文档";
    title.addEventListener("click", () => openDocument(indexedDocument));

    const source = document.createElement("span");
    source.className = "document-source";
    source.textContent = indexedDocument.source || "未知来源";

    const chunks = document.createElement("span");
    chunks.className = "chunk-count";
    chunks.textContent = String(indexedDocument.chunk_count);

    const strategy = document.createElement("span");
    strategy.className = "chunk-strategy";
    strategy.textContent = chunkStrategyLabel(indexedDocument);
    strategy.title = chunkStrategyDetail(indexedDocument);

    const ingest = document.createElement("span");
    ingest.className = "document-ingest";
    ingest.textContent = ingestLabel(indexedDocument);
    ingest.title = ingestDetail(indexedDocument);

    const actions = document.createElement("span");
    actions.className = "doc-row-actions";

    const view = document.createElement("button");
    view.type = "button";
    view.className = "text-button";
    view.textContent = "查看";
    view.addEventListener("click", () => openDocument(indexedDocument));
    actions.append(view);

    if (isAdmin) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "row-action";
      remove.title = "从向量库删除";
      remove.setAttribute("aria-label", `删除 ${title.textContent}`);
      remove.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
      remove.addEventListener("click", () => deleteDocument(indexedDocument));
      actions.append(remove);
    }

    row.append(title, source, chunks, strategy);
    if (isAdmin) {
      row.append(ingest);
    }
    if (canManageAccess) {
      row.append(createAccessSelect(indexedDocument));
    }
    row.append(actions);
    table.append(row);
  }
  el.documentList.append(table);
}

// 每个文档一行下拉，改完立即生效（只改元数据，不重新向量化）。
function createAccessSelect(indexedDocument) {
  const select = document.createElement("select");
  select.className = "config-select doc-access-select";
  select.setAttribute("aria-label", `${indexedDocument.title} 的访问组`);

  const groups = state.access.groups || [];
  const current = indexedDocument.access_group || "public";
  // 当前组若不在已知组里（例如策略被改过），也要显示出来，否则会误显示成别的组。
  const options = groups.includes(current) ? groups : [current, ...groups];
  for (const group of options) {
    const option = document.createElement("option");
    option.value = group;
    option.textContent = group === "public" ? "public（所有人）" : group;
    select.append(option);
  }
  select.value = current;

  select.addEventListener("change", async () => {
    const next = select.value;
    select.disabled = true;
    try {
      await requestJson(
        `/api/documents/${encodeURIComponent(indexedDocument.doc_id)}/access`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ group: next }),
        },
      );
      indexedDocument.access_group = next;
      showToast(`已将「${indexedDocument.title}」改为 ${next}`);
    } catch (error) {
      select.value = current;
      showToast(error.message, "error");
    } finally {
      select.disabled = false;
    }
  });
  return select;
}

async function deleteDocument(indexedDocument) {
  const title = indexedDocument.title || "未命名文档";
  const confirmed = confirm(
    `从向量库删除「${title}」的全部 ${indexedDocument.chunk_count} 个切片？\n\n` +
      `源文件不受影响；若之后重新索引 data/raw，该文档会再次入库。`,
  );
  if (!confirmed) return;

  try {
    const result = await requestJson(
      `/api/documents/${encodeURIComponent(indexedDocument.doc_id)}`,
      { method: "DELETE" },
    );
    showToast(`已删除 ${result.deleted_chunks} 个切片`);
    emit("documents:refresh");
  } catch (error) {
    showToast(error.message, "error");
  }
}

// --- document viewer ------------------------------------------------------

const docState = {
  docId: "",
  chunks: [],
  nextOffset: 0,
  total: 0,
  limit: 50,
  loading: false,
  done: false,
  focusChunkId: "",
  focusApplied: false,
};

function currentDocMode() {
  const checked = document.querySelector('input[name="doc-view-mode"]:checked');
  return checked ? checked.value : "cards";
}

function setDocViewMode(value) {
  const radio = document.querySelector(`input[name="doc-view-mode"][value="${value}"]`);
  if (radio) radio.checked = true;
}

function showList() {
  el.docView.hidden = true;
  el.listView.hidden = false;
  docState.docId = "";
  docState.chunks = [];
  docState.focusChunkId = "";
}

async function openDocument(indexedDocument, focusChunkId = "") {
  docState.docId = indexedDocument.doc_id;
  docState.chunks = [];
  docState.nextOffset = 0;
  docState.total = Number(indexedDocument.chunk_count) || 0;
  docState.loading = false;
  docState.done = false;
  docState.focusChunkId = focusChunkId;
  docState.focusApplied = false;

  // Only the card list can highlight a single chunk.
  if (focusChunkId) setDocViewMode("cards");

  el.docTitle.textContent = indexedDocument.title || "未命名文档";
  el.docMeta.textContent = [indexedDocument.source || "", `${docState.total} 个切片`]
    .filter(Boolean)
    .join(" · ");
  el.docBody.replaceChildren();
  el.docProgress.textContent = `已显示 0 / ${docState.total}`;
  el.docLoadMore.hidden = true;
  el.listView.hidden = true;
  el.docView.hidden = false;

  await loadDocumentPage();
}

async function loadDocumentPage() {
  if (docState.loading || docState.done || !docState.docId) return;
  docState.loading = true;
  el.docLoadMore.disabled = true;

  const isFirstPage = docState.nextOffset === 0;
  const focus =
    isFirstPage && docState.focusChunkId
      ? `&focus=${encodeURIComponent(docState.focusChunkId)}`
      : "";
  try {
    const payload = await requestJson(
      `/api/documents/${encodeURIComponent(docState.docId)}` +
        `?offset=${docState.nextOffset}&limit=${docState.limit}${focus}`,
    );
    const incoming = payload.chunks || [];
    const start = Number.isFinite(Number(payload.offset))
      ? Number(payload.offset)
      : docState.nextOffset;
    docState.total = Number(payload.chunk_count) || docState.total;
    docState.nextOffset = start + incoming.length;
    docState.chunks.push(...incoming);

    if (
      isFirstPage &&
      docState.focusChunkId &&
      !incoming.some((chunk) => chunk.chunk_id === docState.focusChunkId)
    ) {
      showToast("未找到该切片，可能已重新入库", "error");
    }
    if (!incoming.length || docState.nextOffset >= docState.total) {
      docState.done = true;
    }
    renderDocumentView();
  } catch (error) {
    renderListError(el.docBody, error.message);
    docState.done = true;
  }
  docState.loading = false;
  el.docLoadMore.disabled = false;
}

function renderDocumentView() {
  el.docBody.replaceChildren();
  if (currentDocMode() === "merged") {
    renderMergedChunks();
  } else {
    renderChunkCards();
  }
  el.docProgress.textContent = `已显示 ${docState.chunks.length} / ${docState.total}`;
  el.docLoadMore.hidden = docState.done || docState.nextOffset >= docState.total;
  highlightFocusedChunk();
}

function highlightFocusedChunk() {
  if (!docState.focusChunkId) return;
  const target = el.docBody.querySelector(
    `[data-chunk-id="${CSS.escape(docState.focusChunkId)}"]`,
  );
  if (!target) return;
  target.classList.add("doc-chunk-focus");
  if (!docState.focusApplied) {
    docState.focusApplied = true;
    target.scrollIntoView({ block: "center" });
  }
}

function chunkMetaText(chunk) {
  const parts = [];
  if (chunk.page_label) {
    parts.push(`第 ${chunk.page_label} 页`);
  } else if (chunk.page !== null && chunk.page !== undefined) {
    parts.push(`第 ${Number(chunk.page) + 1} 页`);
  }
  if (chunk.section_path) parts.push(chunk.section_path);
  return parts.join(" · ");
}

function renderChunkCards() {
  for (const chunk of docState.chunks) {
    const card = document.createElement("article");
    card.className = "doc-chunk";
    card.dataset.chunkId = chunk.chunk_id;

    const head = document.createElement("div");
    head.className = "doc-chunk-head";
    const identifier = document.createElement("code");
    identifier.textContent = chunk.chunk_id;
    const meta = document.createElement("span");
    meta.textContent = chunkMetaText(chunk);
    head.append(identifier, meta);

    const text = document.createElement("div");
    text.className = "doc-chunk-text";
    text.textContent = chunk.text || "";

    card.append(head, text);
    el.docBody.append(card);
  }
}

function buildMergedSections() {
  const sections = [];
  let current = null;
  for (const chunk of docState.chunks) {
    const key =
      chunk.page === null || chunk.page === undefined ? "document" : `page-${chunk.page}`;
    if (!current || current.key !== key) {
      current = { key, page: chunk.page, label: chunk.page_label, text: "", end: null };
      sections.push(current);
    }

    let text = chunk.text || "";
    const start = Number(chunk.start_char || 0);
    if (current.end !== null && start < current.end) {
      const overlap = current.end - start;
      text = overlap < text.length ? text.slice(overlap) : "";
    } else if (current.text && text && !current.text.endsWith("\n\n")) {
      text = `\n\n${text}`;
    }
    current.text += text;
    current.end = Number(chunk.end_char || start + (chunk.text || "").length);
  }
  return sections;
}

function renderMergedChunks() {
  for (const section of buildMergedSections()) {
    if (section.page !== null && section.page !== undefined) {
      const marker = document.createElement("div");
      marker.className = "doc-page-marker";
      marker.textContent = `第 ${section.label || Number(section.page) + 1} 页`;
      el.docBody.append(marker);
    }
    const block = document.createElement("div");
    block.className = "doc-merged-text";
    block.textContent = section.text;
    el.docBody.append(block);
  }
}

// Citations on the ask page route here with a doc id and a chunk *suffix*.
// The indexed chunk_id is the whole "doc_id#chunk_003", so compose it back.
async function openFromParams(params) {
  const { docId } = params;
  const suffix = params.chunkId || "";
  const chunkId = suffix && !suffix.includes("#") ? `${docId}#${suffix}` : suffix;

  let indexedDocument = state.documents.find((item) => item.doc_id === docId);
  if (!indexedDocument) {
    try {
      const payload = await requestJson(
        `/api/documents/${encodeURIComponent(docId)}?offset=0&limit=1`,
      );
      indexedDocument = {
        doc_id: docId,
        title: payload.title,
        source: payload.source,
        chunk_count: payload.chunk_count,
      };
    } catch (_) {
      showToast("引用的文档已不在知识库中", "error");
      showList();
      return;
    }
  }
  await openDocument(indexedDocument, chunkId);
}

// --- view contract --------------------------------------------------------

export default {
  name: "library",

  mount() {
    if (mounted) return;
    mounted = true;

    el.chooseFilesButton.addEventListener("click", openFilePicker);
    el.dropZone.addEventListener("click", openFilePicker);
    el.dropZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openFilePicker();
      }
    });
    el.fileInput.addEventListener("change", () => {
      uploadFiles(el.fileInput.files);
      el.fileInput.value = "";
    });

    for (const eventName of ["dragenter", "dragover"]) {
      el.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        el.dropZone.classList.add("dragging");
      });
    }
    for (const eventName of ["dragleave", "drop"]) {
      el.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        el.dropZone.classList.remove("dragging");
      });
    }
    el.dropZone.addEventListener("drop", (event) => {
      uploadFiles(event.dataTransfer ? event.dataTransfer.files : null);
    });

    el.reloadDocumentsButton.addEventListener("click", () => emit("documents:refresh"));
    el.docBackButton.addEventListener("click", showList);
    el.docLoadMore.addEventListener("click", loadDocumentPage);
    el.docViewMode.addEventListener("change", renderDocumentView);
    el.docBody.addEventListener("scroll", () => {
      if (docState.done || docState.loading) return;
      const { scrollTop, clientHeight, scrollHeight } = el.docBody;
      if (scrollTop + clientHeight >= scrollHeight - 240) loadDocumentPage();
    });

    on("documents", () => {
      renderStats();
      renderDocuments();
    });
    on("identity", () => {
      renderUploadPermission();
      renderDocuments();
    });
    on("staging:refresh", loadStaging);
    on("documents:refresh", async () => {
      setListLoading(el.documentList, "文档加载中");
      await refreshDocuments();
    });
    on("status", (status) => {
      if (!status) return;
      el.uploadLimit.textContent = `PDF / MD / TXT · 单次最多 ${status.max_upload_files} 个`;
      el.structurizeLabel.textContent = status.mineru_configured
        ? "结构化提取（MinerU 在线解析）"
        : "结构化提取（本地 markitdown，PDF 效果一般）";
    });
  },

  show(params) {
    renderStats();
    renderUploadPermission();
    renderDocuments();
    loadStaging();
    if (!state.polling) startPolling();

    if (params && params.docId) {
      openFromParams(params);
    } else if (docState.docId) {
      el.listView.hidden = true;
      el.docView.hidden = false;
    } else {
      el.listView.hidden = false;
      el.docView.hidden = true;
    }
  },

  hide() {
    stopPolling();
  },

  onDocuments() {
    renderStats();
    renderDocuments();
  },
};
