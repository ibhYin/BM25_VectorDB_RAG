const state = {
  vectorHits: [],
  answerJob: null,
  jobs: [],
  documents: [],
  polling: null,
  retrievalInitialized: false,
};

const elements = {
  serviceSummary: document.querySelector("#service-summary"),
  userSwitcher: document.querySelector("#user-switcher"),
  userButton: document.querySelector("#user-button"),
  userLabel: document.querySelector("#user-label"),
  userMenu: document.querySelector("#user-menu"),
  vectorStatus: document.querySelector("#vector-status"),
  modelStatus: document.querySelector("#model-status"),
  refreshButton: document.querySelector("#refresh-button"),
  configButton: document.querySelector("#config-button"),
  configDialog: document.querySelector("#config-dialog"),
  configForm: document.querySelector("#config-form"),
  configClose: document.querySelector("#config-close"),
  configCancel: document.querySelector("#config-cancel"),
  configSave: document.querySelector("#config-save"),
  configStatus: document.querySelector("#config-status"),
  configLlmModel: document.querySelector("#config-llm-model"),
  configEmbeddingModel: document.querySelector("#config-embedding-model"),
  configLlmModelCustom: document.querySelector("#config-llm-model-custom"),
  configEmbeddingModelCustom: document.querySelector("#config-embedding-model-custom"),
  configEmbeddingWarning: document.querySelector("#config-embedding-warning"),
  configIndexModel: document.querySelector("#config-index-model"),
  indexWarning: document.querySelector("#index-warning"),
  configLlmScope: document.querySelector("#config-llm-scope"),
  configEmbeddingScope: document.querySelector("#config-embedding-scope"),
  configLlmModelsHint: document.querySelector("#config-llm-models-hint"),
  configEmbeddingModelsHint: document.querySelector("#config-embedding-models-hint"),
  configRefreshModels: document.querySelector("#config-refresh-models"),
  configOpenaiBaseUrl: document.querySelector("#config-openai-base-url"),
  configOpenaiApiKey: document.querySelector("#config-openai-api-key"),
  configEmbeddingBaseUrl: document.querySelector("#config-embedding-base-url"),
  configEmbeddingApiKey: document.querySelector("#config-embedding-api-key"),
  configMineruToken: document.querySelector("#config-mineru-token"),
  configMineruClear: document.querySelector("#config-mineru-clear"),
  configMineruModelVersion: document.querySelector("#config-mineru-model-version"),
  configStructurizeProvider: document.querySelector("#config-structurize-provider"),
  configMineruScope: document.querySelector("#config-mineru-scope"),
  structurizeLabel: document.querySelector("#structurize-label"),
  auditButton: document.querySelector("#audit-button"),
  auditDialog: document.querySelector("#audit-dialog"),
  auditClose: document.querySelector("#audit-close"),
  auditRefresh: document.querySelector("#audit-refresh"),
  auditList: document.querySelector("#audit-list"),
  auditNote: document.querySelector("#audit-note"),
  auditStatus: document.querySelector("#audit-status"),
  stagingSection: document.querySelector("#staging-section"),
  stagingList: document.querySelector("#staging-list"),
  stagingSummary: document.querySelector("#staging-summary"),
  documentCount: document.querySelector("#document-count"),
  chunkCount: document.querySelector("#chunk-count"),
  uploadLimit: document.querySelector("#upload-limit"),
  uploadPermission: document.querySelector("#upload-permission"),
  dropZone: document.querySelector("#drop-zone"),
  fileInput: document.querySelector("#file-input"),
  structurizeCheckbox: document.querySelector("#structurize-checkbox"),
  chooseFilesButton: document.querySelector("#choose-files-button"),
  queueSummary: document.querySelector("#queue-summary"),
  jobList: document.querySelector("#job-list"),
  reloadDocumentsButton: document.querySelector("#reload-documents-button"),
  documentList: document.querySelector("#document-list"),
  runStatus: document.querySelector("#run-status"),
  clearChatButton: document.querySelector("#clear-chat-button"),
  chatStream: document.querySelector("#chat-stream"),
  chatView: document.querySelector("#chat-view"),
  documentView: document.querySelector("#document-view"),
  docBackButton: document.querySelector("#doc-back-button"),
  docTitle: document.querySelector("#doc-title"),
  docMeta: document.querySelector("#doc-meta"),
  docBody: document.querySelector("#doc-body"),
  docProgress: document.querySelector("#doc-progress"),
  docLoadMore: document.querySelector("#doc-load-more"),
  docViewMode: document.querySelector("#doc-view-mode"),
  workspace: document.querySelector(".workspace"),
  railLibrary: document.querySelector(".rail-library"),
  railContext: document.querySelector(".rail-context"),
  resizerLeft: document.querySelector("#resizer-left"),
  resizerRight: document.querySelector("#resizer-right"),
  askForm: document.querySelector("#ask-form"),
  askButton: document.querySelector("#ask-button"),
  question: document.querySelector("#question"),
  retrievalMethod: document.querySelector("#retrieval-method"),
  retrievalTopK: document.querySelector("#retrieval-top-k"),
  candidatePool: document.querySelector("#candidate-pool"),
  similarityThreshold: document.querySelector("#similarity-threshold"),
  similarityThresholdValue: document.querySelector("#similarity-threshold-value"),
  contextList: document.querySelector("#context-list"),
  processList: document.querySelector("#process-list"),
  toast: document.querySelector("#toast"),
};

let toastTimer = null;

function showToast(message, variant = "ok") {
  elements.toast.textContent = message;
  elements.toast.className = `toast ${variant}`;
  elements.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    elements.toast.hidden = true;
  }, 2800);
}

function getRetrievalMethod() {
  const checked = document.querySelector('input[name="retrieval-method"]:checked');
  return checked ? checked.value : "vector";
}

function setRetrievalMethod(value) {
  const radio = document.querySelector(`input[name="retrieval-method"][value="${value}"]`);
  if (radio) radio.checked = true;
}

function canUpload() {
  return true;
}

// Without direct write access an upload is parked for review instead of indexed.
function uploadsToStaging() {
  return Boolean(identityState.user) && !identityState.isAdmin;
}

function renderUploadPermission() {
  const staged = uploadsToStaging();
  elements.dropZone.setAttribute("aria-disabled", "false");
  elements.dropZone.classList.remove("disabled");
  elements.chooseFilesButton.disabled = false;
  elements.uploadPermission.hidden = !staged;
  elements.uploadPermission.classList.toggle("info", staged);
  elements.uploadPermission.textContent = staged
    ? "当前身份无直接入库权限，提交后需管理员审核"
    : "";
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("Content-Type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : { error: await response.text() };

  if (!response.ok) {
    throw new Error(payload.error || "请求失败");
  }
  return payload;
}

async function refreshStatus() {
  elements.refreshButton.disabled = true;
  setStatusLoading(true);
  try {
    const status = await requestJson("/api/status");
    const chunkCount = state.documents.reduce(
      (total, document) => total + Number(document.chunk_count || 0),
      0,
    );
    elements.serviceSummary.textContent = `${status.llm_model} · ${status.embedding_model}`;
    elements.vectorStatus.textContent = status.chroma_exists ? "向量库可用" : "向量库未创建";
    elements.vectorStatus.classList.toggle("ok", status.chroma_exists);
    elements.modelStatus.textContent = status.has_openai_api_key ? "LLM 已配置" : "LLM 未配置";
    elements.modelStatus.classList.toggle("ok", status.has_openai_api_key);
    elements.uploadLimit.textContent = `PDF / MD / TXT · 单次最多 ${status.max_upload_files} 个`;
    if (!state.retrievalInitialized) {
      setRetrievalMethod(status.retrieval_method || "vector");
      elements.retrievalTopK.value = status.retrieval_top_k ?? 20;
      elements.candidatePool.value = status.candidate_pool_size ?? 100;
      elements.similarityThreshold.value = status.similarity_threshold ?? 0.35;
      state.retrievalInitialized = true;
      updateRetrievalControls();
    }
    elements.documentCount.textContent = String(state.documents.length);
    elements.chunkCount.textContent = String(chunkCount);
    renderIndexWarning(status);
    renderStructurizeLabel(status.structurize_provider || "markitdown");
  } catch (error) {
    elements.serviceSummary.textContent = "服务连接失败";
    elements.vectorStatus.textContent = "服务不可用";
    elements.modelStatus.textContent = "服务不可用";
    elements.documentCount.textContent = "--";
    elements.chunkCount.textContent = "--";
    elements.indexWarning.hidden = true;
  } finally {
    setStatusLoading(false);
    elements.refreshButton.disabled = false;
  }
}

// A changed embedding model means stored vectors no longer match the config, so
// the warning stays visible at the top of the page until the corpus is reindexed.
function renderIndexWarning(status) {
  const element = elements.indexWarning;
  if (!status.embedding_model_mismatch || !status.index_embedding_model) {
    element.hidden = true;
    return;
  }
  element.textContent =
    `向量模型已变更：索引由 ${status.index_embedding_model} 建立，` +
    `当前配置为 ${status.embedding_model}。检索结果可能失效，请重新入库文档。`;
  element.hidden = false;
}

async function refreshDocuments() {
  setListLoading(elements.documentList, "文档加载中");
  try {
    const payload = await requestJson("/api/documents");
    state.documents = payload.documents || [];
    renderDocuments();
    await refreshStatus();
  } catch (error) {
    renderLibraryError(error.message);
  }
}

function setStatusLoading(loading) {
  for (const element of [elements.vectorStatus, elements.modelStatus]) {
    element.classList.toggle("loading", loading);
    if (loading) {
      element.replaceChildren(createSpinner(), document.createTextNode("加载中"));
    }
  }
  if (loading) {
    elements.serviceSummary.textContent = "正在读取服务状态...";
    elements.documentCount.replaceChildren(createSpinner());
    elements.chunkCount.replaceChildren(createSpinner());
  }
}

function createSpinner() {
  const spinner = document.createElement("span");
  spinner.className = "loading-spinner";
  spinner.setAttribute("aria-hidden", "true");
  return spinner;
}

function setListLoading(element, message) {
  element.replaceChildren();
  element.setAttribute("aria-busy", "true");
  const loading = document.createElement("div");
  loading.className = "loading-state";
  loading.append(createSpinner(), document.createTextNode(message));
  element.append(loading);
}

function clearListLoading(element) {
  element.removeAttribute("aria-busy");
}

function renderDocuments() {
  elements.documentList.replaceChildren();
  clearListLoading(elements.documentList);
  if (!state.documents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无入库文档";
    elements.documentList.append(empty);
  }

  for (const indexedDocument of state.documents) {
    const item = document.createElement("article");
    item.className = "document-item";

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
    chunks.textContent = `${indexedDocument.chunk_count} 片`;

    if (identityState.isAdmin) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "row-action";
      remove.title = "从向量库删除";
      remove.setAttribute("aria-label", `删除 ${title.textContent}`);
      remove.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
      remove.addEventListener("click", () => deleteDocument(indexedDocument));
      item.append(title, source, chunks, remove);
    } else {
      item.append(title, source, chunks);
    }
    elements.documentList.append(item);
  }
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
    await refreshDocuments();
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderLibraryError(message) {
  elements.documentList.replaceChildren();
  clearListLoading(elements.documentList);
  const error = document.createElement("div");
  error.className = "empty-list error";
  error.textContent = message;
  elements.documentList.append(error);
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
  if (elements.structurizeCheckbox.checked) {
    form.append("structurize", "on");
  }

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
      if (index >= 0) {
        state.jobs.splice(index, 1, payload);
      }
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
      await refreshDocuments();
    }

    const active = state.jobs.some(
      (job) => job.status === "queued" || job.status === "indexing",
    );
    if (!active) {
      stopPolling();
    }
  } catch (_) {
    elements.jobList.replaceChildren();
    clearListLoading(elements.jobList);
    const error = document.createElement("div");
    error.className = "empty-list error";
    error.textContent = "任务状态不可用";
    elements.jobList.append(error);
    elements.queueSummary.textContent = "读取失败";
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
    elements.queueSummary.textContent = "空闲";
  }
}

function renderJobs() {
  elements.jobList.replaceChildren();
  clearListLoading(elements.jobList);
  if (!state.jobs.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无任务";
    elements.jobList.append(empty);
    elements.queueSummary.textContent = "空闲";
    return;
  }

  const active = state.jobs.filter(
    (job) => job.status === "queued" || job.status === "indexing" || job.status === "uploading",
  );
  elements.queueSummary.textContent = active.length ? `${active.length} 个进行中` : "空闲";

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

    header.append(title, badge);
    item.append(header, detail);
    elements.jobList.append(item);
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

async function ask(event) {
  event.preventDefault();
  if (elements.askButton.disabled || !elements.askForm.reportValidity()) return;
  const question = elements.question.value.trim();
  if (!question) {
    elements.question.focus();
    return;
  }

  addChatMessage("question", question);
  elements.question.value = "";
  elements.askButton.disabled = true;
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
        top_k: Number(elements.retrievalTopK.value),
        candidate_pool: elements.candidatePool.disabled
          ? undefined
          : Number(elements.candidatePool.value),
        similarity_threshold: elements.similarityThreshold.disabled
          ? undefined
          : Number(elements.similarityThreshold.value),
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
    elements.askButton.disabled = false;
    elements.question.focus();
  }
}

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

const processDefinitions = [
  ["encode", "问题编码"],
  ["vector_search", "向量搜索"],
  ["answer", "生成答案"],
];

function renderProcess() {
  elements.processList.replaceChildren();
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
    const labels = {
      bm25: "BM25 关键词搜索",
      hybrid: "混合检索",
    };
    title.textContent = key === "vector_search" && method !== "vector"
      ? labels[method] : label;
    const detail = document.createElement("span");
    detail.textContent = stageDetail(key, stage);
    body.append(title, detail);
    item.append(indicator, body);
    elements.processList.append(item);
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
  elements.runStatus.replaceChildren();
  if (loading) {
    elements.runStatus.append(createSpinner(), document.createTextNode(` ${message}`));
  } else {
    elements.runStatus.textContent = message;
  }
}

function renderContextLoading() {
  elements.contextList.replaceChildren();
  const loading = document.createElement("div");
  loading.className = "loading-state";
  loading.append(createSpinner(), document.createTextNode("检索中"));
  elements.contextList.append(loading);
}

// A citation may or may not carry a chunk id; bold and inline code ride along in
// the same pass so the whole answer can be rendered without touching innerHTML.
const INLINE_PATTERN =
  /\[source:\s*([^\]#\s]+)(?:#(chunk_\d+))?\]|\*\*([^*]+)\*\*|`([^`]+)`/g;
const DOC_LABEL_LIMIT = 12;

function shortDocLabel(docId) {
  const indexedDocument = state.documents.find((item) => item.doc_id === docId);
  const name = (indexedDocument && indexedDocument.title) || docId;
  return name.length > DOC_LABEL_LIMIT ? `${name.slice(0, DOC_LABEL_LIMIT)}…` : name;
}

function createCitationChip(docId, chunkId) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "citation";
  const label = shortDocLabel(docId);
  chip.textContent = chunkId ? `${label} ¶${chunkId.replace(/^chunk_/, "")}` : label;
  chip.title = chunkId
    ? `在原文中查看 ${docId} · ${chunkId}`
    : `打开文档 ${docId}`;
  chip.addEventListener("click", () => jumpToChunk(docId, chunkId));
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
  const empty = elements.chatStream.querySelector(".empty-chat");
  if (empty) {
    empty.remove();
  }

  const message = document.createElement("article");
  message.className = `chat-message ${type}`;
  if (type === "answer") {
    message.append(renderAnswer(text));
  } else {
    message.textContent = text;
  }
  elements.chatStream.append(message);
  elements.chatStream.scrollTop = elements.chatStream.scrollHeight;
}

function renderContext() {
  const items = state.vectorHits;
  elements.contextList.replaceChildren();

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无上下文";
    elements.contextList.append(empty);
    return;
  }

  for (const item of items) {
    elements.contextList.append(createVectorHitCard(item));
  }
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
  const page = hit.page_label || (hit.page !== null && hit.page !== undefined ? Number(hit.page) + 1 : "");
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

function clearChat() {
  elements.chatStream.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "empty-chat";
  const title = document.createElement("strong");
  title.textContent = "等待提问";
  const detail = document.createElement("span");
  detail.textContent = "答案将引用知识库中的 chunk_id。";
  empty.append(title, detail);
  elements.chatStream.append(empty);
  state.vectorHits = [];
  state.answerJob = null;
  renderProcess();
  renderContext();
  elements.runStatus.textContent = "就绪";
}

elements.refreshButton.addEventListener("click", async () => {
  await Promise.all([refreshStatus(), refreshDocuments(), loadStaging()]);
});
elements.reloadDocumentsButton.addEventListener("click", refreshDocuments);
function openFilePicker() {
  elements.fileInput.click();
}

elements.chooseFilesButton.addEventListener("click", openFilePicker);
elements.dropZone.addEventListener("click", openFilePicker);
elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openFilePicker();
  }
});
elements.fileInput.addEventListener("change", () => {
  uploadFiles(elements.fileInput.files);
  elements.fileInput.value = "";
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (canUpload()) {
      elements.dropZone.classList.add("dragging");
    }
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
}
elements.dropZone.addEventListener("drop", (event) => {
  uploadFiles(event.dataTransfer ? event.dataTransfer.files : null);
});

elements.askForm.addEventListener("submit", ask);
elements.clearChatButton.addEventListener("click", clearChat);
elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    elements.askForm.requestSubmit();
  }
});

function updateRetrievalControls() {
  const method = getRetrievalMethod();
  const thresholdOff = method !== "vector";
  elements.similarityThreshold.disabled = thresholdOff;
  elements.similarityThresholdValue.textContent = thresholdOff
    ? "不适用" : Number(elements.similarityThreshold.value).toFixed(2);
  elements.candidatePool.disabled = method !== "hybrid";
}

for (const element of [elements.retrievalTopK, elements.similarityThreshold]) {
  element.addEventListener("input", () => {
    state.retrievalInitialized = true;
    updateRetrievalControls();
    if (!state.answerJob) renderProcess();
  });
}

elements.retrievalMethod.addEventListener("change", () => {
  state.retrievalInitialized = true;
  updateRetrievalControls();
  if (!state.answerJob) renderProcess();
});

const identityState = { user: null, users: [], isAdmin: false };

async function loadIdentity() {
  try {
    const payload = await requestJson("/api/me");
    identityState.user = payload.user;
    identityState.users = payload.users || [];
    identityState.isAdmin = Boolean(payload.is_admin);
  } catch (_) {
    identityState.user = null;
    identityState.users = [];
    identityState.isAdmin = false;
  }
  renderUserSwitcher();
  renderConfigButton();
  renderUploadPermission();
  if (state.documents.length) {
    renderDocuments();
  }
  renderStaging();
}

function renderUserSwitcher() {
  const { user, users } = identityState;
  elements.userSwitcher.hidden = !user;
  if (!user) return;
  elements.userLabel.textContent = user;
  elements.userMenu.replaceChildren();
  const title = document.createElement("div");
  title.className = "menu-title";
  title.textContent = "切换身份（开发模式）";
  elements.userMenu.append(title);
  for (const item of users) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `menu-item${item.id === user ? " current" : ""}`;
    button.setAttribute("role", "menuitem");
    const name = document.createElement("span");
    name.textContent = item.id === user ? `✓ ${item.id}` : item.id;
    const groups = document.createElement("span");
    groups.className = "groups";
    groups.textContent = (item.groups || []).join(", ") || "无组";
    button.append(name, groups);
    button.addEventListener("click", () => {
      elements.userMenu.hidden = true;
      elements.userButton.setAttribute("aria-expanded", "false");
      if (item.id !== user) {
        location.href = `/login?user=${encodeURIComponent(item.id)}`;
      }
    });
    elements.userMenu.append(button);
  }
}

elements.userButton.addEventListener("click", () => {
  const open = elements.userMenu.hidden;
  elements.userMenu.hidden = !open;
  elements.userButton.setAttribute("aria-expanded", String(open));
});

document.addEventListener("click", (event) => {
  if (!elements.userSwitcher.hidden && !elements.userSwitcher.contains(event.target)) {
    elements.userMenu.hidden = true;
    elements.userButton.setAttribute("aria-expanded", "false");
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.userMenu.hidden) {
    elements.userMenu.hidden = true;
    elements.userButton.setAttribute("aria-expanded", "false");
  }
});

const configState = {
  loaded: null,
  models: { llm: "", embedding: "" },
  clearMineruToken: false,
};

function canConfigure() {
  return !identityState.user || identityState.isAdmin;
}

function renderConfigButton() {
  const allowed = canConfigure();
  elements.configButton.hidden = !allowed;
  elements.auditButton.hidden = !allowed;
}

function setConfigStatus(message, variant = "") {
  elements.configStatus.textContent = message;
  elements.configStatus.className = `config-status${variant ? ` ${variant}` : ""}`;
}

async function openConfig() {
  if (!canConfigure()) return;
  elements.configForm.reset();
  configState.models = { llm: "", embedding: "" };
  setConfigStatus("正在读取当前配置...");
  elements.configSave.disabled = true;
  elements.configDialog.showModal();
  try {
    const config = await requestJson("/api/config");
    configState.loaded = config;
    elements.configOpenaiBaseUrl.value = config.openai_base_url || "";
    elements.configEmbeddingBaseUrl.value = config.embedding_base_url || "";
    configState.models = {
      llm: config.llm_model || "",
      embedding: config.embedding_model || "",
    };
    elements.configOpenaiApiKey.placeholder = config.openai_api_key_set
      ? `已配置 ${config.openai_api_key_hint} · 留空保持不变`
      : "留空则保持不变";
    elements.configEmbeddingApiKey.placeholder = config.embedding_api_key_set
      ? `已配置 ${config.embedding_api_key_hint} · 留空保持不变`
      : "留空则保持不变";
    elements.configMineruModelVersion.value = config.mineru_model_version || "vlm";
    elements.configStructurizeProvider.value = config.structurize_provider || "markitdown";
    elements.configMineruToken.placeholder = config.mineru_api_token_set
      ? `已配置 ${config.mineru_api_token_hint} · 留空保持不变`
      : "留空则保持不变";
    configState.clearMineruToken = false;
    renderMineruScope(Boolean(config.mineru_api_token_set));
    setConfigStatus("");
  } catch (error) {
    setConfigStatus(error.message, "error");
  } finally {
    elements.configSave.disabled = false;
  }
  renderModelSelect("llm", []);
  renderModelSelect("embedding", []);
  renderScope("llm");
  renderScope("embedding");
  renderEmbeddingStatuses();
  await Promise.all([loadModels("llm"), loadModels("embedding")]);
}

// Mirrors app/model_catalog.is_local_endpoint: loopback is "this machine's own
// model server" (Ollama), anything else is treated as a remote gateway.
function isLocalEndpoint(baseUrl) {
  let host = "";
  try {
    host = new URL(baseUrl).hostname.toLowerCase();
  } catch (_) {
    return false;
  }
  if (!host) return false;
  return (
    ["localhost", "127.0.0.1", "::1", "0.0.0.0"].includes(host) ||
    host.endsWith(".local") ||
    /^127\./.test(host)
  );
}

function configTarget(target) {
  const llm = target === "llm";
  return {
    baseUrl: llm ? elements.configOpenaiBaseUrl : elements.configEmbeddingBaseUrl,
    apiKey: llm ? elements.configOpenaiApiKey : elements.configEmbeddingApiKey,
    select: llm ? elements.configLlmModel : elements.configEmbeddingModel,
    custom: llm ? elements.configLlmModelCustom : elements.configEmbeddingModelCustom,
    scope: llm ? elements.configLlmScope : elements.configEmbeddingScope,
    hint: llm ? elements.configLlmModelsHint : elements.configEmbeddingModelsHint,
  };
}

function renderScope(target) {
  const { baseUrl, scope } = configTarget(target);
  const value = baseUrl.value.trim();
  if (!value) {
    scope.textContent = "未配置地址";
    scope.className = "scope-badge empty";
    return;
  }
  const local = isLocalEndpoint(value);
  scope.textContent = local ? "本地" : "云端";
  scope.className = `scope-badge ${local ? "local" : "cloud"}`;
}

const CUSTOM_MODEL = "__custom__";

// A <select> keeps every option visible; a datalist would filter them down to
// whatever the input already contains, hiding the other candidates.
function renderModelSelect(target, models) {
  const { select } = configTarget(target);
  const desired = configState.models[target] || "";
  select.replaceChildren();

  const listed = new Set();
  for (const model of models) {
    if (!model.id || listed.has(model.id)) continue;
    listed.add(model.id);
    // Ollama reports on-disk sizes; the option label is what the user reads.
    select.append(
      new Option(model.size_label ? `${model.id} · ${model.size_label}` : model.id, model.id),
    );
  }
  if (desired && !listed.has(desired)) {
    select.append(new Option(`${desired}（当前）`, desired));
  }
  select.append(new Option("自定义…", CUSTOM_MODEL));

  if (listed.size > 0) {
    select.value = desired || CUSTOM_MODEL;
    syncCustomModel(target, "");
  } else {
    // No catalogue to pick from: hand over a free-text field with the current
    // model prefilled rather than a one-item list.
    select.value = CUSTOM_MODEL;
    syncCustomModel(target, desired);
  }
  if (select.value !== CUSTOM_MODEL) {
    select.dataset.previous = select.value;
  }
}

function syncCustomModel(target, prefill = "") {
  const { select, custom } = configTarget(target);
  const isCustom = select.value === CUSTOM_MODEL;
  custom.hidden = !isCustom;
  if (isCustom) {
    custom.value = prefill || custom.value || configState.models[target] || "";
  }
}

function modelValue(target) {
  const { select, custom } = configTarget(target);
  return (select.value === CUSTOM_MODEL ? custom.value : select.value).trim();
}

function renderEmbeddingStatuses() {
  const loaded = configState.loaded || {};
  const saved = loaded.embedding_model || "";
  elements.configEmbeddingWarning.classList.toggle(
    "warning",
    Boolean(saved) && modelValue("embedding") !== saved,
  );

  // The index records the model that actually produced the stored vectors.
  const indexModel = loaded.index_embedding_model || "";
  const element = elements.configIndexModel;
  element.className = "config-hint";
  if (!indexModel) {
    element.textContent = "尚未记录建库所用的向量模型。";
    return;
  }
  element.textContent = `当前索引由 ${indexModel} 建立。`;
  if (indexModel !== modelValue("embedding")) {
    element.classList.add("warning");
  }
}

function renderMineruScope(configured) {
  const provider = elements.configStructurizeProvider.value;
  const active = configured && provider === "mineru";
  elements.configMineruScope.textContent = active ? "已启用" : configured ? "已配置" : "未配置";
  elements.configMineruScope.className = `scope-badge ${active ? "cloud" : configured ? "local" : "empty"}`;
}

// The upload checkbox means different things depending on the chosen provider.
function renderStructurizeLabel(provider) {
  elements.structurizeLabel.textContent =
    provider === "mineru"
      ? "结构化提取（MinerU 在线解析）"
      : "结构化提取（本地 markitdown）";
}

async function loadModels(target) {
  if (!canConfigure()) return;
  const { baseUrl, apiKey, hint } = configTarget(target);
  const value = baseUrl.value.trim();
  hint.className = "config-model-hint";
  if (!value) {
    hint.textContent = "填写请求地址后可读取模型列表";
    renderModelSelect(target, []);
    renderEmbeddingStatuses();
    return;
  }

  hint.textContent = "正在读取模型列表...";
  try {
    const payload = await requestJson("/api/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target,
        base_url: value,
        api_key: apiKey.value.trim(),
      }),
    });
    const models = payload.models || [];
    renderModelSelect(target, models);
    if (payload.error) {
      hint.textContent = `${payload.error}；可在「自定义…」里手动输入模型名`;
      hint.classList.add("error");
    } else {
      const scopeText = payload.scope === "local" ? "本地" : "云端";
      hint.textContent = models.length
        ? `${scopeText}模型 ${models.length} 个`
        : "该端点未提供模型列表，可在「自定义…」里手动输入";
    }
    renderScope(target);
  } catch (error) {
    renderModelSelect(target, []);
    hint.textContent = `${error.message}；可在「自定义…」里手动输入模型名`;
    hint.classList.add("error");
  }
  renderEmbeddingStatuses();
}

// Only send fields the user actually changed; secrets are never echoed back,
// so a blank key field always means "keep the existing key".
function collectConfigChanges() {
  const loaded = configState.loaded || {};
  const changes = {};
  const baseUrl = elements.configOpenaiBaseUrl.value.trim();
  if (baseUrl !== (loaded.openai_base_url || "")) {
    changes.openai_base_url = baseUrl;
  }
  const embeddingBaseUrl = elements.configEmbeddingBaseUrl.value.trim();
  if (embeddingBaseUrl !== (loaded.embedding_base_url || "")) {
    changes.embedding_base_url = embeddingBaseUrl;
  }
  const llmModel = modelValue("llm");
  if (llmModel !== (loaded.llm_model || "")) {
    changes.llm_model = llmModel;
  }
  const embeddingModel = modelValue("embedding");
  if (embeddingModel !== (loaded.embedding_model || "")) {
    changes.embedding_model = embeddingModel;
  }
  const openaiKey = elements.configOpenaiApiKey.value.trim();
  if (openaiKey) changes.openai_api_key = openaiKey;
  const embeddingKey = elements.configEmbeddingApiKey.value.trim();
  if (embeddingKey) changes.embedding_api_key = embeddingKey;
  const mineruToken = elements.configMineruToken.value.trim();
  if (mineruToken) changes.mineru_api_token = mineruToken;
  if (configState.clearMineruToken) changes.clear_mineru_api_token = true;
  const mineruVersion = elements.configMineruModelVersion.value;
  if (mineruVersion !== (loaded.mineru_model_version || "vlm")) {
    changes.mineru_model_version = mineruVersion;
  }
  const structurizeProvider = elements.configStructurizeProvider.value;
  if (structurizeProvider !== (loaded.structurize_provider || "markitdown")) {
    changes.structurize_provider = structurizeProvider;
  }
  return changes;
}

async function submitConfig(event) {
  event.preventDefault();
  if (!canConfigure()) return;
  const changes = collectConfigChanges();
  if (!Object.keys(changes).length) {
    setConfigStatus("没有需要保存的更改");
    return;
  }

  // Changing the embedding model invalidates every stored vector, and
  // re-embedding the whole corpus is expensive: require an acknowledgement.
  const embeddingChanged = "embedding_model" in changes;
  if (embeddingChanged) {
    const previous = (configState.loaded || {}).embedding_model || "未设置";
    const confirmed = confirm(
      `向量模型将从「${previous}」改为「${changes.embedding_model}」。\n\n` +
        "不同模型的向量互不兼容，已入库的切片必须用新模型全部重新入库，" +
        "否则检索结果会失效。\n\n确定要更换吗？",
    );
    if (!confirmed) {
      setConfigStatus("已取消更换向量模型");
      return;
    }
  }

  elements.configSave.disabled = true;
  setConfigStatus("正在保存...");
  try {
    const config = await requestJson("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(changes),
    });
    configState.loaded = config;
    configState.models = {
      llm: config.llm_model || "",
      embedding: config.embedding_model || "",
    };
    elements.configOpenaiApiKey.value = "";
    elements.configEmbeddingApiKey.value = "";
    elements.configMineruToken.value = "";
    elements.configDialog.close();
    showToast(embeddingChanged ? "已更换向量模型，请重新入库文档" : "模型配置已保存");
    await refreshStatus();
  } catch (error) {
    setConfigStatus(error.message, "error");
  } finally {
    elements.configSave.disabled = false;
  }
}

elements.configButton.addEventListener("click", openConfig);
elements.configForm.addEventListener("submit", submitConfig);
elements.configClose.addEventListener("click", () => elements.configDialog.close());
elements.configCancel.addEventListener("click", () => elements.configDialog.close());
elements.configRefreshModels.addEventListener("click", () => {
  loadModels("llm");
  loadModels("embedding");
});
// Secrets are write-only, so clearing one needs an explicit, visible action.
elements.configMineruClear.addEventListener("click", () => {
  configState.clearMineruToken = true;
  elements.configMineruToken.value = "";
  elements.configMineruToken.placeholder = "保存后将清除 Token";
  renderMineruScope(false);
});
elements.configMineruToken.addEventListener("input", () => {
  if (!elements.configMineruToken.value.trim()) return;
  configState.clearMineruToken = false;
  renderMineruScope(true);
});
elements.configStructurizeProvider.addEventListener("change", () => {
  renderMineruScope(Boolean(configState.loaded?.mineru_api_token_set));
  renderStructurizeLabel(elements.configStructurizeProvider.value);
});
for (const target of ["llm", "embedding"]) {
  const { baseUrl, select, custom } = configTarget(target);
  // Live badge on typing; the model list only refetches once the address settles.
  baseUrl.addEventListener("input", () => renderScope(target));
  baseUrl.addEventListener("change", () => loadModels(target));
  select.addEventListener("change", () => {
    const isCustom = select.value === CUSTOM_MODEL;
    custom.hidden = !isCustom;
    if (isCustom) {
      custom.value = select.dataset.previous || configState.models[target] || "";
      custom.focus();
    } else {
      select.dataset.previous = select.value;
    }
    renderEmbeddingStatuses();
  });
  custom.addEventListener("input", renderEmbeddingStatuses);
}

const stagingState = { items: [], groups: [] };

const AUDIT_EVENT_LABELS = {
  "ingest.directory": "目录入库",
  "ingest.files": "上传入库",
  "delete.documents": "删除文档",
  "staging.submitted": "提交待审",
  "staging.approved": "审核通过",
  "staging.rejected": "审核拒绝",
};

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function formatTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value || "")
    : date.toLocaleString("zh-CN", { hour12: false });
}

async function loadStaging() {
  if (!canConfigure()) {
    elements.stagingSection.hidden = true;
    return;
  }
  try {
    const payload = await requestJson("/api/staging");
    stagingState.items = payload.items || [];
    stagingState.groups = payload.groups || [];
  } catch (_) {
    stagingState.items = [];
  }
  renderStaging();
}

function renderStaging() {
  const pending = stagingState.items.filter((item) => item.status === "pending");
  elements.stagingSection.hidden = !canConfigure() || !pending.length;
  elements.stagingList.replaceChildren();
  elements.stagingSummary.textContent = pending.length
    ? `${pending.length} 项待审核`
    : "空闲";
  for (const item of pending) {
    elements.stagingList.append(createStagingCard(item));
  }
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
  detail.textContent = [
    item.uploader,
    formatBytes(item.size_bytes),
    formatTime(item.created_at),
  ]
    .filter(Boolean)
    .join(" · ");

  const actions = document.createElement("div");
  actions.className = "staging-actions";

  const groupSelect = document.createElement("select");
  groupSelect.className = "config-select";
  groupSelect.setAttribute("aria-label", "入库后的访问组");
  const groups = stagingState.groups.length ? stagingState.groups : ["public"];
  for (const group of groups) {
    const option = document.createElement("option");
    option.value = group;
    option.textContent = group === "public" ? "public（所有人可见）" : group;
    groupSelect.append(option);
  }

  const reject = document.createElement("button");
  reject.type = "button";
  reject.className = "ghost-button";
  reject.textContent = "拒绝";
  reject.addEventListener("click", () =>
    decideStaging(item.id, "reject", "", reject),
  );

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
    await requestJson(
      `/api/staging/${encodeURIComponent(itemId)}/${action}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(action === "approve" ? { group } : {}),
      },
    );
    showToast(action === "approve" ? "已通过，开始入库" : "已拒绝");
    await loadStaging();
    if (action === "approve") startPolling();
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function loadAudit() {
  elements.auditStatus.textContent = "加载中…";
  try {
    const payload = await requestJson("/api/audit?limit=200");
    elements.auditNote.textContent = `日志文件：${payload.path}`;
    renderAudit(payload.entries || []);
    elements.auditStatus.textContent = `共 ${(payload.entries || []).length} 条`;
  } catch (error) {
    elements.auditStatus.textContent = error.message;
  }
}

function renderAudit(entries) {
  elements.auditList.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无写入记录";
    elements.auditList.append(empty);
    return;
  }
  for (const entry of entries) elements.auditList.append(createAuditRow(entry));
}

function createAuditRow(entry) {
  const row = document.createElement("article");
  row.className = "audit-row";

  const head = document.createElement("div");
  head.className = "audit-head";
  const event = document.createElement("strong");
  event.textContent = AUDIT_EVENT_LABELS[entry.event] || String(entry.event || "");
  const time = document.createElement("span");
  time.className = "audit-time";
  time.textContent = formatTime(entry.timestamp);
  head.append(event, time);

  const meta = document.createElement("div");
  meta.className = "audit-meta";
  meta.textContent = [
    `操作者 ${entry.actor}`,
    Array.isArray(entry.files) && entry.files.length
      ? entry.files.join("、")
      : "",
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
  if (entry.access_group) bits.push(`访问组 ${entry.access_group}`);
  if (entry.structurize_provider) bits.push(entry.structurize_provider);
  if (entry.embedding_model) bits.push(entry.embedding_model);
  stats.textContent = bits.join(" · ");

  row.append(head, meta, stats);
  return row;
}

async function openAudit() {
  if (!canConfigure()) return;
  elements.auditDialog.showModal();
  await loadAudit();
}

elements.auditButton.addEventListener("click", openAudit);
elements.auditClose.addEventListener("click", () => elements.auditDialog.close());
elements.auditRefresh.addEventListener("click", loadAudit);

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

// Opens the source document and, when a citation was clicked, lands on its chunk.
async function jumpToChunk(docId, chunkSuffix) {
  // Citations read "[source: <doc_id>#chunk_003]"; the indexed chunk_id is the
  // whole thing, prefix included.
  const chunkId =
    chunkSuffix && !chunkSuffix.includes("#")
      ? `${docId}#${chunkSuffix}`
      : chunkSuffix;

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
      return;
    }
  }
  await openDocument(indexedDocument, chunkId);
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

  elements.docTitle.textContent = indexedDocument.title || "未命名文档";
  elements.docMeta.textContent = [
    indexedDocument.source || "",
    `${docState.total} 个切片`,
  ]
    .filter(Boolean)
    .join(" · ");
  elements.docBody.replaceChildren();
  elements.docProgress.textContent = `已显示 0 / ${docState.total}`;
  elements.docLoadMore.hidden = true;
  elements.chatView.hidden = true;
  elements.documentView.hidden = false;

  await loadDocumentPage();
}

function closeDocument() {
  elements.documentView.hidden = true;
  elements.chatView.hidden = false;
  docState.docId = "";
  docState.chunks = [];
  docState.focusChunkId = "";
}

async function loadDocumentPage() {
  if (docState.loading || docState.done || !docState.docId) return;
  docState.loading = true;
  elements.docLoadMore.disabled = true;

  // Only the first page recentres on the cited chunk; later pages continue on.
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
    elements.docBody.replaceChildren();
    const failed = document.createElement("div");
    failed.className = "empty-list error";
    failed.textContent = error.message;
    elements.docBody.append(failed);
    docState.done = true;
    docState.loading = false;
    elements.docLoadMore.disabled = false;
    return;
  }
  docState.loading = false;
  elements.docLoadMore.disabled = false;
}

function renderDocumentView() {
  elements.docBody.replaceChildren();
  if (currentDocMode() === "merged") {
    renderMergedChunks();
  } else {
    renderChunkCards();
  }
  elements.docProgress.textContent =
    `已显示 ${docState.chunks.length} / ${docState.total}`;
  elements.docLoadMore.hidden = docState.done || docState.nextOffset >= docState.total;
  highlightFocusedChunk();
}

function highlightFocusedChunk() {
  if (!docState.focusChunkId) return;
  const target = elements.docBody.querySelector(
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
  if (chunk.section_path) {
    parts.push(chunk.section_path);
  }
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
    elements.docBody.append(card);
  }
}

function buildMergedSections() {
  const sections = [];
  let current = null;
  for (const chunk of docState.chunks) {
    const key =
      chunk.page === null || chunk.page === undefined
        ? "document"
        : `page-${chunk.page}`;
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
      elements.docBody.append(marker);
    }
    const block = document.createElement("div");
    block.className = "doc-merged-text";
    block.textContent = section.text;
    elements.docBody.append(block);
  }
}

elements.docBackButton.addEventListener("click", closeDocument);
elements.docLoadMore.addEventListener("click", loadDocumentPage);
elements.docViewMode.addEventListener("change", renderDocumentView);
elements.docBody.addEventListener("scroll", () => {
  if (docState.done || docState.loading) return;
  const { scrollTop, clientHeight, scrollHeight } = elements.docBody;
  if (scrollTop + clientHeight >= scrollHeight - 240) {
    loadDocumentPage();
  }
});

const RAIL_STORAGE_KEY = "rag-rail-widths";
const RAIL_LIMITS = {
  left: { min: 220, max: 520, default: 300, cssVar: "--rail-left" },
  right: { min: 260, max: 620, default: 348, cssVar: "--rail-right" },
};

function setRailWidth(side, width) {
  const limits = RAIL_LIMITS[side];
  const clamped = Math.round(Math.min(limits.max, Math.max(limits.min, width)));
  elements.workspace.style.setProperty(limits.cssVar, `${clamped}px`);
  return clamped;
}

function currentRailWidth(side) {
  const rail = side === "left" ? elements.railLibrary : elements.railContext;
  return rail.getBoundingClientRect().width || RAIL_LIMITS[side].default;
}

function saveRailWidths() {
  try {
    localStorage.setItem(
      RAIL_STORAGE_KEY,
      JSON.stringify({
        left: Math.round(currentRailWidth("left")),
        right: Math.round(currentRailWidth("right")),
      }),
    );
  } catch (_) {
    /* localStorage unavailable (private mode); width stays session-only */
  }
}

function restoreRailWidths() {
  try {
    const saved = JSON.parse(localStorage.getItem(RAIL_STORAGE_KEY) || "{}");
    for (const side of ["left", "right"]) {
      if (Number.isFinite(saved[side])) {
        setRailWidth(side, saved[side]);
      }
    }
  } catch (_) {
    /* ignore malformed storage */
  }
}

function enableRailResize(handle, side) {
  const limits = RAIL_LIMITS[side];
  const direction = side === "left" ? 1 : -1;
  let startX = 0;
  let startWidth = 0;
  let dragging = false;

  handle.addEventListener("pointerdown", (event) => {
    dragging = true;
    startX = event.clientX;
    startWidth = currentRailWidth(side);
    handle.classList.add("dragging");
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  handle.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    setRailWidth(side, startWidth + (event.clientX - startX) * direction);
  });

  const stop = (event) => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    if (handle.hasPointerCapture(event.pointerId)) {
      handle.releasePointerCapture(event.pointerId);
    }
    saveRailWidths();
  };
  handle.addEventListener("pointerup", stop);
  handle.addEventListener("pointercancel", stop);

  handle.addEventListener("dblclick", () => {
    setRailWidth(side, limits.default);
    saveRailWidths();
  });

  handle.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const step = (event.shiftKey ? 40 : 12) * direction;
    const delta = event.key === "ArrowRight" ? step : -step;
    setRailWidth(side, currentRailWidth(side) + delta);
    saveRailWidths();
  });
}

restoreRailWidths();
enableRailResize(elements.resizerLeft, "left");
enableRailResize(elements.resizerRight, "right");

renderContext();
renderProcess();
refreshDocuments();
pollJobs();
loadIdentity();
