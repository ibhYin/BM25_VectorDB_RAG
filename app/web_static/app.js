const state = {
  vectorHits: [],
  answerJob: null,
  jobs: [],
  documents: [],
  polling: null,
};

const elements = {
  serviceSummary: document.querySelector("#service-summary"),
  vectorStatus: document.querySelector("#vector-status"),
  modelStatus: document.querySelector("#model-status"),
  refreshButton: document.querySelector("#refresh-button"),
  documentCount: document.querySelector("#document-count"),
  chunkCount: document.querySelector("#chunk-count"),
  uploadLimit: document.querySelector("#upload-limit"),
  dropZone: document.querySelector("#drop-zone"),
  fileInput: document.querySelector("#file-input"),
  chooseFilesButton: document.querySelector("#choose-files-button"),
  queueSummary: document.querySelector("#queue-summary"),
  jobList: document.querySelector("#job-list"),
  reloadDocumentsButton: document.querySelector("#reload-documents-button"),
  documentList: document.querySelector("#document-list"),
  runStatus: document.querySelector("#run-status"),
  clearChatButton: document.querySelector("#clear-chat-button"),
  chatStream: document.querySelector("#chat-stream"),
  askForm: document.querySelector("#ask-form"),
  askButton: document.querySelector("#ask-button"),
  question: document.querySelector("#question"),
  contextList: document.querySelector("#context-list"),
  processList: document.querySelector("#process-list"),
};

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
    elements.documentCount.textContent = String(state.documents.length);
    elements.chunkCount.textContent = String(chunkCount);
  } catch (error) {
    elements.serviceSummary.textContent = "服务连接失败";
    elements.vectorStatus.textContent = "服务不可用";
    elements.modelStatus.textContent = "服务不可用";
    elements.documentCount.textContent = "--";
    elements.chunkCount.textContent = "--";
  } finally {
    setStatusLoading(false);
    elements.refreshButton.disabled = false;
  }
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

    const title = document.createElement("strong");
    title.textContent = indexedDocument.title || "未命名文档";

    const source = document.createElement("span");
    source.className = "document-source";
    source.textContent = indexedDocument.source || "未知来源";

    const chunks = document.createElement("span");
    chunks.className = "chunk-count";
    chunks.textContent = `${indexedDocument.chunk_count} 片`;

    item.append(title, source, chunks);
    elements.documentList.append(item);
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
  const question = elements.question.value.trim();
  if (!question) {
    elements.question.focus();
    return;
  }

  addChatMessage("question", question);
  elements.question.value = "";
  elements.askButton.disabled = true;
  setRunStatusLoading(true, "正在处理问题...");
  state.answerJob = null;
  renderProcess();
  renderContextLoading();

  try {
    const job = await requestJson("/api/answer-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
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
    if (job.vector_hits && job.vector_hits.length) {
      state.vectorHits = job.vector_hits;
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
    setRunStatusLoading(true, "正在搜索向量数据库...");
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
    title.textContent = label;
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
  return "·";
}

function stageDetail(key, stage) {
  const detail = stage.detail || {};
  const duration = detail.duration_ms !== undefined ? ` · ${detail.duration_ms} ms` : "";
  if (stage.status === "pending") return "等待执行";
  if (stage.status === "running") return "正在执行";
  if (stage.status === "failed") return "执行失败";
  if (key === "encode") {
    const preview = Array.isArray(detail.vector_preview)
      ? ` · 前 8 维 [${detail.vector_preview.join(", ")}]`
      : "";
    return `${detail.model || "Embedding"} · ${detail.dimensions || 0} 维${duration}${preview}`;
  }
  if (key === "vector_search") {
    return `${detail.matches || 0} 个匹配 · Top ${detail.top_k || 0}${duration}`;
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

function addChatMessage(type, text) {
  const empty = elements.chatStream.querySelector(".empty-chat");
  if (empty) {
    empty.remove();
  }

  const message = document.createElement("article");
  message.className = `chat-message ${type}`;
  message.textContent = text;
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
  distance.title = "Chroma 距离，数值越小表示越相近";
  distance.textContent = `距离 ${Number(hit.distance).toFixed(4)}`;
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
  await Promise.all([refreshStatus(), refreshDocuments()]);
});
elements.reloadDocumentsButton.addEventListener("click", refreshDocuments);
elements.chooseFilesButton.addEventListener("click", () => elements.fileInput.click());
elements.dropZone.addEventListener("click", () => elements.fileInput.click());
elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
});
elements.fileInput.addEventListener("change", () => {
  uploadFiles(elements.fileInput.files);
  elements.fileInput.value = "";
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
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

renderContext();
renderProcess();
refreshDocuments();
pollJobs();
