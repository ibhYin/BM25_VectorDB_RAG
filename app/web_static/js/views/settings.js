// 设置页：模型接入配置（LLM / Embedding / MinerU）与索引状态。

import { on, refreshStatus, requestJson, showToast, state } from "../core.js";

const el = {
  form: document.querySelector("#config-form"),
  status: document.querySelector("#config-status"),
  save: document.querySelector("#config-save"),
  reset: document.querySelector("#config-reset"),

  openaiBaseUrl: document.querySelector("#config-openai-base-url"),
  openaiApiKey: document.querySelector("#config-openai-api-key"),
  llmModel: document.querySelector("#config-llm-model"),
  llmModelCustom: document.querySelector("#config-llm-model-custom"),
  llmScope: document.querySelector("#config-llm-scope"),
  llmModelsHint: document.querySelector("#config-llm-models-hint"),

  embeddingBaseUrl: document.querySelector("#config-embedding-base-url"),
  embeddingApiKey: document.querySelector("#config-embedding-api-key"),
  embeddingModel: document.querySelector("#config-embedding-model"),
  embeddingModelCustom: document.querySelector("#config-embedding-model-custom"),
  embeddingScope: document.querySelector("#config-embedding-scope"),
  embeddingModelsHint: document.querySelector("#config-embedding-models-hint"),
  embeddingWarning: document.querySelector("#config-embedding-warning"),
  indexModel: document.querySelector("#config-index-model"),

  mineruProvider: document.querySelector("#config-structurize-provider"),
  mineruToken: document.querySelector("#config-mineru-token"),
  mineruClear: document.querySelector("#config-mineru-clear"),
  mineruVersion: document.querySelector("#config-mineru-model-version"),
  mineruScope: document.querySelector("#config-mineru-scope"),

  refreshModels: document.querySelector("#config-refresh-models"),

  indexBuiltBy: document.querySelector("#index-built-by"),
  indexMismatch: document.querySelector("#index-mismatch"),
  pathData: document.querySelector("#path-data"),
  pathChroma: document.querySelector("#path-chroma"),
  pathConfig: document.querySelector("#path-config"),
};

let mounted = false;
const configState = { loaded: null, models: { llm: "", embedding: "" }, clearMineruToken: false };
const CUSTOM_MODEL = "__custom__";

// --- helpers --------------------------------------------------------------

function setConfigStatus(message, variant = "") {
  el.status.textContent = message;
  el.status.className = `config-status${variant ? ` ${variant}` : ""}`;
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
    baseUrl: llm ? el.openaiBaseUrl : el.embeddingBaseUrl,
    apiKey: llm ? el.openaiApiKey : el.embeddingApiKey,
    select: llm ? el.llmModel : el.embeddingModel,
    custom: llm ? el.llmModelCustom : el.embeddingModelCustom,
    scope: llm ? el.llmScope : el.embeddingScope,
    hint: llm ? el.llmModelsHint : el.embeddingModelsHint,
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
  if (select.value !== CUSTOM_MODEL) select.dataset.previous = select.value;
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

async function loadModels(target) {
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
      body: JSON.stringify({ target, base_url: value, api_key: apiKey.value.trim() }),
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

// --- embedding / mineru status -------------------------------------------

function renderEmbeddingStatuses() {
  const loaded = configState.loaded || {};
  const saved = loaded.embedding_model || "";
  el.embeddingWarning.classList.toggle(
    "warning",
    Boolean(saved) && modelValue("embedding") !== saved,
  );

  const indexModel = loaded.index_embedding_model || "";
  el.indexModel.className = "config-hint";
  if (!indexModel) {
    el.indexModel.textContent = "尚未记录建库所用的向量模型。";
    return;
  }
  el.indexModel.textContent = `当前索引由 ${indexModel} 建立。`;
  if (indexModel !== modelValue("embedding")) el.indexModel.classList.add("warning");
}

function renderMineruScope(configured) {
  el.mineruScope.textContent = configured ? "已启用" : "未配置";
  el.mineruScope.className = `scope-badge ${configured ? "cloud" : "empty"}`;
}

// --- load / save ----------------------------------------------------------

function renderIndexStatus() {
  const status = state.status || {};
  const builtBy = status.index_embedding_model || "";
  el.indexBuiltBy.textContent = builtBy || "（未记录）";
  el.indexMismatch.hidden = !status.embedding_model_mismatch;

  el.pathData.textContent = status.data_dir || "—";
  el.pathChroma.textContent = status.chroma_persist_dir || "—";
  el.pathConfig.textContent = status.runtime_config_file || "—";
}

async function loadConfig() {
  setConfigStatus("正在读取当前配置...");
  el.save.disabled = true;
  try {
    const config = await requestJson("/api/config");
    configState.loaded = config;
    el.openaiBaseUrl.value = config.openai_base_url || "";
    el.embeddingBaseUrl.value = config.embedding_base_url || "";
    configState.models = {
      llm: config.llm_model || "",
      embedding: config.embedding_model || "",
    };
    el.openaiApiKey.placeholder = config.openai_api_key_set
      ? `已配置 ${config.openai_api_key_hint} · 留空保持不变`
      : "留空则保持不变";
    el.embeddingApiKey.placeholder = config.embedding_api_key_set
      ? `已配置 ${config.embedding_api_key_hint} · 留空保持不变`
      : "留空则保持不变";
    el.mineruVersion.value = config.mineru_model_version || "vlm";
    el.mineruProvider.value = config.structurize_provider || "markitdown";
    el.mineruToken.placeholder = config.mineru_api_token_set
      ? `已配置 ${config.mineru_api_token_hint} · 留空保持不变`
      : "留空则保持不变";
    configState.clearMineruToken = false;
    renderMineruScope(Boolean(config.mineru_api_token_set));
    setConfigStatus("");
  } catch (error) {
    setConfigStatus(error.message, "error");
  } finally {
    el.save.disabled = false;
  }

  renderModelSelect("llm", []);
  renderModelSelect("embedding", []);
  renderScope("llm");
  renderScope("embedding");
  renderEmbeddingStatuses();
  await Promise.all([loadModels("llm"), loadModels("embedding")]);
}

// Only send fields the user actually changed; secrets are never echoed back,
// so a blank key field always means "keep the existing key".
function collectConfigChanges() {
  const loaded = configState.loaded || {};
  const changes = {};

  const baseUrl = el.openaiBaseUrl.value.trim();
  if (baseUrl !== (loaded.openai_base_url || "")) changes.openai_base_url = baseUrl;
  const embeddingBaseUrl = el.embeddingBaseUrl.value.trim();
  if (embeddingBaseUrl !== (loaded.embedding_base_url || "")) {
    changes.embedding_base_url = embeddingBaseUrl;
  }
  const llmModel = modelValue("llm");
  if (llmModel !== (loaded.llm_model || "")) changes.llm_model = llmModel;
  const embeddingModel = modelValue("embedding");
  if (embeddingModel !== (loaded.embedding_model || "")) {
    changes.embedding_model = embeddingModel;
  }
  const openaiKey = el.openaiApiKey.value.trim();
  if (openaiKey) changes.openai_api_key = openaiKey;
  const embeddingKey = el.embeddingApiKey.value.trim();
  if (embeddingKey) changes.embedding_api_key = embeddingKey;
  const mineruToken = el.mineruToken.value.trim();
  if (mineruToken) changes.mineru_api_token = mineruToken;
  if (configState.clearMineruToken) changes.clear_mineru_api_token = true;
  if (el.mineruVersion.value !== (loaded.mineru_model_version || "vlm")) {
    changes.mineru_model_version = el.mineruVersion.value;
  }
  if (el.mineruProvider.value !== (loaded.structurize_provider || "markitdown")) {
    changes.structurize_provider = el.mineruProvider.value;
  }
  return changes;
}

async function submitConfig(event) {
  event.preventDefault();
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

  el.save.disabled = true;
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
    el.openaiApiKey.value = "";
    el.embeddingApiKey.value = "";
    el.mineruToken.value = "";
    configState.clearMineruToken = false;
    renderMineruScope(Boolean(config.mineru_api_token_set));
    showToast(embeddingChanged ? "已更换向量模型，请重新入库文档" : "配置已保存");
    await refreshStatus();
    renderIndexStatus();
    setConfigStatus("已保存");
  } catch (error) {
    setConfigStatus(error.message, "error");
  } finally {
    el.save.disabled = false;
  }
}

// --- view contract --------------------------------------------------------

export default {
  name: "settings",

  mount() {
    if (mounted) return;
    mounted = true;

    el.form.addEventListener("submit", submitConfig);
    el.reset.addEventListener("click", loadConfig);
    el.refreshModels.addEventListener("click", () => {
      loadModels("llm");
      loadModels("embedding");
    });

    el.mineruClear.addEventListener("click", () => {
      configState.clearMineruToken = true;
      el.mineruToken.value = "";
      el.mineruToken.placeholder = "保存后将清除 Token";
      renderMineruScope(false);
    });
    el.mineruToken.addEventListener("input", () => {
      if (!el.mineruToken.value.trim()) return;
      configState.clearMineruToken = false;
      renderMineruScope(true);
    });

    for (const target of ["llm", "embedding"]) {
      const { baseUrl, select, custom } = configTarget(target);
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

    on("status", renderIndexStatus);
  },

  show() {
    renderIndexStatus();
    loadConfig();
  },
};
