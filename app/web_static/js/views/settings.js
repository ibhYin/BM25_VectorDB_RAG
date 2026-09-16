// 设置页：模型接入配置（LLM / Embedding / MinerU）与索引状态。

import { formatBytes, on, refreshStatus, requestJson, showToast, state } from "../core.js";
import { createModelPicker } from "../model-picker.js";

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
  pathCache: document.querySelector("#path-cache"),
};

let mounted = false;
const configState = { loaded: null, models: { llm: "", embedding: "" }, clearMineruToken: false };

// 模型选择控件（下拉 + 自定义输入）交给共享模块。字段与提交仍归本页管，
// picker 只负责展示——它已经被问答页复用，所以不能耦合本页的 configState。
const pickers = {
  llm: createModelPicker({
    select: el.llmModel,
    custom: el.llmModelCustom,
    hint: el.llmModelsHint,
    scopeBadge: el.llmScope,
    target: "llm",
    requireBaseUrl: true,
  }),
  embedding: createModelPicker({
    select: el.embeddingModel,
    custom: el.embeddingModelCustom,
    hint: el.embeddingModelsHint,
    scopeBadge: el.embeddingScope,
    target: "embedding",
    requireBaseUrl: true,
  }),
};

// --- helpers --------------------------------------------------------------

function setConfigStatus(message, variant = "") {
  el.status.textContent = message;
  el.status.className = `config-status${variant ? ` ${variant}` : ""}`;
}

// 模型下拉/自定义框/徽章已经归 picker 管，这里只剩提交与拉列表要用的地址和密钥。
function configTarget(target) {
  const llm = target === "llm";
  return {
    baseUrl: llm ? el.openaiBaseUrl : el.embeddingBaseUrl,
    apiKey: llm ? el.openaiApiKey : el.embeddingApiKey,
  };
}

/** 把「当前保存的模型」告诉 picker：它不读 configState（已被问答页复用）。 */
function syncPickers() {
  pickers.llm.setValue(configState.models.llm);
  pickers.embedding.setValue(configState.models.embedding);
}

async function loadModels(target) {
  const { baseUrl, apiKey } = configTarget(target);
  await pickers[target].load({ baseUrl: baseUrl.value, apiKey: apiKey.value });
  renderEmbeddingStatuses();
}

// --- embedding / mineru status -------------------------------------------

function renderEmbeddingStatuses() {
  const loaded = configState.loaded || {};
  const saved = loaded.embedding_model || "";
  el.embeddingWarning.classList.toggle(
    "warning",
    Boolean(saved) && pickers.embedding.value() !== saved,
  );

  const indexModel = loaded.index_embedding_model || "";
  el.indexModel.className = "config-hint";
  if (!indexModel) {
    el.indexModel.textContent = "尚未记录建库所用的向量模型。";
    return;
  }
  el.indexModel.textContent = `当前索引由 ${indexModel} 建立。`;
  if (indexModel !== pickers.embedding.value()) el.indexModel.classList.add("warning");
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

  // 缓存可随时删除，删了下次入库重新解析；这里顺带显示条目数方便判断。
  const cache = status.markdown_cache || {};
  const entries = Number(cache.entries) || 0;
  el.pathCache.textContent = cache.dir
    ? entries
      ? `${cache.dir}（${entries} 项 · ${formatBytes(cache.bytes)}）`
      : `${cache.dir}（空）`
    : "—";
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
    syncPickers();
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

  pickers.llm.renderEmpty();
  pickers.embedding.renderEmpty();
  pickers.llm.renderScope();
  pickers.embedding.renderScope();
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
  const llmModel = pickers.llm.value();
  if (llmModel !== (loaded.llm_model || "")) changes.llm_model = llmModel;
  const embeddingModel = pickers.embedding.value();
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
    syncPickers();
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
      const { baseUrl } = configTarget(target);
      const picker = pickers[target];
      baseUrl.addEventListener("input", () => picker.renderScope());
      baseUrl.addEventListener("change", () => loadModels(target));
      picker.select.addEventListener("change", () => {
        picker.syncFromSelect({ focus: true });
        renderEmbeddingStatuses();
      });
      picker.custom.addEventListener("input", renderEmbeddingStatuses);
    }

    on("status", renderIndexStatus);
  },

  show() {
    renderIndexStatus();
    loadConfig();
  },
};
