// 设置页：模型接入配置（LLM / Embedding / Rerank / MinerU / ASR）与索引状态。

import { formatBytes, on, refreshStatus, requestJson, showToast, state } from "../core.js";
import { createModelPicker } from "../model-picker.js";
import { resetSelectSearch } from "../select-search.js";

const el = {
  rerankUrl: document.querySelector("#config-rerank-url"),
  rerankApiKey: document.querySelector("#config-rerank-api-key"),
  rerankModel: document.querySelector("#config-rerank-model"),
  rerankCustom: document.querySelector("#config-rerank-model-custom"),
  rerankHint: document.querySelector("#config-rerank-models-hint"),
  rerankRefreshModels: document.querySelector("#config-rerank-refresh-models"),
  rerankPool: document.querySelector("#config-rerank-pool-size"),
  rerankTimeout: document.querySelector("#config-rerank-timeout"),
  chunkSize: document.querySelector("#config-chunk-size"),
  chunkOverlap: document.querySelector("#config-chunk-overlap"),
  embeddingMaxInputBytes: document.querySelector("#config-embedding-max-input-bytes"),
  embeddingBatchSize: document.querySelector("#config-embedding-batch-size"),
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

  mineruToken: document.querySelector("#config-mineru-token"),
  mineruClear: document.querySelector("#config-mineru-clear"),
  mineruVersion: document.querySelector("#config-mineru-model-version"),
  mineruScope: document.querySelector("#config-mineru-scope"),

  asrProvider: document.querySelector("#config-asr-provider"),
  asrOpenai: document.querySelector("#config-asr-openai"),
  asrAliyun: document.querySelector("#config-asr-aliyun"),
  asrBaseUrl: document.querySelector("#config-asr-base-url"),
  asrApiKey: document.querySelector("#config-asr-api-key"),
  asrModel: document.querySelector("#config-asr-model"),
  asrScope: document.querySelector("#config-asr-scope"),

  aliyunAkId: document.querySelector("#config-aliyun-ak-id"),
  aliyunAkSecret: document.querySelector("#config-aliyun-ak-secret"),
  aliyunAppkey: document.querySelector("#config-aliyun-appkey"),
  aliyunRegion: document.querySelector("#config-aliyun-region"),
  aliyunBucket: document.querySelector("#config-aliyun-bucket"),
  aliyunEndpoint: document.querySelector("#config-aliyun-endpoint"),
  aliyunPrefix: document.querySelector("#config-aliyun-prefix"),

  refreshModels: document.querySelector("#config-refresh-models"),

  indexBuiltBy: document.querySelector("#index-built-by"),
  indexMismatch: document.querySelector("#index-mismatch"),
  pathData: document.querySelector("#path-data"),
  pathChroma: document.querySelector("#path-chroma"),
  pathConfig: document.querySelector("#path-config"),
  pathCache: document.querySelector("#path-cache"),
  codeVersion: document.querySelector("#code-version"),
};

let mounted = false;
const configState = { loaded: null, clearMineruToken: false };

// 模型选择控件（下拉 + 自定义输入）交给共享模块。字段与提交仍归本页管，
// picker 只负责展示——它已经被问答页复用，所以不能耦合本页的 configState。
const pickers = {
  rerank: createModelPicker({ select: el.rerankModel, custom: el.rerankCustom,
    hint: el.rerankHint, target: "rerank", requireBaseUrl: true }),
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

function renderEmbeddingStatuses() {
  pickers.llm.renderScope(el.openaiBaseUrl.value);
  pickers.embedding.renderScope(el.embeddingBaseUrl.value || el.openaiBaseUrl.value);
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

function renderAsrScope(configured) {
  el.asrScope.textContent = configured ? "已启用" : "未配置";
  el.asrScope.className = `scope-badge ${configured ? "cloud" : "empty"}`;
}

function asrConfigured(loaded) {
  const provider = (loaded && loaded.audio_transcribe_provider) || el.asrProvider.value || "openai";
  if (provider === "aliyun") {
    return Boolean(loaded && loaded.aliyun_appkey_set && loaded.aliyun_oss_bucket);
  }
  return Boolean((loaded && loaded.audio_transcribe_model) || el.asrModel.value.trim());
}

function toggleAsrFields() {
  const provider = el.asrProvider.value;
  el.asrOpenai.hidden = provider !== "openai";
  el.asrAliyun.hidden = provider !== "aliyun";
  for (const key of ["aliyun_access_key_id", "aliyun_access_key_secret", "aliyun_appkey", "aliyun_region", "aliyun_oss_bucket"]) {
    fields[key].required = provider === "aliyun" && (!secretKeys.has(key) || !configState.loaded?.[`${key}_set`]);
  }
}

function renderIndexStatus() {
  const status = state.status || {};
  const builtBy = status.index_embedding_model || "";
  el.indexBuiltBy.textContent = builtBy || "（未记录）";
  el.indexMismatch.hidden = !status.embedding_model_mismatch;

  el.pathData.textContent = status.data_dir || "—";
  el.pathChroma.textContent = status.chroma_persist_dir || "—";
  el.pathConfig.textContent = status.runtime_config_file || "—";
  el.codeVersion.textContent = status.code_version || "—";

  // 缓存可随时删除，删了下次入库重新解析；这里顺带显示条目数方便判断。
  const cache = status.markdown_cache || {};
  const entries = Number(cache.entries) || 0;
  el.pathCache.textContent = cache.dir
    ? entries
      ? `${cache.dir}（${entries} 项 · ${formatBytes(cache.bytes)}）`
      : `${cache.dir}（空）`
    : "—";
}

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
  const asrBaseUrl = el.asrBaseUrl.value.trim();
  if (asrBaseUrl !== (loaded.audio_transcribe_base_url || "")) {
    changes.audio_transcribe_base_url = asrBaseUrl;
  }
  const asrModel = el.asrModel.value.trim();
  if (asrModel !== (loaded.audio_transcribe_model || "")) {
    changes.audio_transcribe_model = asrModel;
  }
  const asrKey = el.asrApiKey.value.trim();
  if (asrKey) changes.audio_transcribe_api_key = asrKey;
  const provider = el.asrProvider.value;
  if (provider !== (loaded.audio_transcribe_provider || "openai")) {
    changes.audio_transcribe_provider = provider;
  }
  const akId = el.aliyunAkId.value.trim();
  if (akId) changes.aliyun_access_key_id = akId;
  const akSecret = el.aliyunAkSecret.value.trim();
  if (akSecret) changes.aliyun_access_key_secret = akSecret;
  const appkey = el.aliyunAppkey.value.trim();
  if (appkey) changes.aliyun_appkey = appkey;
  const region = el.aliyunRegion.value.trim();
  if (region && region !== (loaded.aliyun_region || "cn-shanghai")) {
    changes.aliyun_region = region;
  }
  const bucket = el.aliyunBucket.value.trim();
  if (bucket !== (loaded.aliyun_oss_bucket || "")) changes.aliyun_oss_bucket = bucket;
  const endpoint = el.aliyunEndpoint.value.trim();
  if (endpoint !== (loaded.aliyun_oss_endpoint || "")) {
    changes.aliyun_oss_endpoint = endpoint;
  }
  const prefix = el.aliyunPrefix.value.trim();
  if (prefix !== (loaded.aliyun_oss_prefix || "")) changes.aliyun_oss_prefix = prefix;
  for (const key of ["chunk_size", "chunk_overlap", "embedding_max_input_bytes", "embedding_batch_size"]) {
    const value = Number(fields[key].value);
    if (value !== Number(loaded[key])) changes[key] = value;
  }
  if (el.rerankUrl.value.trim() !== (loaded.rerank_url || "")) changes.rerank_url = el.rerankUrl.value.trim();
  if (el.rerankApiKey.value.trim()) changes.rerank_api_key = el.rerankApiKey.value.trim();
  if (pickers.rerank.value() !== (loaded.rerank_model || "")) changes.rerank_model = pickers.rerank.value();
  for (const key of ["rerank_pool_size", "rerank_timeout"]) {
    const value = Number(fields[key].value);
    if (value !== Number(loaded[key])) changes[key] = value;
  }
  return changes;
}

const fields = {
  rerank_url: el.rerankUrl, rerank_api_key: el.rerankApiKey,
  rerank_pool_size: el.rerankPool, rerank_timeout: el.rerankTimeout,
  openai_base_url: el.openaiBaseUrl, openai_api_key: el.openaiApiKey,
  embedding_base_url: el.embeddingBaseUrl, embedding_api_key: el.embeddingApiKey,
  mineru_api_token: el.mineruToken, mineru_model_version: el.mineruVersion,
  audio_transcribe_provider: el.asrProvider, audio_transcribe_base_url: el.asrBaseUrl,
  audio_transcribe_api_key: el.asrApiKey, audio_transcribe_model: el.asrModel,
  aliyun_access_key_id: el.aliyunAkId, aliyun_access_key_secret: el.aliyunAkSecret,
  aliyun_appkey: el.aliyunAppkey, aliyun_region: el.aliyunRegion,
  aliyun_oss_bucket: el.aliyunBucket, aliyun_oss_endpoint: el.aliyunEndpoint,
  aliyun_oss_prefix: el.aliyunPrefix, chunk_size: el.chunkSize, chunk_overlap: el.chunkOverlap,
  embedding_max_input_bytes: el.embeddingMaxInputBytes, embedding_batch_size: el.embeddingBatchSize,
};
const sections = {
  rerank: ["rerank_url", "rerank_api_key", "rerank_model", "rerank_pool_size", "rerank_timeout"],
  llm: ["openai_base_url", "openai_api_key", "llm_model"],
  embedding: ["embedding_base_url", "embedding_api_key", "embedding_model", "chunk_size", "chunk_overlap", "embedding_max_input_bytes", "embedding_batch_size"],
  mineru: ["mineru_api_token", "clear_mineru_api_token", "mineru_model_version"],
  asr: Object.keys(fields).filter(key => key.startsWith("audio_") || key.startsWith("aliyun_")),
};
const secretKeys = new Set(["rerank_api_key", "openai_api_key", "embedding_api_key", "mineru_api_token", "audio_transcribe_api_key", "aliyun_access_key_id", "aliyun_access_key_secret", "aliyun_appkey"]);
const actions = Object.fromEntries(Object.keys(sections).map(name => [name, {
  save: document.querySelector(`#config-${name}-save`),
  reset: document.querySelector(`#config-${name}-reset`),
  status: document.querySelector(`#config-${name}-status`),
  group: document.querySelector(`#config-section-${name}`),
}]));
let saving = false;
let loading = false;
let refreshing = false;
const messages = {};

function changesFor(name) {
  const changes = collectConfigChanges();
  return name === "all" ? changes : Object.fromEntries(Object.entries(changes).filter(([key]) => sections[name].includes(key)));
}

function message(name, text, error = false) {
  messages[name] = { text, error };
  const node = name === "all" ? el.status : actions[name].status;
  node.textContent = text;
  node.className = `config-status${error ? " error" : ""}`;
}

function renderActions(edited = "") {
  for (const name of Object.keys(sections)) {
    const dirty = configState.loaded && Object.keys(changesFor(name)).length > 0;
    actions[name].save.disabled = !dirty || saving;
    actions[name].reset.disabled = !dirty || saving;
    if (edited === name || edited === "all") message(name, dirty ? "有未保存修改" : "与已保存配置一致");
  }
  const dirty = configState.loaded && Object.keys(changesFor("all")).length > 0;
  el.save.disabled = !dirty || saving;
  el.reset.disabled = !dirty || saving;
  if (edited) message("all", dirty ? "有未保存修改" : "与已保存配置一致");
}

function applySection(name) {
  const config = configState.loaded;
  for (const key of sections[name]) {
    const field = fields[key];
    if (!field) continue;
    if (secretKeys.has(key)) {
      field.value = "";
      field.placeholder = config[`${key}_set`] ? `已配置 ${config[`${key}_hint`]} · 留空保持不变` : "留空保持不变";
    } else field.value = config[key] ?? field.defaultValue ?? "";
  }
  if (pickers[name]) pickers[name].setValue(config[`${name}_model`] || "");
  for (const select of actions[name].group.querySelectorAll("select")) resetSelectSearch(select);
  if (name === "mineru") configState.clearMineruToken = false;
  toggleAsrFields();
  renderMineruScope(Boolean(config.mineru_api_token_set));
  renderAsrScope(asrConfigured(config));
  renderEmbeddingStatuses();
}

async function loadConfig() {
  if (loading || configState.loaded) return;
  loading = true;
  message("all", "正在读取配置…");
  try {
    configState.loaded = await requestJson("/api/config");
    Object.keys(sections).forEach(applySection);
    message("all", "已读取；密钥留空保持不变，MinerU 清除需单独保存本节");
    renderActions();
    await refreshModels();
  } catch (error) {
    message("all", error.message, true);
  } finally { loading = false; }
}

function discard(name) {
  if (saving || !configState.loaded || !Object.keys(changesFor(name)).length) return;
  const names = name === "all" ? Object.keys(sections) : [name];
  names.forEach(section => { applySection(section); message(section, "已恢复最近一次成功保存的值"); });
  if (name === "all") message("all", "已恢复全部已保存值");
  renderActions();
}

async function save(name) {
  if (saving || !configState.loaded) return;
  const changes = changesFor(name);
  if (!Object.keys(changes).length) return;
  const names = name === "all" ? Object.keys(sections) : [name];
  if (names.some(section => ["llm", "embedding", "rerank"].includes(section) && !pickers[section].value())) {
    message(name, "请选择或填写本节的模型名", true); return;
  }
  const dirtyBefore = Object.fromEntries(Object.keys(sections).map(key => [key, Boolean(Object.keys(changesFor(key)).length)]));
  const button = name === "all" ? el.save : actions[name].save;
  if (Object.entries(changes).some(([key, value]) => secretKeys.has(key) && /[•]|\*\*\*|已配置/.test(value))) {
    message(name, "请填写真实密钥，不能提交脱敏占位符", true); return;
  }
  // Validate only this section; unrelated unfinished fields must not block it.
  if (!names.every(section => [...actions[section].group.querySelectorAll("input, select")].every(field => field.reportValidity()))) return;
  saving = true;
  names.forEach(section => { actions[section].group.disabled = true; });
  button.textContent = "保存中…";
  message(name, "保存中…");
  renderActions();
  try {
    const config = await requestJson("/api/config", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...(name === "all" ? {} : { section: name }), ...changes }),
    });
    const missed = Object.keys(changes).filter(key => !(config.updated || []).includes(key === "clear_mineru_api_token" ? "mineru_api_token" : key));
    if (missed.length) throw new Error(`后端未确认保存：${missed.join("、")}`);
    // Preserve other dirty sections, even if server-side endpoint fallback changed.
    configState.loaded = config;
    for (const section of Object.keys(sections)) {
      if (names.includes(section) || (!dirtyBefore[section] && !Object.keys(changesFor(section)).length)) applySection(section);
    }
    const clock = new Date().toLocaleTimeString("zh-CN", { hour12: false });
    names.forEach(section => message(section, `已保存 · ${clock}`));
    message(name, `已保存 · ${clock}${"embedding_model" in changes ? "；目标模型已保存，请重建索引" : ""}`);
    await refreshStatus();
    renderIndexStatus();
  } catch (error) { message(name, `保存失败：${error.message}；输入已保留，可重试`, true); }
  finally {
    saving = false;
    names.forEach(section => { actions[section].group.disabled = false; });
    button.textContent = name === "all" ? "保存全部修改" : "保存本节";
    renderActions();
  }
}

async function refreshModels() {
  if (refreshing) return;
  const feedback = document.querySelector("#config-models-feedback");
  if ([el.openaiApiKey, el.embeddingApiKey, el.rerankApiKey].some(field => /[•]|\*\*\*|已配置/.test(field.value))) {
    feedback.textContent = "刷新失败：请使用真实密钥或留空使用已保存密钥，不能提交脱敏占位符"; return;
  }
  refreshing = true;
  el.refreshModels.disabled = true;
  el.rerankRefreshModels.disabled = true;
  el.refreshModels.textContent = "刷新中…";
  feedback.textContent = "刷新中…使用当前输入的地址与密钥；密钥留空使用已保存密钥，不保存配置。";
  try {
    const results = await Promise.all(["llm", "embedding", "rerank"].map(target => pickers[target].load({
      baseUrl: target === "rerank" ? el.rerankUrl.value : target === "llm" ? el.openaiBaseUrl.value : (el.embeddingBaseUrl.value.trim() || el.openaiBaseUrl.value),
      apiKey: (target === "rerank" ? el.rerankApiKey : target === "llm" ? el.openaiApiKey : el.embeddingApiKey).value,
    })));
    feedback.textContent = results.map((result, index) => `${["LLM", "Embedding", "重排"][index]}：${result.error ? `刷新失败 ${result.error}` : `${result.count} 个模型`}`).join("；");
    feedback.classList.toggle("error", results.some(result => result.error));
  } catch (error) { feedback.textContent = `刷新失败：${error.message}；可重试`; }
  finally {
    refreshing = false;
    el.refreshModels.disabled = false;
    el.rerankRefreshModels.disabled = false;
    el.refreshModels.textContent = "刷新模型列表";
    renderEmbeddingStatuses();
    renderActions();
  }
}

async function refreshRerankModels() {
  if (refreshing) return;
  const button = el.rerankRefreshModels;
  if (/[•]|\*\*\*|已配置/.test(el.rerankApiKey.value)) {
    message("rerank", "请使用真实密钥或留空使用已保存密钥，不能使用脱敏占位符", true);
    return;
  }
  refreshing = true;
  button.disabled = true;
  el.refreshModels.disabled = true;
  button.textContent = "获取中…";
  message("rerank", "正在用当前地址和密钥获取重排模型；不会保存配置。");
  try {
    const result = await pickers.rerank.load({
      baseUrl: el.rerankUrl.value,
      apiKey: el.rerankApiKey.value,
    });
    message("rerank", result.error
      ? `获取重排模型失败：${result.error}`
      : `已获取 ${result.count} 个重排模型`, Boolean(result.error));
  } catch (error) {
    message("rerank", `获取重排模型失败：${error.message}`, true);
  } finally {
    refreshing = false;
    button.disabled = false;
    el.refreshModels.disabled = false;
    button.textContent = "获取重排模型";
    renderActions();
  }
}

export default {
  name: "settings",
  mount() {
    if (mounted) return;
    mounted = true;
    el.form.addEventListener("invalid", event => { event.target.closest("details").open = true; }, true);
    el.form.addEventListener("submit", event => { event.preventDefault(); save("all"); });
    el.form.addEventListener("input", event => { renderEmbeddingStatuses(); renderActions(event.target.closest("fieldset")?.id.replace("config-section-", "")); });
    el.form.addEventListener("change", event => { toggleAsrFields(); renderActions(event.target.closest("fieldset")?.id.replace("config-section-", "")); });
    for (const name of Object.keys(sections)) {
      actions[name].save.addEventListener("click", () => save(name));
      actions[name].reset.addEventListener("click", () => discard(name));
    }
    el.reset.addEventListener("click", () => discard("all"));
    el.refreshModels.addEventListener("click", refreshModels);
    el.rerankRefreshModels.addEventListener("click", refreshRerankModels);
    document.querySelector("#config-rerank-test").addEventListener("click", async event => {
      const button = event.currentTarget;
      if (Object.keys(changesFor("rerank")).length) { message("rerank", "请先保存或放弃本节修改，再测试已保存连接", true); return; }
      button.disabled = true;
      message("rerank", "正在发送内置样例测试…");
      try {
        const result = await requestJson("/api/rerank/test", { method: "POST" });
        message("rerank", `${result.status === "completed" ? "连接成功" : "连接失败"} · ${result.elapsed_ms} ms${result.error_message ? ` · ${result.error_message}` : ""}${result.provider_request_id ? ` · 请求 ${result.provider_request_id}` : ""}`, result.status !== "completed");
      } catch (error) { message("rerank", `测试失败：${error.message}`, true); }
      finally { button.disabled = false; }
    });
    el.mineruClear.addEventListener("click", () => {
      configState.clearMineruToken = true;
      el.mineruToken.value = "";
      el.mineruToken.placeholder = "保存本节后将清除 Token";
      renderActions("mineru");
    });
    el.mineruToken.addEventListener("input", () => {
      if (el.mineruToken.value.trim()) configState.clearMineruToken = false;
      renderActions("mineru");
    });
    for (const target of ["llm", "embedding", "rerank"]) {
      pickers[target].select.addEventListener("change", () => {
        pickers[target].syncFromSelect();
        renderEmbeddingStatuses(); renderActions(target);
      });
    }
    on("status", status => {
      if (status && configState.loaded) {
        configState.loaded.index_embedding_model = status.index_embedding_model;
        renderEmbeddingStatuses();
      }
      renderIndexStatus();
    });
  },
  show() { renderIndexStatus(); loadConfig(); },
};
