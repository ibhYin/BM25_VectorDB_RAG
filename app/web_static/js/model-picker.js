// 设置页与问答页共用单个可搜索模型输入框，原 select 只保存完整目录。
//
// 抽出来是因为这里有一堆容易分叉的细节：`__custom__` 哨兵值、端点是本地还是云端
// 的判定、模型列表里没有当前模型时要把它补进去、端点不可达时要退回到自由输入
// 而不是把配置卡死。复制一份迟早会两边不一样。

import { requestJson } from "./core.js";
import { enhanceSelect } from "./select-search.js";

const CUSTOM_MODEL = "__custom__";

// Mirrors app/model_catalog.is_local_endpoint: loopback 是「本机自己的模型服务」
// （Ollama），其它都当作远程网关。
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

/**
 * @param requireBaseUrl 为真时，地址为空就只提示「先填地址」而不去请求。
 *   设置页要这个行为（它自己有一个地址输入框）；问答页没有地址框，用的是已保存的
 *   地址，所以要传 false，让服务端用它自己那份凭据。
 */
export function createModelPicker({
  select,
  custom,
  hint,
  scopeBadge,
  target,
  requireBaseUrl = false,
}) {
  let desired = "";
  let catalog = [];
  let requestSerial = 0;
  const control = enhanceSelect(select, { input: custom, allowCustom: true, customValue: CUSTOM_MODEL });

  function renderSelect(models, { preserveInput = false } = {}) {
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

    select.value = desired || CUSTOM_MODEL;
    control.sync({ preserveInput, value: desired });
  }

  function renderEmpty() {
    renderSelect([]);
  }

  function renderScope(baseUrl = "", responseScope = "") {
    if (!scopeBadge) return;
    const value = String(baseUrl || "").trim();
    // 问答页没有地址框，用服务端回的 scope
    const local = value ? isLocalEndpoint(value) : responseScope === "local";
    if (!value && !responseScope) {
      scopeBadge.textContent = "未配置地址";
      scopeBadge.className = "scope-badge empty";
      return;
    }
    scopeBadge.textContent = local ? "本地" : "云端";
    scopeBadge.className = `scope-badge ${local ? "local" : "cloud"}`;
  }

  async function load({ baseUrl = "", apiKey = "" } = {}) {
    const serial = ++requestSerial;
    const value = String(baseUrl || "").trim();
    hint.className = "config-model-hint";
    if (requireBaseUrl && !value) {
      hint.textContent = "填写请求地址后可读取模型列表";
      return { error: "请填写请求地址", count: 0 };
    }

    hint.textContent = "正在读取模型列表...";
    const body = { target };
    if (value) body.base_url = value;
    if (apiKey) body.api_key = apiKey;
    try {
      const payload = await requestJson("/api/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (serial !== requestSerial) return { superseded: true };
      const models = payload.models || [];
      // Refresh only the catalog; never promote an unfinished search to a selection.
      desired = control.value();
      if (!payload.error) {
        catalog = models;
        renderSelect(models, { preserveInput: document.activeElement === custom });
      }
      if (payload.error) {
        hint.textContent = `${payload.error}；仍可在模型框中直接输入模型名`;
        hint.classList.add("error");
      } else {
        const scopeText = payload.scope === "local" ? "本地" : "云端";
        hint.textContent = models.length
          ? `${scopeText}模型 ${models.length} 个`
          : "该端点未提供模型列表，可直接输入模型名";
      }
      renderScope(value, payload.scope);
      return { count: models.length, error: payload.error || "" };
    } catch (error) {
      if (serial !== requestSerial) return { superseded: true };
      hint.textContent = `${error.message}；仍可在模型框中直接输入模型名`;
      hint.classList.add("error");
      return { count: 0, error: error.message };
    }
  }

  return {
    /** 喂「当前保存的模型」，之后渲染列表时会把它标成当前值。 */
    setValue(model) {
      desired = String(model || "").trim();
      renderSelect(catalog);
    },
    value() {
      return control.value();
    },
    /** 原生 select 的变更与单个输入框保持同步。 */
    syncFromSelect() { control.sync(); },
    renderEmpty,
    renderScope,
    load,
    select,
    custom,
  };
}

export { CUSTOM_MODEL };
