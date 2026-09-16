// 模型选择控件：一个 <select>（选项来自端点的模型列表）+ 一个自由输入框
// （端点列不出来、或想手填时用）。设置页与问答页共用。
//
// 抽出来是因为这里有一堆容易分叉的细节：`__custom__` 哨兵值、端点是本地还是云端
// 的判定、模型列表里没有当前模型时要把它补进去、端点不可达时要退回到自由输入
// 而不是把配置卡死。复制一份迟早会两边不一样。

import { requestJson } from "./core.js";

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

  function syncCustom(prefill = "") {
    const isCustom = select.value === CUSTOM_MODEL;
    custom.hidden = !isCustom;
    if (isCustom) custom.value = prefill || custom.value || desired || "";
  }

  function renderSelect(models) {
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
      syncCustom("");
    } else {
      // 没有可挑的目录：给一个预填好当前模型的自由输入，而不是一个只有一项的列表
      select.value = CUSTOM_MODEL;
      syncCustom(desired);
    }
    if (select.value !== CUSTOM_MODEL) select.dataset.previous = select.value;
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
    const value = String(baseUrl || "").trim();
    hint.className = "config-model-hint";
    if (requireBaseUrl && !value) {
      hint.textContent = "填写请求地址后可读取模型列表";
      renderEmpty();
      return;
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
      const models = payload.models || [];
      renderSelect(models);
      if (payload.error) {
        hint.textContent = `${payload.error}；可在「自定义…」里手动输入模型名`;
        hint.classList.add("error");
      } else {
        const scopeText = payload.scope === "local" ? "本地" : "云端";
        hint.textContent = models.length
          ? `${scopeText}模型 ${models.length} 个`
          : "该端点未提供模型列表，可在「自定义…」里手动输入";
      }
      renderScope(value, payload.scope);
    } catch (error) {
      renderEmpty();
      hint.textContent = `${error.message}；可在「自定义…」里手动输入模型名`;
      hint.classList.add("error");
    }
  }

  return {
    /** 喂「当前保存的模型」，之后渲染列表时会把它标成当前值。 */
    setValue(model) {
      desired = String(model || "").trim();
    },
    value() {
      return (select.value === CUSTOM_MODEL ? custom.value : select.value).trim();
    },
    /** 用户在下拉里改了选择：切到自定义时预填上一个选中的模型并聚焦。 */
    syncFromSelect({ focus = false } = {}) {
      const isCustom = select.value === CUSTOM_MODEL;
      custom.hidden = !isCustom;
      if (isCustom) {
        custom.value = select.dataset.previous || desired || "";
        if (focus) custom.focus();
      } else {
        select.dataset.previous = select.value;
      }
    },
    renderEmpty,
    renderScope,
    load,
    select,
    custom,
  };
}

export { CUSTOM_MODEL };
