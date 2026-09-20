// Shared shell: application state, the hash router, the identity/status
// pipeline, and small DOM helpers every view uses.

export const state = {
  status: null,
  documents: [],
  access: { enabled: false, manageable: false, groups: [] },
  identity: { user: null, users: [], isAdmin: false },
  jobs: [],
  polling: null,
  retrievalInitialized: false,
};

export const el = {
  serviceSummary: document.querySelector("#service-summary"),
  vectorStatus: document.querySelector("#vector-status"),
  modelStatus: document.querySelector("#model-status"),
  refreshButton: document.querySelector("#refresh-button"),
  indexWarning: document.querySelector("#index-warning"),
  navAudit: document.querySelector("#nav-audit"),
  navEval: document.querySelector("#nav-eval"),
  navSettings: document.querySelector("#nav-settings"),
  navLibraryBadge: document.querySelector("#nav-library-badge"),
  userSwitcher: document.querySelector("#user-switcher"),
  userButton: document.querySelector("#user-button"),
  userLabel: document.querySelector("#user-label"),
  userMenu: document.querySelector("#user-menu"),
  toast: document.querySelector("#toast"),
};

// --- tiny event bus -------------------------------------------------------
// Views subscribe to state changes instead of core reaching into their DOM.

const listeners = new Map();

export function on(event, handler) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(handler);
}

export function emit(event, payload) {
  for (const handler of listeners.get(event) || []) handler(payload);
}

/**
 * 把事件派发给**已挂载**视图的同名钩子。
 *
 * 路由只负责 mount / show / hide；onStatus / onIdentity / onDocuments 这类钩子是给
 * 事件用的。此前漏了这一环，于是各视图都实现过、却从没人调用——等于死代码：
 * 问答页「按服务端默认值回填检索参数」、知识库页「文档变了重渲染」、审计与评估页
 * 「切换身份时清掉本地状态」都一直没生效。
 */
function notifyViews(hook, payload) {
  for (const [name, view] of views) {
    if (mounted.has(name)) view[hook]?.(payload);
  }
}

// --- DOM helpers ----------------------------------------------------------

export function createSpinner() {
  const spinner = document.createElement("span");
  spinner.className = "loading-spinner";
  spinner.setAttribute("aria-hidden", "true");
  return spinner;
}

export function setListLoading(element, message) {
  element.replaceChildren();
  element.setAttribute("aria-busy", "true");
  const loading = document.createElement("div");
  loading.className = "loading-state";
  loading.append(createSpinner(), document.createTextNode(message));
  element.append(loading);
}

export function clearListLoading(element) {
  element.removeAttribute("aria-busy");
}

export function renderListError(element, message) {
  element.replaceChildren();
  clearListLoading(element);
  const error = document.createElement("div");
  error.className = "empty-list error";
  error.textContent = message;
  element.append(error);
}

export function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

export function formatTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value || "")
    : date.toLocaleString("zh-CN", { hour12: false });
}

let toastTimer = null;

export function showToast(message, variant = "ok") {
  el.toast.textContent = message;
  el.toast.className = `toast ${variant}`;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.toast.hidden = true;
  }, 2800);
}

// --- API ------------------------------------------------------------------

export async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("Content-Type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : { error: await response.text() };

  if (response.status === 401 && !url.startsWith("/api/login")) {
    location.replace("/login");
    throw new Error("请重新登录");
  }
  if (!response.ok) {
    throw new Error(payload.error || "请求失败");
  }
  return payload;
}

// --- router ---------------------------------------------------------------

const views = new Map();
const mounted = new Set();
let currentName = "";
let pendingParams = null;

function nameFromHash() {
  return location.hash.replace(/^#\/?/, "").split("?")[0].trim();
}

function activate(name, params) {
  const view = views.get(name);
  if (!view) return false;

  if (currentName !== name) {
    if (currentName) views.get(currentName)?.hide?.();
    currentName = name;
    if (!mounted.has(name)) {
      mounted.add(name);
      view.mount?.();
    }
    for (const key of views.keys()) {
      const section = document.getElementById(`page-${key}`);
      if (section) section.hidden = key !== name;
    }
    for (const link of document.querySelectorAll(".nav-link")) {
      const active = link.dataset.route === name;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
  }
  view.show?.(params);
  return true;
}

export const router = {
  register(name, view) {
    views.set(name, view);
  },

  // Params are stashed rather than encoded into the hash: chunk ids contain "#"
  // and would otherwise be ambiguous.
  navigate(name, params) {
    pendingParams = params || null;
    if (nameFromHash() === name) {
      activate(name, pendingParams);
      pendingParams = null;
      return;
    }
    location.hash = `#/${name}`;
  },

  current: () => currentName,

  start(defaultName) {
    const go = () => {
      const name = nameFromHash() || defaultName;
      if (!activate(name, pendingParams)) {
        location.hash = `#/${defaultName}`;
        return;
      }
      pendingParams = null;
    };
    window.addEventListener("hashchange", go);
    go();
  },
};

// --- identity -------------------------------------------------------------

// Open mode (no policy configured) behaves like an admin; a logged-in
// non-admin gets the read-only experience.
export function canConfigure() {
  return !state.identity.user || state.identity.isAdmin;
}

export function uploadsToStaging() {
  return Boolean(state.identity.user) && !state.identity.isAdmin;
}

export async function loadIdentity() {
  try {
    const payload = await requestJson("/api/me");
    state.identity = {
      user: payload.user,
      users: payload.users || [],
      isAdmin: Boolean(payload.is_admin),
    };
  } catch (_) {
    state.identity = { user: null, users: [], isAdmin: false };
  }
  renderUserSwitcher();
  renderNavGating();
  emit("identity", state.identity);
  notifyViews("onIdentity", state.identity);
}

// 仅管理员可见的页面。加新页时只改这一处——此前导航显隐和「离开无权页面」是
// 两份各自维护的名单，加页时很容易只改一个。
const ADMIN_PAGES = ["settings", "audit", "eval"];

function renderNavGating() {
  const allowed = canConfigure();
  const links = {
    settings: el.navSettings,
    audit: el.navAudit,
    eval: el.navEval,
  };
  for (const name of ADMIN_PAGES) {
    const link = links[name];
    if (link) link.hidden = !allowed;
  }
  // Leaving a now-forbidden page (user switched to non-admin).
  if (!allowed && ADMIN_PAGES.includes(currentName)) {
    router.navigate("ask");
  }
}

function renderUserSwitcher() {
  const { user, users } = state.identity;
  el.userSwitcher.hidden = !user;
  if (!user) return;
  el.userLabel.textContent = user;
  el.userMenu.replaceChildren();
  const title = document.createElement("div");
  title.className = "menu-title";
  title.textContent = "当前账号";
  el.userMenu.append(title);

  const current = users.find((item) => item.id === user) || { id: user, groups: [] };
  const identity = document.createElement("div");
  identity.className = "menu-item current";
  identity.setAttribute("role", "menuitem");
  const name = document.createElement("span");
  name.textContent = `✓ ${current.id}`;
  const groups = document.createElement("span");
  groups.className = "groups";
  groups.textContent = (current.groups || []).join(", ") || "无组";
  identity.append(name, groups);
  el.userMenu.append(identity);

  const divider = document.createElement("div");
  divider.className = "menu-divider";
  el.userMenu.append(divider);

  const logout = document.createElement("button");
  logout.type = "button";
  logout.className = "menu-item logout";
  logout.setAttribute("role", "menuitem");
  logout.textContent = "退出 / 切换账号";
  logout.addEventListener("click", async () => {
    logout.disabled = true;
    el.userMenu.hidden = true;
    el.userButton.setAttribute("aria-expanded", "false");
    await fetch("/api/logout", { method: "POST" });
    location.replace("/login");
  });
  el.userMenu.append(logout);
}

// --- status ---------------------------------------------------------------

function setStatusLoading(loading) {
  for (const element of [el.vectorStatus, el.modelStatus]) {
    element.classList.toggle("loading", loading);
    if (loading) {
      element.replaceChildren(createSpinner(), document.createTextNode("加载中"));
    }
  }
  if (loading) el.serviceSummary.textContent = "正在读取服务状态...";
}

// Shell-only: model chips + the embedding-mismatch banner. Everything tied to
// the document list (counts, retrieval defaults) belongs to its own view.
export async function refreshStatus() {
  el.refreshButton.disabled = true;
  setStatusLoading(true);
  try {
    const status = await requestJson("/api/status");
    state.status = status;
    el.serviceSummary.textContent = `${status.llm_model} · ${status.embedding_model}`;
    el.vectorStatus.textContent = status.chroma_exists ? "向量库可用" : "向量库未创建";
    el.vectorStatus.classList.toggle("ok", status.chroma_exists);
    el.modelStatus.textContent = status.has_openai_api_key ? "LLM 已配置" : "LLM 未配置";
    el.modelStatus.classList.toggle("ok", status.has_openai_api_key);
    renderIndexWarning(status);
    emit("status", status);
    notifyViews("onStatus", status);
  } catch (_) {
    state.status = null;
    el.serviceSummary.textContent = "服务连接失败";
    el.vectorStatus.textContent = "服务不可用";
    el.modelStatus.textContent = "服务不可用";
    el.indexWarning.hidden = true;
    emit("status", null);
    notifyViews("onStatus", null);
  } finally {
    setStatusLoading(false);
    el.refreshButton.disabled = false;
  }
}

// A changed embedding model means stored vectors no longer match the config, so
// the warning stays visible until the corpus is reindexed.
export function renderIndexWarning(status) {
  if (!status || !status.embedding_model_mismatch || !status.index_embedding_model) {
    el.indexWarning.hidden = true;
    return;
  }
  el.indexWarning.textContent =
    `向量模型已变更：索引由 ${status.index_embedding_model} 建立，` +
    `当前配置为 ${status.embedding_model}。检索结果可能失效，请重新入库文档。`;
  el.indexWarning.hidden = false;
}

export async function refreshDocuments() {
  try {
    const payload = await requestJson("/api/documents");
    state.documents = payload.documents || [];
    state.access = payload.access || { enabled: false, manageable: false, groups: [] };
    emit("documents", state.documents);
    notifyViews("onDocuments", state.documents);
  } catch (error) {
    emit("documents:error", error.message);
  }
  await refreshStatus();
}

export function shortDocLabel(docId) {
  const LIMIT = 12;
  const found = state.documents.find((item) => item.doc_id === docId);
  const name = (found && found.title) || docId;
  return name.length > LIMIT ? `${name.slice(0, LIMIT)}…` : name;
}
