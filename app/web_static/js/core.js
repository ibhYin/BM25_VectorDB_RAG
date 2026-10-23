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
  serverStats: document.querySelector("#server-stats"),
  refreshButton: document.querySelector("#refresh-button"),
  indexWarning: document.querySelector("#index-warning"),
  navAudit: document.querySelector("#nav-audit"),
  navEval: document.querySelector("#nav-eval"),
  navSettings: document.querySelector("#nav-settings"),
  navUsers: document.querySelector("#nav-users"),
  navLibraryBadge: document.querySelector("#nav-library-badge"),
  userSwitcher: document.querySelector("#user-switcher"),
  userButton: document.querySelector("#user-button"),
  userLabel: document.querySelector("#user-label"),
  userMenu: document.querySelector("#user-menu"),
  toast: document.querySelector("#toast"),
  backTop: document.querySelector("#back-top"),
};

// --- back to top -----------------------------------------------------------
// 全局浮动「回到顶部」：任何页面滚动超过阈值就出现。滚动容器不固定（window 或
// 当前 .page），所以 window 和所有 .page 都监听，取最大滚动量。

const BACK_TOP_THRESHOLD = 400;

function backTopVisible() {
  let y = window.scrollY || 0;
  for (const page of document.querySelectorAll(".page")) {
    if (page.scrollTop > y) y = page.scrollTop;
  }
  return y >= BACK_TOP_THRESHOLD;
}

export function setupBackTop() {
  const update = () => {
    el.backTop.hidden = !backTopVisible();
  };
  window.addEventListener("scroll", update, { passive: true });
  for (const page of document.querySelectorAll(".page")) {
    page.addEventListener("scroll", update, { passive: true });
  }
  el.backTop.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
    for (const page of document.querySelectorAll(".page")) {
      page.scrollTo({ top: 0, behavior: "smooth" });
    }
  });
  update();
}

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
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value / 1024;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(amount >= 100 ? 0 : 1)} ${units[unit]}`;
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
  renderIndexWarning(state.status);
  emit("identity", state.identity);
  notifyViews("onIdentity", state.identity);
}

// 仅管理员可见的页面。加新页时只改这一处——此前导航显隐和「离开无权页面」是
// 两份各自维护的名单，加页时很容易只改一个。
const ADMIN_PAGES = ["settings", "audit", "eval", "users"];

function renderNavGating() {
  const allowed = canConfigure();
  const links = {
    settings: el.navSettings,
    audit: el.navAudit,
    eval: el.navEval,
    users: el.navUsers,
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
  document.querySelector("#index-warning-text").textContent =
    `向量模型已变更：索引由 ${status.index_embedding_model} 建立，` +
    `已保存目标为 ${status.embedding_model}。向量/混合检索已暂停；可查看文档。重建成功后切换，失败保留旧索引。`;
  document.querySelector("#warning-rebuild").hidden = !state.identity.isAdmin;
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

// --- server resources -----------------------------------------------------

const SERVER_STATS_POLL_MS = 5000;

export async function refreshServerStats() {
  let stats;
  try {
    stats = await requestJson("/api/system-stats");
  } catch (_) {
    el.serverStats.textContent = "资源不可用";
    el.serverStats.classList.remove("ok", "warn");
    return;
  }
  const memory = stats.memory || {};
  const disk = stats.disk || {};
  const memPercent = memory.percent;
  const diskPercent = disk.percent;
  const load = stats.cpu_load;

  const parts = [];
  parts.push(`CPU ${load === null || load === undefined ? "?" : load.toFixed(2)}`);
  parts.push(`内存 ${memPercent === null || memPercent === undefined ? "?" : `${memPercent}%`}`);
  parts.push(`盘 ${diskPercent === null || diskPercent === undefined ? "?" : `${diskPercent}%`}`);
  el.serverStats.textContent = parts.join(" · ");

  // tooltip：完整明细
  const cores = stats.cpu_cores ?? "?";
  const detail = [
    `CPU 负载 ${load === null || load === undefined ? "未知" : load.toFixed(2)} / ${cores} 核`,
    memPercent === null || memPercent === undefined
      ? "内存 未知"
      : `内存 ${formatBytes(memory.used)} / ${formatBytes(memory.total)}（可用 ${formatBytes(memory.available)}）`,
    diskPercent === null || diskPercent === undefined
      ? "磁盘 未知"
      : `磁盘 ${formatBytes(disk.used)} / ${formatBytes(disk.total)}`,
    stats.process_rss ? `Web 进程 RSS ${formatBytes(stats.process_rss)}` : null,
    stats.embedding_online === true
      ? "向量服务 在线"
      : stats.embedding_online === false
        ? "向量服务 离线"
        : "向量服务 未知",
  ].filter(Boolean);
  el.serverStats.title = detail.join("\n");

  const warned =
    (memPercent !== null && memPercent !== undefined && memPercent >= 90) ||
    (diskPercent !== null && diskPercent !== undefined && diskPercent >= 90) ||
    stats.embedding_online === false;
  el.serverStats.classList.toggle("warn", warned);
  el.serverStats.classList.toggle("ok", !warned && load !== null && load !== undefined);
}

export function startServerStatsPolling() {
  refreshServerStats();
  setInterval(refreshServerStats, SERVER_STATS_POLL_MS);
}

export function shortDocLabel(docId) {
  const LIMIT = 12;
  const found = state.documents.find((item) => item.doc_id === docId);
  const name = (found && found.title) || docId;
  return name.length > LIMIT ? `${name.slice(0, LIMIT)}…` : name;
}

let rebuildSubmitting = false;
export async function requestIndexRebuild(force = false) {
  const dialog = document.querySelector("#rebuild-dialog");
  if (rebuildSubmitting || dialog.open) return;
  rebuildSubmitting = true;
  const buttons = ["#warning-rebuild", "#rebuild-index"].map(id => document.querySelector(id));
  const title = document.querySelector("#rebuild-dialog-title");
  const message = document.querySelector("#rebuild-dialog-message");
  const close = document.querySelector("#rebuild-dialog-close");
  const confirmButton = document.querySelector("#rebuild-dialog-confirm");
  const trigger = document.activeElement;
  let submitting = false;
  close.onclick = () => dialog.close();
  dialog.oncancel = event => { if (submitting) event.preventDefault(); };
  dialog.onclose = () => { if (trigger?.isConnected) trigger.focus(); };
  const report = (heading, text, busy = false) => {
    title.textContent = heading;
    message.textContent = text;
    message.setAttribute("aria-busy", String(busy));
  };
  close.disabled = false;
  close.textContent = "取消";
  confirmButton.hidden = true;
  buttons.forEach(button => { button.disabled = true; });
  report("检查重建条件", "正在检查当前文档与缓存…", true);
  dialog.showModal();
  try {
    const plan = await requestJson(`/api/index-rebuild/preview?force=${force ? 1 : 0}`);
    if (!dialog.open) return;
    report("确认重建索引", `知识库：${plan.knowledge_base_id}，共 ${plan.documents} 份已入库文档。\n原索引模型：${plan.original_model}\n目标模型：${plan.target_model}\n\n${force ? "已选择强制重新解析原文件。" : "默认复用切片正文或完整解析缓存。"}\n复用切片 ${plan.reuse_chunks} 份，缓存重切 ${plan.resplit} 份，读取原文件解析 ${plan.parse} 份。\n\n完成后切换；失败保留旧索引。\n重建期间暂停上传、删除和配置修改。`);
    confirmButton.hidden = false;
    const confirmed = await new Promise(resolve => {
      const finish = value => {
        confirmButton.removeEventListener("click", accept);
        dialog.removeEventListener("close", cancel);
        resolve(value);
      };
      const accept = () => finish(true);
      const cancel = () => finish(false);
      confirmButton.addEventListener("click", accept);
      dialog.addEventListener("close", cancel);
    });
    if (!confirmed) return;
    submitting = true;
    confirmButton.hidden = true;
    close.disabled = true;
    report("提交重建任务", "正在提交，请稍候…", true);
    const job = await requestJson("/api/index-rebuild", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan_id: plan.plan_id, force_reparse: force }),
    });
    report("重建任务已提交", "可在知识库的处理任务中查看阶段、进度与结果；刷新页面后仍可查看。任务完成后才会切换索引。");
    state.jobs.unshift(job);
    router.navigate("library");
    emit("jobs:refresh");
  } catch (error) {
    if (dialog.open) report("无法确认重建任务", `${error.message}\n请先查看处理任务；若没有创建任务，关闭后可重新发起。`);
  }
  finally {
    submitting = false;
    close.disabled = false;
    close.textContent = "关闭";
    if (dialog.open) close.focus();
    rebuildSubmitting = false;
    const active = state.jobs.some(job => job.kind === "rebuild" && ["queued", "indexing"].includes(job.status));
    buttons.forEach(button => { button.disabled = active; });
  }
}
