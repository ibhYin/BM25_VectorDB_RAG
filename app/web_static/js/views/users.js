// 用户管理页：开发人员建号、改密、分配所属组、授予/收回管理员权限、删除账号。
// 后端是唯一防线；这里只在体验上提前禁用（删自己/移除最后一位管理员）。

import {
  clearListLoading,
  renderListError,
  requestJson,
  setListLoading,
  showToast,
  state,
} from "../core.js";

const el = {
  form: document.querySelector("#users-create-form"),
  username: document.querySelector("#users-create-username"),
  password: document.querySelector("#users-create-password"),
  createGroups: document.querySelector("#users-create-groups"),
  admin: document.querySelector("#users-create-admin"),
  createStatus: document.querySelector("#users-create-status"),
  list: document.querySelector("#users-list"),
};

let mounted = false;
let users = [];
let editing = null; // 正在内联编辑的用户名

function availableGroups() {
  return [...state.access.groups].sort();
}

function badge(text, kind) {
  const span = document.createElement("span");
  span.className = `users-badge ${kind}`;
  span.textContent = text;
  return span;
}

function button(text, className) {
  const element = document.createElement("button");
  element.type = "button";
  element.className = className;
  element.textContent = text;
  return element;
}

function renderCreateGroups() {
  el.createGroups.replaceChildren();
  const groups = availableGroups();
  if (!groups.length) {
    const hint = document.createElement("span");
    hint.className = "config-hint";
    hint.textContent = "暂无可用组（public 自动对所有用户开放）";
    el.createGroups.append(hint);
    return;
  }
  for (const group of groups) {
    const label = document.createElement("label");
    label.className = "users-check";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = group;
    label.append(box, document.createTextNode(group));
    el.createGroups.append(label);
  }
}

function renderList() {
  el.list.replaceChildren();
  if (!users.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无账号";
    el.list.append(empty);
    return;
  }
  const adminNames = users.filter((user) => user.is_admin).map((user) => user.username);
  for (const user of users) {
    el.list.append(createRow(user, adminNames));
    if (editing === user.username) el.list.append(createEditForm(user, adminNames));
  }
}

function createRow(user, adminNames) {
  const isSelf = user.username === state.identity.user;
  const lastAdmin = user.is_admin && adminNames.length === 1;

  const row = document.createElement("article");
  row.className = "users-row";

  const main = document.createElement("div");
  main.className = "users-main";

  const nameRow = document.createElement("div");
  nameRow.className = "users-name-row";

  const name = document.createElement("strong");
  name.className = "users-name";
  name.textContent = user.username;

  const badges = document.createElement("div");
  badges.className = "users-badges";
  if (isSelf) badges.append(badge("当前账号", "self"));
  if (user.is_admin) badges.append(badge("管理员", "admin"));
  if (!user.has_password) badges.append(badge("未设密码", "no-login"));
  if (!user.in_policy) badges.append(badge("未纳入策略", "warn"));
  nameRow.append(name, badges);

  const groups = document.createElement("div");
  groups.className = "users-groups";
  groups.append(badge("所属组", "label"));
  const groupNames = (user.groups || []).length ? [...user.groups].sort() : ["仅公开"];
  for (const group of groupNames) groups.append(badge(group, "group"));

  main.append(nameRow, groups);

  const actions = document.createElement("div");
  actions.className = "users-actions";
  if (editing !== user.username) {
    const editButton = button("编辑", "ghost-button");
    editButton.addEventListener("click", () => {
      editing = user.username;
      renderList();
    });
    const deleteButton = button("删除", "ghost-button danger");
    deleteButton.disabled = isSelf || lastAdmin;
    deleteButton.title = isSelf
      ? "不能删除自己的账号"
      : lastAdmin
        ? "至少保留一位管理员"
        : "";
    deleteButton.addEventListener("click", () => confirmDelete(user));
    actions.append(editButton, deleteButton);
  }
  row.append(main, actions);
  return row;
}

function createEditForm(user, adminNames) {
  const isSelf = user.username === state.identity.user;
  const lastAdmin = user.is_admin && adminNames.length === 1;

  const form = document.createElement("form");
  form.className = "users-edit";
  form.noValidate = true;

  const groupsField = document.createElement("div");
  groupsField.className = "users-field";
  const groupsTitle = document.createElement("span");
  groupsTitle.textContent = "所属组";
  const groupsBox = document.createElement("div");
  groupsBox.className = "users-groups";
  const unionGroups = [...new Set([...availableGroups(), ...(user.groups || [])])].sort();
  for (const group of unionGroups) {
    const label = document.createElement("label");
    label.className = "users-check";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = group;
    box.checked = (user.groups || []).includes(group);
    label.append(box, document.createTextNode(group));
    groupsBox.append(label);
  }
  groupsField.append(groupsTitle, groupsBox);

  const adminField = document.createElement("label");
  adminField.className = "users-toggle";
  const adminBox = document.createElement("input");
  adminBox.type = "checkbox";
  adminBox.checked = user.is_admin;
  if (isSelf || lastAdmin) adminBox.disabled = true;
  const adminText = document.createElement("span");
  adminText.textContent = lastAdmin
    ? "管理员（至少保留一位，不可移除）"
    : isSelf
      ? "管理员（不能取消自己的管理员权限）"
      : "管理员（可检索全部文档并管理账号）";
  adminField.append(adminBox, adminText);

  const passwordField = document.createElement("label");
  passwordField.className = "users-field";
  const passwordTitle = document.createElement("span");
  passwordTitle.textContent = "新密码";
  const passwordInput = document.createElement("input");
  passwordInput.type = "password";
  passwordInput.autocomplete = "new-password";
  passwordInput.spellcheck = false;
  passwordInput.placeholder = "留空保持不变";
  passwordField.append(passwordTitle, passwordInput);

  const clearField = document.createElement("label");
  clearField.className = "users-toggle";
  const clearBox = document.createElement("input");
  clearBox.type = "checkbox";
  const clearText = document.createElement("span");
  clearText.textContent = "清除密码（该账号将无法登录）";
  clearField.append(clearBox, clearText);

  const actions = document.createElement("div");
  actions.className = "config-input-row";
  const saveButton = button("保存", "primary-button");
  saveButton.type = "submit";
  const cancelButton = button("取消", "ghost-button");
  cancelButton.addEventListener("click", () => {
    editing = null;
    renderList();
  });
  const status = document.createElement("span");
  status.className = "config-status";
  actions.append(saveButton, cancelButton, status);

  form.append(groupsField, adminField, passwordField, clearField, actions);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = {
      groups: [...groupsBox.querySelectorAll("input:checked")].map((box) => box.value),
      is_admin: adminBox.checked,
      clear_password: clearBox.checked,
    };
    if (passwordInput.value && !clearBox.checked) payload.password = passwordInput.value;
    status.textContent = "保存中…";
    try {
      await requestJson(`/api/users/${encodeURIComponent(user.username)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      editing = null;
      status.textContent = "";
      showToast(`已更新 ${user.username}`, "ok");
      await loadUsers();
    } catch (error) {
      status.textContent = "";
      showToast(error.message, "error");
    }
  });
  return form;
}

function confirmDelete(user) {
  const confirmed = confirm(`确定删除账号 ${user.username}？该账号将无法登录，其历史对话仍保留在服务器上。`);
  if (!confirmed) return;
  (async () => {
    try {
      await requestJson(`/api/users/${encodeURIComponent(user.username)}`, { method: "DELETE" });
      showToast(`已删除 ${user.username}`, "ok");
      await loadUsers();
    } catch (error) {
      showToast(error.message, "error");
    }
  })();
}

async function loadUsers() {
  setListLoading(el.list, "加载中");
  try {
    const payload = await requestJson("/api/users");
    users = payload.users || [];
    renderList();
    clearListLoading(el.list);
  } catch (error) {
    users = [];
    renderListError(el.list, error.message);
  }
}

function onSubmitCreate(event) {
  event.preventDefault();
  const username = el.username.value.trim();
  const payload = {
    username,
    password: el.password.value,
    groups: [...el.createGroups.querySelectorAll("input:checked")].map((box) => box.value),
    is_admin: el.admin.checked,
  };
  el.createStatus.textContent = "创建中…";
  (async () => {
    try {
      await requestJson("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      el.createStatus.textContent = "";
      el.form.reset();
      showToast(`已创建 ${username}`, "ok");
      await loadUsers();
    } catch (error) {
      el.createStatus.textContent = "";
      showToast(error.message, "error");
    }
  })();
}

export default {
  name: "users",

  mount() {
    if (mounted) return;
    mounted = true;
    el.form.addEventListener("submit", onSubmitCreate);
  },

  show() {
    renderCreateGroups();
    loadUsers();
  },

  onIdentity() {
    // 切到非管理员时 core 会踢回问答页；这里清掉本地状态。
    if (!state.identity.user && !state.identity.isAdmin) users = [];
  },

  onDocuments() {
    // 组列表随策略变化（如文档改组），回到页面时重新渲染组复选框。
    renderCreateGroups();
  },
};
