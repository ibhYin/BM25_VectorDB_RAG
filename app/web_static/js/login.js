const form = document.querySelector("#login-form");
const message = document.querySelector("#message");
const username = document.querySelector("#username");
const password = document.querySelector("#password");
const loginButton = document.querySelector("#login-button");
const toggleButton = document.querySelector("#toggle-password");
const capsLockNotice = document.querySelector("#caps-lock");
const rememberCheckbox = document.querySelector("#remember-username");
const search = new URLSearchParams(location.search);
const REMEMBER_KEY = "rag.login.username";

function setStatus(text, variant = "error") {
  message.textContent = text;
  message.classList.toggle("success", variant === "success" && Boolean(text));
}

async function checkExistingSession() {
  try {
    const response = await fetch("/api/me");
    if (response.ok) location.replace("/");
  } catch (_) {
    // Login page remains usable when this check cannot complete.
  }
}

function restoreUsername() {
  const savedUser = localStorage.getItem(REMEMBER_KEY) || "";
  const requestedUser = search.get("user");
  username.value = requestedUser || savedUser;
  rememberCheckbox.checked = Boolean(savedUser) || !requestedUser;
  if (username.value) password.focus();
  else username.focus();
}

toggleButton.addEventListener("click", () => {
  const showPassword = password.type === "password";
  password.type = showPassword ? "text" : "password";
  toggleButton.classList.toggle("show-password", showPassword);
  toggleButton.setAttribute("aria-pressed", String(showPassword));
  toggleButton.setAttribute("aria-label", showPassword ? "隐藏密码" : "显示密码");
});

password.addEventListener("keyup", (event) => {
  const capsLockOn = event.getModifierState?.("CapsLock") || false;
  capsLockNotice.hidden = !capsLockOn;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const trimmedUser = username.value.trim();

  if (!trimmedUser || !password.value) {
    setStatus("请输入用户名和密码");
    return;
  }

  loginButton.disabled = true;
  loginButton.classList.add("loading");
  setStatus("");
  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: trimmedUser,
        password: password.value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "登录失败");

    if (rememberCheckbox.checked) localStorage.setItem(REMEMBER_KEY, trimmedUser);
    else localStorage.removeItem(REMEMBER_KEY);

    setStatus("登录成功，正在进入工作台", "success");
    location.replace("/");
  } catch (error) {
    setStatus(error.message || "登录失败，请稍后再试");
    password.select();
  } finally {
    loginButton.disabled = false;
    loginButton.classList.remove("loading");
  }
});

restoreUsername();
checkExistingSession();
