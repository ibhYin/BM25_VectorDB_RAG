const form = document.querySelector("#login-form");
const message = document.querySelector("#message");
const username = document.querySelector("#username");
const password = document.querySelector("#password");
const search = new URLSearchParams(location.search);
if (search.get("user")) username.value = search.get("user");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  message.textContent = "";
  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: username.value.trim(),
        password: password.value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "登录失败");
    location.replace("/");
  } catch (error) {
    message.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
