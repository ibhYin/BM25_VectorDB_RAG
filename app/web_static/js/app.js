// Bootstrap: register the views, wire the shell, start the router.

import {
  el,
  emit,
  loadIdentity,
  refreshDocuments,
  router,
  setupBackTop,
  startServerStatsPolling,
} from "./core.js";
import ask from "./views/ask.js";
import audit from "./views/audit.js";
import evaluation from "./views/eval.js";
import library from "./views/library.js";
import settings from "./views/settings.js";
import users from "./views/users.js";

router.register("ask", ask);
router.register("library", library);
router.register("settings", settings);
router.register("audit", audit);
router.register("eval", evaluation);
router.register("users", users);

// --- shell controls -------------------------------------------------------

el.refreshButton.addEventListener("click", () => {
  emit("staging:refresh");
  refreshDocuments();
});

el.userButton.addEventListener("click", () => {
  const open = el.userMenu.hidden;
  el.userMenu.hidden = !open;
  el.userButton.setAttribute("aria-expanded", String(open));
});

document.addEventListener("click", (event) => {
  if (!el.userSwitcher.hidden && !el.userSwitcher.contains(event.target)) {
    el.userMenu.hidden = true;
    el.userButton.setAttribute("aria-expanded", "false");
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !el.userMenu.hidden) {
    el.userMenu.hidden = true;
    el.userButton.setAttribute("aria-expanded", "false");
  }
});

// --- boot -----------------------------------------------------------------

setupBackTop();
router.start("ask");
refreshDocuments();
loadIdentity();
startServerStatsPolling();
