// One editable combobox; the native select remains the complete source of truth.
const controls = new WeakMap();
let nextId = 0;

export function resetSelectSearch(select) { controls.get(select)?.reset(); }

export function enhanceSelect(select, { input = null, allowCustom = false, customValue = "__custom__" } = {}) {
  if (controls.has(select)) return controls.get(select);
  const label = (select.getAttribute("aria-label") || select.labels?.[0]?.querySelector("span")?.textContent || "选项").replace(/\s+/g, " ").trim();
  const wrapper = document.createElement("div");
  wrapper.className = "search-select";
  input ||= document.createElement("input");
  input.type = "text";
  input.classList.add("search-select-input");
  input.placeholder = `输入以搜索${label}${allowCustom ? "；自定义按 Enter 确认" : ""}`;
  input.autocomplete = "off";
  input.spellcheck = false;
  input.hidden = false;
  input.setAttribute("role", "combobox");
  input.setAttribute("aria-label", label);
  input.setAttribute("aria-autocomplete", "list");
  input.setAttribute("aria-expanded", "false");
  if (select.required) select.setAttribute("aria-required", "true");
  select.required = false;
  select.classList.add("search-select-source");
  select.setAttribute("aria-hidden", "true");
  select.tabIndex = -1;
  const list = document.createElement("div");
  list.id = `select-options-${++nextId}`;
  list.className = "search-select-options";
  list.setAttribute("role", "listbox");
  list.setAttribute("aria-label", `${label}选项`);
  list.hidden = true;
  input.setAttribute("aria-controls", list.id);
  select.insertAdjacentElement("beforebegin", wrapper);
  wrapper.append(input, select, list);
  let query = "";
  let candidates = [];
  let active = -1;
  let committed = "";
  const display = () => {
    const selected = [...select.options].find(option => option.value === select.value);
    return allowCustom ? committed : (selected?.textContent || "");
  };
  function close() {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
  }
  function highlight() {
    [...list.children].forEach((node, index) => node.classList.toggle("active", index === active));
    const node = active >= 0 ? list.children[active] : null;
    if (node) {
      input.setAttribute("aria-activedescendant", node.id);
      node.scrollIntoView?.({ block: "nearest" });
    } else input.removeAttribute("aria-activedescendant");
  }
  function commit(candidate = null) {
    const model = candidate?.value ?? (allowCustom ? input.value.trim() : null);
    if (model === null || (allowCustom && !model)) { control.reset(); return; }
    const option = [...select.options].find(item => item.value === model && !item.disabled);
    if (!option && !allowCustom) return;
    select.value = option ? option.value : customValue;
    const changed = model !== committed;
    committed = model;
    input.value = display();
    query = "";
    close();
    if (changed) select.dispatchEvent(new Event("change", { bubbles: true }));
  }
  function render() {
    const words = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    // ponytail: current options are flat; preserve optgroup structure if groups are added later.
    candidates = [...select.options].filter(option => (!allowCustom || option.value !== customValue) && !option.disabled && words.every(word => `${option.textContent} ${option.value}`.toLocaleLowerCase().includes(word)))
      .map(option => ({ value: option.value, label: option.textContent }));
    if (allowCustom && query.trim() && !candidates.some(option => option.value === query.trim())) {
      candidates.push({ value: query.trim(), label: `使用“${query.trim()}”（自定义）` });
    }
    list.replaceChildren();
    for (const [index, option] of candidates.entries()) {
      const row = document.createElement("div");
      row.id = `${list.id}-${index}`;
      row.setAttribute("role", "option");
      row.setAttribute("aria-selected", String(option.value === committed));
      row.textContent = option.label;
      row.addEventListener("pointerdown", event => event.preventDefault());
      row.addEventListener("click", event => {
        event.preventDefault(); // Do not let an enclosing label reopen the input.
        event.stopPropagation();
        commit(option);
        input.focus();
      });
      list.append(row);
    }
    if (!candidates.length) {
      const empty = document.createElement("div");
      empty.className = "search-select-empty";
      empty.textContent = "没有匹配的选项";
      empty.setAttribute("role", "status");
      list.append(empty);
    }
    active = query ? (candidates.length ? 0 : -1) : candidates.findIndex(option => option.value === committed);
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
    highlight();
  }
  function sync({ preserveInput = false, value } = {}) {
    wrapper.hidden = select.hidden;
    input.disabled = select.disabled;
    input.required = select.getAttribute("aria-required") === "true";
    committed = value ?? (select.value === customValue ? committed : select.value);
    if (!preserveInput) { input.value = display(); query = ""; }
    if (wrapper.hidden || input.disabled) close();
    else if (!list.hidden) render();
  }
  const control = { input, list, sync, value: () => committed, reset() { close(); sync(); } };
  controls.set(select, control);
  input.addEventListener("focus", () => { query = ""; render(); input.select(); });
  input.addEventListener("click", () => { if (list.hidden) { query = ""; render(); } });
  input.addEventListener("input", event => {
    event.stopPropagation(); // Search text is never the selected configuration value.
    query = input.value;
    if (!event.isComposing && !query.trim()) { input.value = display(); query = ""; input.select(); }
    render();
  });
  input.addEventListener("change", event => event.stopPropagation());
  input.addEventListener("blur", () => control.reset());
  input.addEventListener("keydown", event => {
    if (event.isComposing) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (list.hidden) { query = ""; render(); }
      if (candidates.length) active = active < 0 ? (event.key === "ArrowDown" ? 0 : candidates.length - 1) : (active + (event.key === "ArrowDown" ? 1 : -1) + candidates.length) % candidates.length;
      highlight();
    } else if (event.key === "Enter") {
      event.preventDefault();
      commit(!list.hidden && active >= 0 ? candidates[active] : null);
    } else if (event.key === "Escape") {
      event.preventDefault();
      control.reset();
    } else if (event.key === "Tab") control.reset();
  });
  select.addEventListener("change", () => sync());
  new MutationObserver(records => sync({ preserveInput: records.every(record => record.type === "attributes") || document.activeElement === input }))
    .observe(select, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ["hidden", "disabled"] });
  sync();
  return control;
}

export function setupSelectSearch() {
  document.querySelectorAll("select").forEach(select => enhanceSelect(select));
  new MutationObserver(records => {
    for (const record of records) for (const node of record.addedNodes) {
      if (node.nodeType !== 1) continue;
      if (node.matches("select")) enhanceSelect(node);
      node.querySelectorAll("select").forEach(select => enhanceSelect(select));
    }
  }).observe(document.body, { childList: true, subtree: true });
}
