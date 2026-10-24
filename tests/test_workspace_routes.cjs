// Run: node tests/test_workspace_routes.cjs. Real router and registrations, no browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
let hashchange;
let shown;
const sections = new Map();
const context = vm.createContext({
  location: { hash: '#/aggregate' },
  window: { addEventListener(type, handler) { assert.equal(type, 'hashchange'); hashchange = handler; } },
  document: {
    getElementById(id) { if (!sections.has(id)) sections.set(id, { hidden: true }); return sections.get(id); },
    querySelectorAll() { return []; },
  },
  queueMicrotask,
  ask: { show(params) { shown = params; } },
  audit: {}, evaluation: {}, library: {}, settings: {}, users: {},
});
const core = fs.readFileSync('app/web_static/js/core.js', 'utf8');
vm.runInContext(core.slice(core.indexOf('const views ='), core.indexOf('// --- identity')).replace('export const router', 'const router'), context);
const app = fs.readFileSync('app/web_static/js/app.js', 'utf8');
vm.runInContext(app.slice(app.indexOf('router.register('), app.indexOf('document.querySelector("#warning-rebuild")')), context);
(async () => {
  vm.runInContext('router.start("ask")', context);
  await Promise.resolve();
  assert.equal(context.location.hash, '#/ask');
  hashchange();
  assert.equal(shown.mode, 'aggregate', 'old aggregate bookmarks must open the statistics mode');
  assert.equal(sections.get('page-ask').hidden, false);
  vm.runInContext('router.navigate("ask", {mode: "ask"})', context);
  assert.equal(shown.mode, 'ask');
  console.log('PASS: legacy aggregate links redirect into the shared workspace with the correct mode.');
})().catch(error => { console.error(error); process.exitCode = 1; });
