// Run: node tests/test_settings_sections.cjs. Real modules, tiny DOM, no browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('app/web_static/index.html', 'utf8');
const settingsHtml = html.slice(html.indexOf('<section class="page" id="page-settings"'), html.indexOf('<!-- ============================ 审计'));
const collapsibleSections = [...settingsHtml.matchAll(/<details\b([^>]*)>([\s\S]*?)<\/details>/g)];
assert.equal(collapsibleSections.length, 5, 'all four configuration sections and index status must collapse');
for (const [index, [, attributes, content]] of collapsibleSections.entries()) {
  assert.doesNotMatch(attributes, /\bopen\b/, 'settings sections must default to collapsed');
  assert.match(content, /^\s*<summary>[\s\S]+?<\/summary>/, 'each section needs a native keyboard-accessible summary');
  if (index < 4) assert.match(content, new RegExp(`</summary>\\s*<fieldset id="config-section-${['llm', 'embedding', 'mineru', 'asr'][index]}"`));
  else assert.match(content, /索引状态/);
}
class Element {
  constructor(id = '') { this.id = id; this.value = ''; this.placeholder = ''; this.textContent = ''; this.dataset = {}; this.children = []; this.disabled = false; this.hidden = false; this.classList = { add() {}, toggle() {} }; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; this.value = ''; }
  addEventListener(type, listener, capture) { (this.listeners ??= {})[type] = { listener, capture }; }
  querySelectorAll() { return []; }
  getAttribute(name) { return name === 'aria-required' ? 'true' : null; }
  reportValidity() { return true; }
  focus() {}
}
const nodes = new Map();
const document = { querySelector(id) { if (!nodes.has(id)) nodes.set(id, new Element(id)); return nodes.get(id); } };
let saved = {
  openai_base_url: 'https://old-llm.test/v1', llm_model: 'chat-old',
  embedding_base_url: 'https://old-embed.test/v1', embedding_model: 'embed-old',
  mineru_model_version: 'vlm', audio_transcribe_provider: 'openai', audio_transcribe_base_url: '', audio_transcribe_model: '',
  aliyun_region: 'cn-shanghai', chunk_size: 800, chunk_overlap: 120, embedding_max_input_bytes: 512, embedding_batch_size: 10,
};
const requests = [];
let saveFailure = false;
let modelFailure = false;
let pendingSave = null;
let pendingModels = null;
const context = vm.createContext({ document,
  // Combobox behavior is exercised with the real picker in test_form_controls.cjs.
  enhanceSelect(select, { input }) { let selected = ''; return { value: () => selected, sync({ value, preserveInput = false } = {}) {
    selected = value ?? (select.value === '__custom__' ? selected : select.value);
    if (!preserveInput) input.value = selected;
    input.hidden = false;
    input.required = select.getAttribute('aria-required') === 'true';
  } }; },
  Option: function(label, value) { const e = new Element(); e.textContent = label; e.value = value; return e; },
  state: { status: {} }, on() {}, formatBytes: String, refreshStatus: async () => {},
  requestJson: async (url, options) => {
    const body = options?.body ? JSON.parse(options.body) : null;
    requests.push({ url, body });
    if (url === '/api/config' && !body) return structuredClone(saved);
    if (url === '/api/config') {
      if (pendingSave) await pendingSave;
      if (saveFailure) throw new Error('disk full');
      const changes = { ...body }; delete changes.section;
      saved = { ...saved, ...changes, updated: Object.keys(changes) };
      return structuredClone(saved);
    }
    if (url === '/api/models') {
      if (pendingModels) await pendingModels;
      if (modelFailure) throw new Error('endpoint offline');
      return { scope: 'cloud', models: [{ id: 'listed' }] };
    }
    throw new Error(url);
  },
});
let picker = fs.readFileSync('app/web_static/js/model-picker.js', 'utf8');
picker = picker.slice(picker.indexOf('const CUSTOM_MODEL')).replace('export function ', 'function ').replace(/export \{ CUSTOM_MODEL \};/, '');
vm.runInContext(picker, context);
const source = fs.readFileSync('app/web_static/js/views/settings.js', 'utf8');
vm.runInContext(source.slice(source.indexOf('const el =')).replace('export default', 'const settingsView ='), context);
const node = id => document.querySelector(`#config-${id}`);
(async () => {
  vm.runInContext('settingsView.mount()', context);
  const invalid = node('form').listeners.invalid;
  assert.equal(invalid.capture, true, 'native invalid events must be captured before focusing a collapsed field');
  const collapsed = { open: false };
  invalid.listener({ target: { closest(selector) { assert.equal(selector, 'details'); return collapsed; } } });
  assert.equal(collapsed.open, true, 'invalid fields must reveal their section for correction');
  await context.loadConfig();
  assert.equal(requests.filter(request => request.url === '/api/models').length, 2, 'opening settings must load both full model catalogs');
  assert.equal(node('llm-save').disabled, true);
  assert.equal(node('embedding-reset').disabled, true);
  assert.equal(node('aliyun-region').value, 'cn-shanghai');
  assert.equal(node('chunk-size').value, 800);
  assert.equal(node('llm-model-custom').required, true);
  node('llm-model-custom').value = 'chat';
  context.renderActions('llm');
  assert.equal(node('llm-save').disabled, true, 'an unfinished model search must not enable saving');
  vm.runInContext('pickers.llm.setValue("chat-old")', context);
  context.renderActions('llm');
  assert.equal(node('llm-save').disabled, true, 'selecting the same model must stay clean');
  node('openai-base-url').value = 'https://new-llm.test/v1';
  node('embedding-base-url').value = 'https://draft-embedding.test/v1';
  node('mineru-token').value = 'unsaved-token';
  context.renderActions('llm');
  assert.equal(node('llm-save').disabled, false);
  let resolveSave;
  pendingSave = new Promise(resolve => { resolveSave = resolve; });
  const save = context.save('llm');
  assert.equal(node('llm-save').textContent, '保存中…');
  assert.equal(node('llm-save').disabled, true);
  const count = requests.length;
  await context.save('llm');
  assert.equal(requests.length, count, 'duplicate save must be suppressed');
  node('asr-model').value = 'edited-during-request';
  resolveSave(); await save; pendingSave = null;
  assert.deepEqual(requests.find(r => r.body?.section === 'llm').body, { section: 'llm', openai_base_url: 'https://new-llm.test/v1' });
  assert.equal(node('embedding-base-url').value, 'https://draft-embedding.test/v1');
  assert.equal(node('asr-model').value, 'edited-during-request');
  assert.equal(node('mineru-token').value, 'unsaved-token');
  assert.match(node('llm-status').textContent, /已保存/);
  const llmMessage = node('llm-status').textContent;
  context.renderActions('embedding');
  assert.equal(node('llm-status').textContent, llmMessage, 'editing another section must retain save time');

  node('openai-base-url').value = 'https://discard.test/v1';
  const beforeDiscard = requests.length;
  context.discard('llm');
  assert.equal(requests.length, beforeDiscard, 'discard must restore locally without a reload');
  assert.equal(node('openai-base-url').value, 'https://new-llm.test/v1');
  assert.equal(node('mineru-token').value, 'unsaved-token');
  assert.match(node('llm-status').textContent, /已恢复/);

  saveFailure = true;
  node('embedding-api-key').value = 'new-private-key';
  await context.save('embedding');
  assert.equal(node('embedding-api-key').value, 'new-private-key');
  assert.equal(node('embedding-base-url').value, 'https://draft-embedding.test/v1');
  assert.match(node('embedding-status').textContent, /保存失败.*disk full/);
  assert.equal(node('embedding-save').disabled, false);
  saveFailure = false;
  await context.save('embedding');
  assert.equal(node('embedding-api-key').value, '');
  assert.equal(node('mineru-token').value, 'unsaved-token');
  assert.match(node('embedding-status').textContent, /已保存/);

  // Refresh uses current inputs, preserves custom edits even made while awaiting.
  vm.runInContext('pickers.llm.setValue("custom-draft")', context);
  document.activeElement = node('llm-model-custom');
  node('openai-base-url').value = 'https://unsaved-refresh.test/v1';
  let resolveModels;
  pendingModels = new Promise(resolve => { resolveModels = resolve; });
  const beforeRefresh = requests.filter(r => r.url === '/api/config' && r.body).length;
  const refreshing = context.refreshModels();
  assert.equal(node('refresh-models').textContent, '刷新中…');
  node('llm-model-custom').value = 'custom-edited-while-waiting';
  resolveModels(); await refreshing; pendingModels = null;
  assert.equal(vm.runInContext('pickers.llm.value()', context), 'custom-draft');
  assert.equal(node('llm-model-custom').value, 'custom-edited-while-waiting');
  assert.equal(node('openai-base-url').value, 'https://unsaved-refresh.test/v1');
  assert.equal(node('mineru-token').value, 'unsaved-token');
  assert.equal(requests.filter(r => r.url === '/api/config' && r.body).length, beforeRefresh);
  assert.match(node('models-feedback').textContent, /LLM：1 个模型/);
  modelFailure = true;
  await context.refreshModels();
  assert.match(node('models-feedback').textContent, /刷新失败.*endpoint offline/);
  assert.equal(vm.runInContext('pickers.llm.value()', context), 'custom-draft');
  assert.equal(node('llm-model-custom').value, 'custom-edited-while-waiting');
  assert.equal(node('refresh-models').disabled, false);
  node('asr-provider').value = 'aliyun';
  context.toggleAsrFields();
  assert.equal(node('aliyun-ak-id').required, true);
  assert.equal(node('aliyun-region').required, true);
  vm.runInContext('configState.loaded.aliyun_access_key_id_set = true', context);
  context.toggleAsrFields();
  assert.equal(node('aliyun-ak-id').required, false, 'saved credentials must not require re-entry');
  node('asr-provider').value = 'openai';
  context.toggleAsrFields();
  assert.equal(node('aliyun-region').required, false, 'inactive provider must not block saves');
  console.log('PASS: section save/discard isolation, real request feedback, failed-save input retention, model refresh preservation.');
})().catch(error => { console.error(error); process.exitCode = 1; });
