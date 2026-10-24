// Run: node tests/test_form_controls.cjs. Actual UI modules, DOM seam, no browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const observers = [];
class Element {
  constructor(id = '', tag = 'div') {
    Object.assign(this, { id, tagName: tag.toUpperCase(), children: [], dataset: {}, attributes: {}, events: {}, _value: '', textContent: '', hidden: false, disabled: false, open: false, isConnected: true });
    this.className = '';
    this.classList = {
      add: name => { this.className = [...new Set([...this.className.split(' ').filter(Boolean), name])].join(' '); },
      remove: name => { this.className = this.className.split(' ').filter(value => value !== name).join(' '); },
      contains: name => this.className.split(' ').includes(name),
      toggle: (name, force) => { const enabled = force ?? !this.classList.contains(name); this.classList[enabled ? 'add' : 'remove'](name); return enabled; },
    };
  }
  get options() { return this.children; }
  get value() { return this._value; }
  set value(value) { this._value = this.tagName === 'SELECT' && !this.children.some(child => child.value === value) ? '' : value; }
  getAttribute(name) { return this.attributes[name] ?? null; }
  hasAttribute(name) { return name in this.attributes; }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  insertAdjacentElement(_position, element) {
    if (this.parentElement) {
      const parent = this.parentElement;
      parent.children.splice(parent.children.indexOf(this), 0, element);
      element.parentElement = parent;
    }
  }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  append(...children) {
    for (const child of children) {
      if (child.parentElement) child.parentElement.children = child.parentElement.children.filter(item => item !== child);
      child.parentElement = this;
      this.children.push(child);
    }
    if (this.tagName === 'SELECT' && !this.children.some(child => child.value === this._value)) this._value = this.children[0]?.value || '';
  }
  addEventListener(name, handler) { (this.events[name] ??= new Set()).add(handler); }
  removeEventListener(name, handler) { this.events[name]?.delete(handler); }
  fire(name, props = {}) {
    const event = { stopped: false, prevented: false, target: this, stopPropagation() { this.stopped = true; }, preventDefault() { this.prevented = true; }, ...props };
    this[`on${name}`]?.(event);
    for (const handler of [...(this.events[name] || [])]) handler(event);
    return event;
  }
  dispatchEvent(event) { return this.fire(event.type, event); }
  showModal() { this.open = true; }
  close() { this.open = false; this.fire('close'); }
  focus() { if (document.activeElement !== this) { document.activeElement = this; this.fire('focus'); } }
  select() {}
}
const nodes = new Map();
const document = {
  activeElement: new Element('trigger'),
  createElement: tag => new Element('', tag),
  querySelector(id) { if (!nodes.has(id)) nodes.set(id, new Element(id)); return nodes.get(id); },
};
const option = (label, value) => Object.assign(new Element('', 'option'), { textContent: label, value });
let request;
const context = vm.createContext({
  document, Option: function(label, value) { return option(label, value); },
  Event: class { constructor(type, options) { Object.assign(this, { type }, options); } },
  MutationObserver: class {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe() {} disconnect() {} takeRecords() { return []; }
  },
  state: { jobs: [] },
  router: { navigate(name) { assert.equal(name, 'library'); } },
  emit() {},
  requestJson: (...args) => request(...args),
});
vm.runInContext(fs.readFileSync('app/web_static/js/select-search.js', 'utf8').replaceAll('export function ', 'function '), context);
const select = new Element('models', 'select');
select.attributes['aria-label'] = '模型';
select.append(option('Embedding 中文', 'embed-v1'), option('Chat 英文', 'chat-v1'), option('自定义…', '__custom__'));
select.value = 'embed-v1';
const container = new Element(); container.append(select);
const control = context.enhanceSelect(select);
const filterObserver = observers[0];
assert.equal(container.children.length, 1, 'one integrated control replaces the old select layout');
assert.equal(container.children[0].children.filter(node => node.tagName === 'INPUT').length, 1);
assert.equal(container.children[0].children[0], control.input, 'implicit labels must target the visible input');
assert.equal(select.classList.contains('search-select-source'), true);
control.input.focus();
assert.equal(control.list.children.length, 3, 'opening a fixed select shows every option, even IDs that resemble a model sentinel');
control.input.value = 'CHAT';
assert.equal(control.input.fire('input').stopped, true, 'generic search must not mark settings dirty');
assert.deepEqual(control.list.children.map(row => row.textContent), ['Chat 英文']);
assert.deepEqual(select.options.map(o => o.value), ['embed-v1', 'chat-v1', '__custom__'], 'filter never removes source options');
assert.equal(select.value, 'embed-v1', 'search cannot change selected config');
control.input.value = '中文'; control.input.fire('input');
assert.deepEqual(control.list.children.map(row => row.textContent), ['Embedding 中文']);
control.input.value = 'not-present'; control.input.fire('input');
assert.match(control.list.children[0].textContent, /没有匹配/);
assert.equal(select.value, 'embed-v1');
control.input.fire('blur');
assert.equal(control.input.value, 'Embedding 中文', 'uncommitted generic search restores its selected label');
let changes = 0; select.addEventListener('change', () => changes++);
control.input.value = 'chat'; control.input.fire('input');
assert.equal(control.input.fire('keydown', { key: 'Enter' }).prevented, true);
assert.equal(select.value, 'chat-v1');
assert.equal(changes, 1);
assert.equal(control.list.hidden, true);
select.value = 'embed-v1'; context.resetSelectSearch(select);
assert.equal(control.input.value, 'Embedding 中文', 'discard syncs the single input with the saved option');
select.replaceChildren(option('New model', 'new-model'), option('Other', 'other'));
select.value = 'new-model';
document.activeElement = null;
filterObserver.callback([{ type: 'childList' }]);
assert.equal(control.input.value, 'New model');
control.input.focus();
assert.equal(control.list.children.length, 2, 'catalog reload exposes all new options');
assert.equal(control.list.children[1].fire('click').prevented, true, 'label default activation must not reopen the menu after selection');
assert.equal(select.value, 'other');
control.input.fire('click');
control.input.fire('keydown', { key: 'ArrowUp' });
control.input.fire('keydown', { key: 'Enter' });
assert.equal(select.value, 'new-model', 'arrow keys navigate before Enter commits');
select.hidden = true; select.disabled = true;
filterObserver.callback([{ type: 'attributes' }]);
assert.equal(container.children[0].hidden, true);
assert.equal(control.input.disabled, true);

// Integrate the actual model picker, including async catalog refresh and custom names.
let picker = fs.readFileSync('app/web_static/js/model-picker.js', 'utf8');
picker = picker.slice(picker.indexOf('const CUSTOM_MODEL')).replace('export function ', 'function ').replace(/export \{ CUSTOM_MODEL \};/, '');
vm.runInContext(picker, context);
const source = new Element('llm', 'select');
source.setAttribute('aria-required', 'true');
const custom = new Element('custom', 'input');
const modelField = new Element(); modelField.append(source, custom);
const pickerControl = context.createModelPicker({ select: source, custom, hint: new Element(), target: 'llm' });
pickerControl.setValue('saved-model');
const combo = context.enhanceSelect(source);
assert.equal(modelField.children.length, 1, 'model select and custom input share one wrapper');
assert.equal(combo.input, custom, 'reuse the existing model input rather than adding a second row');
assert.equal(custom.hidden, false);
assert.equal(custom.required, true);
assert.equal(custom.value, 'saved-model');

const core = fs.readFileSync('app/web_static/js/core.js', 'utf8');
vm.runInContext(core.slice(core.indexOf('let rebuildSubmitting =')).replace('export async function ', 'async function '), context);
const node = id => document.querySelector(`#rebuild-dialog${id ? `-${id}` : ''}`);
const plan = { knowledge_base_id: 'default', documents: 12, original_model: 'old', target_model: 'new', reuse_chunks: 12, resplit: 0, parse: 0, plan_id: 'fingerprint' };
const tick = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  request = async () => ({ models: [{ id: 'Chat-model-1' }, { id: 'Chat-model-2' }, { id: 'embed-other' }] });
  await pickerControl.load();
  custom.focus();
  assert.equal(combo.list.children.length, 4, 'full fetched catalog plus saved model are available');
  let sameModelChanges = 0;
  source.addEventListener('change', () => sameModelChanges++);
  custom.value = 'saved';
  assert.equal(custom.fire('input').stopped, true, 'model search must not mark settings dirty');
  assert.equal(pickerControl.value(), 'saved-model', 'search text must not replace the selected model');
  combo.list.children.find(row => row.textContent.startsWith('saved-model')).fire('click');
  assert.equal(sameModelChanges, 0, 'reselecting the same model must not report a change');
  custom.value = 'abandoned-query'; custom.fire('input'); custom.fire('change'); custom.fire('blur');
  assert.equal(custom.value, 'saved-model', 'blur restores the selected model instead of committing a search');
  assert.equal(pickerControl.value(), 'saved-model');
  custom.value = ''; custom.fire('input');
  assert.equal(custom.value, 'saved-model', 'clearing the search restores the selected label');
  assert.equal(sameModelChanges, 0);
  custom.value = 'CHAT 2'; custom.fire('input');
  assert.equal(combo.list.children[0].textContent, 'Chat-model-2', 'case-insensitive token matching finds a non-selected model');
  custom.fire('keydown', { key: 'Enter' });
  assert.equal(pickerControl.value(), 'Chat-model-2');
  let modelChanges = 0; source.addEventListener('change', () => modelChanges++);
  custom.value = 'unlisted-model'; custom.fire('input');
  assert.match(combo.list.children.at(-1).textContent, /自定义/);
  custom.fire('keydown', { key: 'Enter', isComposing: true });
  assert.equal(modelChanges, 0, 'IME Enter must not commit a partial composition');
  custom.fire('keydown', { key: 'Enter' });
  assert.equal(pickerControl.value(), 'unlisted-model');
  assert.equal(source.value, '__custom__');
  assert.equal(modelChanges, 1);
  let finishCatalog;
  request = async () => new Promise(resolve => { finishCatalog = resolve; });
  const reload = pickerControl.load();
  custom.value = 'edited-during-refresh'; custom.fire('input');
  finishCatalog({ models: [{ id: 'newly-listed' }] }); await reload;
  assert.equal(pickerControl.value(), 'unlisted-model', 'refresh cannot turn a search into the selected model');
  assert.equal(custom.value, 'edited-during-refresh', 'refresh preserves the visible query');
  assert.equal(combo.list.children.length, 1, 'refresh preserves the active query as well as input');
  request = async () => { throw new Error('catalog offline'); };
  const failedRefresh = await pickerControl.load();
  assert.match(failedRefresh.error, /catalog offline/);
  assert.equal(pickerControl.value(), 'unlisted-model');
  assert.equal(custom.value, 'edited-during-refresh', 'failed catalog load cannot erase search input');
  custom.focus(); custom.fire('keydown', { key: 'Escape' });
  assert.equal(combo.list.hidden, true);
  assert.equal(custom.value, 'unlisted-model', 'Escape restores even a committed custom model');
  custom.value = 'tab-abandoned'; custom.fire('input'); custom.fire('keydown', { key: 'Tab' });
  assert.equal(custom.value, 'unlisted-model');
  custom.value = ''; custom.fire('input'); custom.fire('blur');
  assert.equal(custom.value, 'unlisted-model', 'clear and blur restore a custom selection too');
  pickerControl.setValue('restored-model');
  assert.equal(custom.value, 'restored-model');
  assert.equal(modelField.children.length, 1);

  let finishPreview;
  const calls = [];
  request = async (url, options) => {
    calls.push({ url, options });
    if (!options) return new Promise(resolve => { finishPreview = resolve; });
    return { id: 'rebuild-job', kind: 'rebuild', status: 'queued' };
  };
  const run = context.requestIndexRebuild();
  assert.equal(node('').open, true);
  assert.match(node('message').textContent, /正在检查当前文档与缓存/);
  assert.equal(node('message').getAttribute('aria-busy'), 'true');
  assert.equal(calls.length, 1);
  await context.requestIndexRebuild();
  assert.equal(calls.length, 1, 'duplicate requests suppressed');
  finishPreview(plan); await tick();
  assert.match(node('message').textContent, /12 份/);
  assert.match(node('message').textContent, /旧索引/);
  assert.equal(node('confirm').hidden, false);
  node('confirm').fire('click');
  await run;
  assert.equal(calls.length, 2);
  assert.deepEqual(JSON.parse(calls[1].options.body), { plan_id: 'fingerprint', force_reparse: false });
  assert.match(node('title').textContent, /任务已提交/);
  assert.equal(node('message').getAttribute('aria-busy'), 'false');
  assert.equal(node('close').disabled, false);
  node('close').fire('click');
  context.state.jobs = [];

  const cancelled = context.requestIndexRebuild();
  node('close').fire('click');
  finishPreview(plan); await cancelled;
  assert.equal(calls.filter(call => call.options).length, 1, 'cancel during preview never submits');

  request = async () => { throw new Error('cache unavailable'); };
  await context.requestIndexRebuild();
  assert.match(node('message').textContent, /cache unavailable/);
  assert.equal(node('close').disabled, false);
  assert.equal(document.querySelector('#warning-rebuild').disabled, false);
  node('close').fire('click');

  let finishSubmit;
  request = async (_url, options) => options ? new Promise(resolve => { finishSubmit = resolve; }) : plan;
  const pending = context.requestIndexRebuild(true); await tick();
  assert.match(node('message').textContent, /强制重新解析/);
  node('confirm').fire('click'); await tick();
  assert.equal(node('close').disabled, true);
  assert.equal(node('').fire('cancel').prevented, true, 'Escape cannot abandon an in-flight submission');
  finishSubmit({ id: 'forced-job', kind: 'rebuild', status: 'queued' }); await pending;
  node('close').fire('click');

  // Exercise the real ask workspace with the real shared model picker.
  context.state.status = { llm_model: 'global-default' };
  context.canConfigure = () => true;
  context.showToast = () => {};
  let aggregateRequest;
  context.submitAggregate = async (question, params) => { aggregateRequest = { question, params }; };
  document.getElementById = id => document.querySelector(`#${id}`);
  document.querySelector('#retrieval-top-k').closest = () => document.querySelector('#top-k-label');
  document.querySelector('#ask-form').reportValidity = () => true;
  document.querySelector('input[name="retrieval-method"]:checked').value = 'hybrid';
  document.querySelector('input[name="fusion-method"]:checked').value = 'rrf';
  document.querySelector('#retrieval-top-k').value = '20';
  document.querySelector('#candidate-pool').value = '100';
  const modelRequests = [];
  request = async (url, options) => {
    modelRequests.push(url);
    assert.equal(url, '/api/models', 'changing a conversation model must never write global config');
    return { scope: 'cloud', models: [{ id: 'global-default' }, { id: 'session-model' }] };
  };
  const askSource = fs.readFileSync('app/web_static/js/views/ask.js', 'utf8');
  vm.runInContext(askSource.slice(askSource.indexOf('const el =')).replace('export default', 'const askView ='), context);
  context.bindModelSwitcher();
  await context.setupModelSwitcher();
  assert.equal(document.querySelector('#ask-model-open').textContent, 'global-default');
  const modelDialog = document.querySelector('#ask-model-dialog');
  await document.querySelector('#ask-model-open').fire('click');
  assert.equal(modelDialog.open, true);
  assert.equal(document.activeElement, document.querySelector('#ask-llm-model-custom'));
  document.querySelector('#ask-llm-model-custom').value = 'unfinished search';
  await context.setupModelSwitcher();
  assert.equal(document.querySelector('#ask-llm-model-custom').value, 'unfinished search', 'status refresh cannot erase an active model search');
  modelDialog.fire('keydown', { key: 'Escape' });
  assert.equal(modelDialog.open, false);
  assert.equal(context.requestParams().llm_model, 'global-default', 'Escape cancels the search without selecting a new model');
  modelDialog.showModal();
  vm.runInContext('llmPicker.setValue("session-model")', context);
  document.querySelector('#ask-llm-model').fire('change');
  assert.equal(modelDialog.open, false);
  assert.equal(document.activeElement, document.querySelector('#ask-model-open'));
  assert.equal(context.requestParams().llm_model, 'session-model');
  context.state.status.llm_model = 'new-global-default';
  await context.setupModelSwitcher();
  assert.equal(context.requestParams().llm_model, 'session-model', 'global refresh cannot replace this conversation selection');
  context.setWorkspaceMode('aggregate');
  assert.equal(document.querySelector('#chat-stream').hidden, true);
  assert.equal(document.querySelector('#aggregate-workspace').hidden, false);
  assert.equal(document.querySelector('#aggregate-sample-size').disabled, false);
  assert.equal(document.querySelector('#retrieval-top-k').disabled, true);
  document.querySelector('#question').value = '归类主题';
  await context.ask({ preventDefault() {} });
  assert.equal(aggregateRequest.question, '归类主题');
  assert.equal(aggregateRequest.params.llm_model, 'session-model');
  assert.equal(aggregateRequest.params.retrieval_method, 'hybrid');
  context.setWorkspaceMode('ask');
  assert.equal(document.querySelector('#aggregate-sample-size').disabled, true, 'hidden statistics fields cannot block Q&A validation');
  assert.equal(document.querySelector('#retrieval-top-k').disabled, false);
  assert.equal(document.querySelector('#question').value, '归类主题', 'both modes share one input without losing the draft');
  context.canConfigure = () => false;
  await context.setupModelSwitcher();
  assert.equal(document.querySelector('#ask-model-open').hidden, true);
  assert.equal(document.querySelector('#ask-llm-readonly').hidden, false);
  assert.equal(context.requestParams().llm_model, 'new-global-default');
  assert.ok(modelRequests.length);

  const html = fs.readFileSync('app/web_static/index.html', 'utf8');
  for (const input of html.matchAll(/<input\b[^>]*type="(?:number|text|url|password)"[^>]*>/g)) assert.match(input[0], /placeholder="[^"]+"/, input[0]);
  assert.match(html, /id="config-aliyun-region"[^>]*value="cn-shanghai"/);
  assert.match(html, /id="config-llm-model"[^>]*aria-required="true"/);
  assert.doesNotMatch(html, /warning-rebuild-feedback/);
  console.log('PASS: model search and dialog focus, conversation model isolation/permissions, shared Q&A/statistics input, form validation, rebuild confirmation, placeholders and defaults.');
})().catch(error => { console.error(error); process.exitCode = 1; });
