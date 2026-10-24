// Run: node tests/test_ask_preferences.cjs. Account-scoped retrieval preference seam, no browser/API.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor() {
    this.value = '';
    this.checked = false;
    this.disabled = false;
    this.required = false;
    this.hidden = false;
    this.textContent = '';
    this.className = '';
    this.classList = {
      add: name => { if (!this.classList.contains(name)) this.className += ` ${name}`; },
      remove: name => { this.className = this.className.split(/\s+/).filter(item => item && item !== name).join(' '); },
      contains: name => this.className.split(/\s+/).includes(name),
    };
  }
}

const el = {
  retrievalTopK: new Element(), rerankPoolSize: new Element(), rerankEnabled: new Element(),
  rerankScoreFilterEnabled: new Element(), rerankScoreThreshold: new Element(),
  rerankModelLabel: new Element(), rerankPreferencesStatus: new Element(), rerankPreferencesRetry: new Element(),
};
let preferences = {
  top_k: 70, rerank_enabled: true, rerank_pool_size: 80,
  rerank_score_filter_enabled: true, rerank_score_threshold: .42,
  rerank_model: 'BAAI/bge-reranker-v2-m3', rerank_configured: true,
};
const requests = [];
const context = vm.createContext({
  el,
  requestJson: async (url, options = {}) => {
    requests.push({ url, body: options.body ? JSON.parse(options.body) : null });
    return options.method === 'POST' ? { ...preferences, ...JSON.parse(options.body) } : { ...preferences };
  },
});
const source = fs.readFileSync('app/web_static/js/views/ask.js', 'utf8');
const start = source.indexOf('function updateRerankControls()');
const end = source.indexOf('function updateRetrievalControls()');
assert.ok(start >= 0 && end > start, 'preference helpers must remain in one bounded seam');
vm.runInContext(`
  let retrievalPreferences = null;
  let retrievalPreferencesEpoch = 0;
  let preferencesSaving = false;
  let preferencesDirty = false;
  ${source.slice(start, end)}
`, context);

(async () => {
  await context.loadRetrievalPreferences();
  assert.equal(el.retrievalTopK.value, 70, 'saved Top-K must be restored');
  assert.equal(el.rerankEnabled.checked, true, 'saved rerank switch must be restored');
  assert.equal(el.rerankPoolSize.value, 80, 'saved rerank pool must be restored');
  assert.equal(el.rerankScoreThreshold.value, .42);
  assert.equal(el.rerankScoreThreshold.disabled, false);
  assert.match(el.rerankModelLabel.textContent, /BAAI\/bge-reranker-v2-m3/);

  el.rerankEnabled.checked = false;
  context.updateRerankControls();
  assert.equal(el.rerankPoolSize.disabled, true, 'disabled rerank must not allow editing its pool');
  await context.saveRetrievalPreferences();
  const saved = requests.find(request => request.url === '/api/preferences' && request.body);
  assert.equal(saved.body.rerank_enabled, false);
  assert.equal(saved.body.rerank_pool_size, 80);
  assert.match(el.rerankPreferencesStatus.textContent, /已保存/);

  preferences = { ...preferences, rerank_enabled: true, rerank_configured: false };
  await context.loadRetrievalPreferences();
  assert.equal(el.rerankEnabled.disabled, true, 'unconfigured rerank service must disable the switch');
  assert.equal(el.rerankEnabled.checked, false, 'unconfigured rerank must not be sent as enabled');
  assert.match(el.rerankModelLabel.textContent, /未配置/);
  console.log('PASS: account preferences restore, auto-save, and honor missing rerank configuration.');
})().catch(error => { console.error(error); process.exitCode = 1; });
