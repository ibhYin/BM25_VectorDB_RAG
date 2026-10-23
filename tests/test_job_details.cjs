// Run with: node tests/test_job_details.cjs (no browser or dependencies).
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

// Only the DOM operations used by the task renderer are needed here.
class Element {
  constructor() {
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.open = false;
    this.classList = { toggle() {} };
  }
  setAttribute() {}
  prepend(...children) { this.children.unshift(...children); }
  querySelector(selector) {
    if (selector === "input[data-select-all]") return this.children.flatMap(child => [child, ...child.allChildren()]).find(child => child.dataset.selectAll);
    return null;
  }
  allChildren() { return this.children.flatMap(child => [child, ...child.allChildren()]); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener() {}
  querySelectorAll(selector) {
    return this.children.flatMap(child => [
      ...(child.className === selector.slice(1) ? [child] : []),
      ...child.querySelectorAll(selector),
    ]);
  }
}

const nodes = new Map();
const document = {
  createElement: () => new Element(),
  querySelector(selector) {
    if (!nodes.has(selector)) nodes.set(selector, new Element());
    return nodes.get(selector);
  },
};
const job = id => ({
  id, status: "indexing", filenames: ["note.md"],
  file_details: [{ name: "note.md", status: "loading" }],
});
let serverJobs = [job("a"), job("b")];
let timerId = 0;
const timers = new Map();
const context = vm.createContext({
  AbortSignal,
  document,
  state: { jobs: [], polling: null, identity: { isAdmin: true } },
  requestJson: async () => ({ jobs: structuredClone(serverJobs) }),
  clearListLoading() {},
  emit() {},
  showToast() {},
  uploadsToStaging: () => false,
  renderListError(_element, message) { throw new Error(message); },
  setTimeout(callback, delay) {
    timers.set(++timerId, { callback, delay });
    return timerId;
  },
  clearTimeout(id) { timers.delete(id); },
});
const source = fs.readFileSync(
  path.join(__dirname, "../app/web_static/js/views/library.js"), "utf8",
);
vm.runInContext(source.slice(source.indexOf("const el ="), source.indexOf("export default")), context);
const details = () => nodes.get("#job-list").querySelectorAll(".job-files");

(async () => {
  await context.pollJobs();
  assert.deepEqual(details().map(box => box.open), [false, false]);
  details()[0].open = true; // User expands task a.
  await context.pollJobs();
  assert.equal(details()[0].open, true, "polling must preserve manual expansion");
  assert.equal(details()[1].open, false, "other tasks must remain collapsed");

  serverJobs.reverse();
  serverJobs[1].file_details.push({ name: "next.md", status: "loading" });
  await context.pollJobs();
  assert.deepEqual(details().map(box => box.open), [false, true], "state must follow task ID");
  assert.equal(details()[1].children[1].children.length, 2, "expanded content must still update");

  details()[1].open = false; // User collapses task a.
  details()[0].open = true; // User expands task b.
  context.renderJobs(); // Upload progress also uses this renderer, without polling.
  assert.deepEqual(details().map(box => box.open), [true, false]);
  serverJobs[0].status = "completed";
  await context.pollJobs();
  assert.deepEqual(details().map(box => box.open), [true, false], "completion must preserve state");

  const beforeFailure = details()[0];
  context.requestJson = async () => { throw new Error("offline"); };
  await context.pollJobs();
  assert.equal(details()[0], beforeFailure, "failure must preserve existing DOM");
  assert.match(nodes.get("#queue-summary").textContent, /连接异常/);
  context.requestJson = async () => ({ jobs: structuredClone(serverJobs) });
  await context.pollJobs();
  assert.equal(details()[0].open, true, "recovery must preserve expansion");

  let resolveRequest;
  let requests = 0;
  context.requestJson = () => {
    requests += 1;
    return new Promise(resolve => { resolveRequest = resolve; });
  };
  context.startPolling();
  const firstPoll = context.pollJobs();
  const samePoll = context.pollJobs();
  assert.equal(requests, 1, "poll requests must never overlap");
  assert.equal(timers.size, 0, "next poll must wait for completion");
  resolveRequest({ jobs: structuredClone(serverJobs) });
  await Promise.all([firstPoll, samePoll]);
  assert.equal(timers.size, 1);
  assert.equal([...timers.values()][0].delay, 1500);
  context.stopPolling();
  assert.equal(timers.size, 0);

  context.startPolling();
  const stalePoll = context.pollJobs();
  context.stopPolling();
  resolveRequest({ jobs: [job("stale")] });
  await stalePoll;
  assert.equal(context.state.jobs[0].id, serverJobs[0].id, "late responses after leaving must be ignored");
  assert.equal(timers.size, 0, "late responses must not restart polling");

  context.requestJson = async () => { throw new Error("offline"); };
  context.startPolling();
  await context.pollJobs();
  assert.equal([...timers.values()][0].delay, 3000, "failed requests must back off");
  context.stopPolling();

  context.startPolling = () => {};
  context.FormData = class { append() {} };
  const accepted = [];
  let busyReplies = 0;
  let attempts = 0;
  context.XMLHttpRequest = class {
    constructor() {
      this.handlers = {};
      this.upload = { addEventListener() {} };
    }
    open() {}
    addEventListener(event, handler) { this.handlers[event] = handler; }
    send() {
      attempts += 1;
      if (accepted.length === 5 && busyReplies < 2) {
        busyReplies += 1;
        this.status = 503;
        this.responseText = JSON.stringify({ error: "Upload queue is full" });
      } else {
        const local = context.state.jobs.find(entry => entry.status === "uploading");
        accepted.push([...local.filenames]);
        this.status = 202;
        this.responseText = JSON.stringify({ ...job(`accepted-${accepted.length}`), filenames: local.filenames });
      }
      this.handlers.load();
    }
  };
  context.state.jobs = [];
  const files = Array.from({ length: 61 }, (_, i) => ({ name: `note-${i}.md`, size: 10 }));
  const uploading = context.uploadFiles(files);
  // Drain resolved upload promises until the sixth batch reaches the full queue.
  for (let i = 0; i < 30; i++) await Promise.resolve();
  assert.equal(accepted.length, 5);
  assert.equal(attempts, 6, "later batches must wait while the current one is full");
  assert(context.state.jobs.some(entry => entry.status === "waiting_upload"));
  for (let retry = 0; retry < 2; retry++) {
    assert.equal(timers.size, 1);
    const [id, timer] = [...timers.entries()][0];
    timers.delete(id);
    timer.callback();
    for (let i = 0; i < 30; i++) await Promise.resolve();
  }
  await uploading;
  assert.equal(accepted.length, 7);
  assert.equal(accepted.flat().length, 61);
  assert.equal(new Set(accepted.flat()).size, 61, "retry must not duplicate accepted files");
  assert.equal(attempts, 9);
  assert.equal(timers.size, 0);
  context.state.jobs = [];
  context.state.access = { manageable: false };
  context.state.documents = [
    { doc_id: "a", title: "A", chunk_count: 1 },
    { doc_id: "b", title: "B", chunk_count: 1 },
  ];
  context.renderDocuments();
  context.selectDocument("a", true);
  context.selectDocument("b", true);
  context.renderDocuments();
  assert.match(nodes.get("#document-selection-count").textContent, /已选 2/);
  context.renderDocuments();
  assert.match(nodes.get("#document-selection-count").textContent, /已选 2/, "refresh must retain valid selection");
  context.confirm = () => true;
  let resolveDelete;
  context.requestJson = () => new Promise(resolve => { resolveDelete = resolve; });
  context.refreshDocuments = async () => {
    context.state.documents = [{ doc_id: "b", title: "B", chunk_count: 1 }];
    context.renderDocuments();
  };
  const deleting = context.deleteDocuments(["a", "b"]);
  assert.equal(context.state.documents.length, 2, "no optimistic removal while request is pending");
  resolveDelete({ deleted: ["a"], not_deleted: [], failed: ["b"], error: "partial write failure" });
  await deleting;
  assert.match(nodes.get("#document-delete-feedback").textContent, /未完成 1.*partial write failure/);
  assert.match(nodes.get("#document-selection-count").textContent, /已选 1/, "failed document must remain selected");
  context.requestJson = async () => { throw new Error("delete offline"); };
  await context.deleteDocuments(["b"]);
  assert.equal(context.state.documents.length, 1);
  assert.match(nodes.get("#document-delete-feedback").textContent, /delete offline/);
  context.state.documents = [];
  context.renderDocuments();
  assert.match(nodes.get("#document-selection-count").textContent, /已选 0/);
  console.log("PASS: details, serial polling/recovery, queue retries, selection persistence and partial-delete feedback.");

})().catch(error => { console.error(error); process.exitCode = 1; });
