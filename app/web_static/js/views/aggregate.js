// 问答工作区的归类统计模式；复用输入区，保留独立后台任务。

import { requestJson, router } from "../core.js";

let pollingJobId = "";

const el = {
  sampleSize: document.querySelector("#aggregate-sample-size"),
  topCategories: document.querySelector("#aggregate-top-categories"),
  submit: document.querySelector("#ask-button"),
  status: document.querySelector("#aggregate-status"),
  output: document.querySelector("#aggregate-output"),
};

const STAGE_LABELS = {
  queued: "等待处理…",
  retrieve: "正在检索样本…",
  extract: "正在抽取语义类别…",
  cluster: "正在聚类归一…",
  count: "正在计数排序…",
  done: "完成",
  failed: "失败",
};

function stageText(job) {
  const stage = job.stage || "";
  const base = STAGE_LABELS[stage] || stage;
  const progress = job.progress || {};
  if ((stage === "extract" || stage === "count") && progress.total) {
    return `${base}（${progress.done ?? 0}/${progress.total}）`;
  }
  return base;
}

function renderError(message) {
  el.output.replaceChildren();
  const error = document.createElement("div");
  error.className = "empty-list error";
  error.textContent = message;
  el.output.append(error);
}

function evidenceItem(item) {
  const card = document.createElement("article");
  card.className = "aggregate-evidence-item";

  const head = document.createElement("div");
  head.className = "aggregate-evidence-head";
  const title = document.createElement("strong");
  title.textContent = item.title || item.doc_id || "未命名文档";
  const source = document.createElement("span");
  source.textContent = item.source || "";
  head.append(title, source);

  const snippet = document.createElement("p");
  snippet.textContent = item.snippet || "（无原文摘录）";

  const open = document.createElement("button");
  open.type = "button";
  open.className = "ghost-button";
  open.textContent = "在原文查看";
  open.addEventListener("click", () => {
    router.navigate("library", { docId: item.doc_id, chunkId: item.chunk_id });
  });

  card.append(head, snippet, open);
  return card;
}

function categoryCard(category, rank) {
  const card = document.createElement("article");
  card.className = "panel aggregate-category";

  const head = document.createElement("div");
  head.className = "aggregate-category-head";
  const label = document.createElement("strong");
  label.textContent = `#${rank} ${category.label}`;
  const meta = document.createElement("span");
  meta.className = "aggregate-category-meta";
  const ratio = Math.round((Number(category.ratio) || 0) * 1000) / 10;
  meta.textContent = `${category.doc_count} 篇文档 · ${category.chunk_count} 个片段 · 占样本 ${ratio}%`;
  head.append(label, meta);

  const bar = document.createElement("div");
  bar.className = "aggregate-bar";
  const fill = document.createElement("span");
  fill.style.width = `${Math.min(100, Math.max(0, ratio))}%`;
  bar.append(fill);

  const evidence = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `查看证据（${(category.evidence || []).length}）`;
  const list = document.createElement("div");
  list.className = "aggregate-evidence";
  for (const item of category.evidence || []) list.append(evidenceItem(item));
  evidence.append(summary, list);

  card.append(head, bar, evidence);
  return card;
}

function renderResult(result) {
  el.output.replaceChildren();
  const sample = result.sample || {};
  const meta = result.meta || {};
  const heading = document.createElement("h3");
  heading.textContent = result.question || "归类统计结果";
  el.output.append(heading);

  const note = document.createElement("div");
  note.className = "aggregate-note";
  note.textContent = [
    `检索样本：${sample.chunks || 0} 个片段 / ${sample.docs || 0} 篇文档`,
    `模型 ${meta.model || "未知"}`,
    `耗时 ${((Number(meta.elapsed_ms) || 0) / 1000).toFixed(1)} 秒`,
  ].join(" · ");
  el.output.append(note);

  for (const warning of meta.warnings || []) {
    const line = document.createElement("p");
    line.className = "aggregate-warning";
    line.textContent = warning;
    el.output.append(line);
  }

  const categories = result.categories || [];
  if (!categories.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "没有可展示的类别。可以换一种问法或增大样本片段数后重试。";
    el.output.append(empty);
    return;
  }
  categories.forEach((category, index) => {
    el.output.append(categoryCard(category, index + 1));
  });
}

async function poll(jobId) {
  if (pollingJobId !== jobId) return;
  try {
    const job = await requestJson(`/api/aggregate-jobs/${jobId}`);
    if (pollingJobId !== jobId) return;
    el.status.textContent = stageText(job);
    if (job.status === "completed") {
      pollingJobId = "";
      el.submit.disabled = false;
      el.status.textContent = "完成";
      renderResult(job.result || {});
      return;
    }
    if (job.status === "failed") {
      pollingJobId = "";
      el.submit.disabled = false;
      el.status.textContent = "失败";
      renderError(job.error || "聚合处理失败");
      return;
    }
  } catch (error) {
    pollingJobId = "";
    el.submit.disabled = false;
    el.status.textContent = "失败";
    renderError(error.message);
    return;
  }
  setTimeout(() => poll(jobId), 400);
}
export async function submitAggregate(question, params) {
  if (!question || pollingJobId) return;

  el.submit.disabled = true;
  el.output.replaceChildren();
  el.status.textContent = "正在提交…";
  try {
    const job = await requestJson("/api/aggregate-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        sample_size: Number(el.sampleSize.value),
        top_categories: Number(el.topCategories.value),
        ...params,
      }),
    });
    pollingJobId = job.id;
    el.status.textContent = stageText(job);
    poll(job.id);
  } catch (error) {
    el.submit.disabled = false;
    el.status.textContent = "提交失败";
    renderError(error.message);
  }
}
