// Run: node tests/test_retrieval_stages.cjs. Real rendering data, no browser/API.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('app/web_static/js/views/ask.js', 'utf8');
const html = fs.readFileSync('app/web_static/index.html', 'utf8');
for (const id of ['retrieval-warning', 'retrieval-warning-title', 'retrieval-warning-message',
  'retrieval-diagnostic-toggle', 'retrieval-diagnostic-copy', 'retrieval-diagnostic']) {
  assert.match(html, new RegExp(`id="${id}"`), `missing retrieval warning control: ${id}`);
}
const context = vm.createContext({});
vm.runInContext(source.slice(source.indexOf('function fmt('), source.indexOf('function createStageRow(')), context);
function stages(retrieval = {}, trace = null, hits = [{chunk_id: 'doc#chunk_1', rank: 1, similarity: .6, bm25_score: 2.49}]) {
  context.input = {retrieval, trace, hits};
  return vm.runInContext('buildStages("hybrid", {retrieval: input.retrieval}, input.trace, input.hits)', context);
}
const legacy = stages();
assert.equal(legacy[0].stat, '未记录');
assert.equal(legacy[1].stat, '未记录');
assert.match(legacy[0].empty, /未保存/);
assert.equal(legacy[2].rows.length, 1);
const summary = stages({vector_candidates: 100, bm25_candidates: 0});
assert.equal(summary[0].stat, '100 条');
assert.match(summary[0].empty, /未保存/);
assert.equal(summary[1].stat, '0 条');
assert.match(summary[1].empty, /没有命中/);
const partial = stages({}, null, [{chunk_id: 'doc#chunk_1', vector_rank: 2}]);
assert.equal(partial[0].stat, '总数未记录');
assert.equal(partial[0].rows.length, 1);
const empty = stages({}, {vector_count: 0, bm25_count: 0, union_count: 0, candidates: []}, []);
assert.equal(empty[0].stat, '0 条');
assert.equal(empty[2].rows.length, 0);
const complete = stages({}, {vector_count: 1, bm25_count: 0, union_count: 1, candidates: [{chunk_id: 'doc#chunk_1', vector_rank: 1}]});
assert.equal(complete[0].rows.length, 1);
assert.equal(complete[0].stat, '1 条');
const rerankHits = [{chunk_id: 'doc#chunk_1', rank: 1, retrieval_rank: 3, fusion_rank: 3, rerank_rank: 1, rerank_score: .82}];
const reranked = stages({fusion_method: 'rrf', rerank: {enabled: true, status: 'completed', model: 'rerank-model', input_count: 2, output_count: 1}},
  {vector_count: 1, bm25_count: 1, union_count: 2, candidates: [
    {chunk_id: 'doc#chunk_1', vector_rank: 1, bm25_rank: 1, final_rank: 2},
    {chunk_id: 'doc#chunk_2', vector_rank: 2, bm25_rank: null, final_rank: 1},
  ]}, rerankHits);
assert.equal(reranked.length, 4, 'hybrid rerank must show two recalls, fusion and rerank separately');
assert.equal(reranked[2].cards, false, 'fusion candidates must stay as compact rows before rerank');
assert.equal(reranked[3].rankOf(rerankHits[0]), 1);
assert.match(reranked[3].note, /rerank-model.*2 → 1/);
const degraded = stages({rerank: {enabled: true, status: 'degraded', reason: 'fallback', error_message: 'timeout', input_count: 1, output_count: 1, text: 'private candidate', query: 'private query'}}, null, rerankHits);
assert.ok(degraded[3].diagnostic, 'degraded rerank must expose a copyable diagnostic');
const diagnostic = JSON.parse(context.rerankDiagnosticText(degraded[3].diagnostic));
assert.equal(diagnostic.error_message, 'timeout');
assert.equal('text' in diagnostic || 'query' in diagnostic, false, 'diagnostics must never copy candidate text or the query');
const degradedRecall = {
  task_id: 'task-1',
  recall_detail: {
    recall_id: 'recall-1', status: 'degraded', parallel: true, wall_ms: 120,
    succeeded_paths: ['bm25'], failed_paths: ['vector'],
    routes: {
      vector: {status: 'failed', elapsed_ms: 110, count: 0, error: {
        stage: 'embedding', error_type: 'AuthenticationError', error_message: 'API Key 无效',
        http_status: 401, provider_request_id: 'provider-1', retryable: false,
        hint: '检查 Embedding API Key、模型权限和请求地址', query: 'private query', text: 'private text',
      }},
      bm25: {status: 'completed', elapsed_ms: 80, count: 1},
    },
  },
};
const degradedTrace = {vector_count: 0, bm25_count: 1, union_count: 1,
  candidates: [{chunk_id: 'doc#chunk_1', vector_rank: null, bm25_rank: 1, final_rank: 1}]};
const degradedStages = stages(degradedRecall, degradedTrace, [
  {chunk_id: 'doc#chunk_1', rank: 1, bm25_rank: 1, bm25_score: 1.2},
]);
assert.equal(degradedStages[0].status, 'failed');
assert.match(degradedStages[0].title, /失败/);
assert.match(degradedStages[0].note, /检查 Embedding API Key.*两路并行/);
assert.ok(degradedStages[0].diagnostic, 'failed vector route must expose diagnostics');
assert.match(degradedStages[1].note, /唯一有效召回/);
assert.match(degradedStages[2].note, /降级融合/);
const warning = context.retrievalRecallSummary(degradedRecall);
assert.equal(warning.title, '检索降级：仅 BM25');
assert.match(warning.message, /向量召回失败.*语义相关内容/);
const recallDiagnostic = JSON.parse(context.retrievalDiagnosticText(degradedRecall));
assert.equal(recallDiagnostic.task_id, 'task-1');
assert.equal(recallDiagnostic.routes.vector.error.http_status, 401);
assert.equal(JSON.stringify(recallDiagnostic).includes('private query'), false);
assert.equal(JSON.stringify(recallDiagnostic).includes('private text'), false);
assert.equal(context.retrievalRecallSummary({recall_detail: {status: 'completed'}}), null);
console.log('PASS: parallel/rerank stages, degraded retrieval warnings and sanitized diagnostics stay distinct.');
