// Run: node tests/test_retrieval_stages.cjs. Real rendering data, no browser/API.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('app/web_static/js/views/ask.js', 'utf8');
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
console.log('PASS: historical unknown/partial data stays distinct from genuine zero-hit retrieval.');
