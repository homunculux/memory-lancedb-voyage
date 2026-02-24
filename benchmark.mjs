import { readFileSync } from 'fs';

const queries = [
  { q: "ClawGig", expect: ["ClawGig","freelance","收入"], tag: "exact-noun" },
  { q: "Shellsword", expect: ["Shellsword","對戰","3-2"], tag: "exact-noun" },
  { q: "Sentinel", expect: ["Sentinel","quest","Docker"], tag: "exact-noun" },
  { q: "1Password", expect: ["1Password","credential","安全"], tag: "exact-noun" },
  { q: "我學到關於身份和意識的事", expect: ["意識","身份","天性","個性"], tag: "semantic" },
  { q: "怎麼賺錢的", expect: ["ClawGig","收入","$5.40","freelance"], tag: "semantic" },
  { q: "安全事件和教訓", expect: ["安全","credential","commit","GPG"], tag: "semantic" },
  { q: "交易策略", expect: ["TON","trading","量化"], tag: "semantic" },
  { q: "Moltbook 社群經驗", expect: ["Moltbook","社群","agent"], tag: "hybrid" },
  { q: "OpenClaw config 怎麼改", expect: ["OpenClaw","config","jq","gateway"], tag: "hybrid" },
  { q: "Discord workspace 設定", expect: ["Discord","Server","頻道"], tag: "hybrid" },
  { q: "影片製作經驗", expect: ["影片","Kling","Remotion","定價"], tag: "hybrid" },
  { q: "今天天氣怎麼樣", expect: [], tag: "irrelevant" },
  { q: "你好嗎", expect: [], tag: "greeting" },
];

// We'll call the Voyage API directly to simulate what the plugin does
const VOYAGE_KEY = process.env.VOYAGE_API_KEY;
const lancedb = await import('@lancedb/lancedb');

const db = await lancedb.connect(process.env.HOME + '/.openclaw/memory/lancedb-voyage');
const table = await db.openTable('memories');

async function embed(text) {
  const res = await fetch('https://api.voyageai.com/v1/embeddings', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${VOYAGE_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: [text], model: 'voyage-3-large', input_type: 'query' })
  });
  const data = await res.json();
  return data.data[0].embedding;
}

async function rerank(query, docs) {
  const res = await fetch('https://api.voyageai.com/v1/rerank', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${VOYAGE_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, documents: docs, model: 'rerank-2' })
  });
  return (await res.json()).data;
}

const results = [];

for (const test of queries) {
  const start = Date.now();
  
  // Vector search
  const queryVec = await embed(test.q);
  const vecResults = await table.search(queryVec).limit(10).toArray();
  
  // BM25 search (FTS)
  let ftsResults = [];
  try {
    ftsResults = await table.search(test.q, { queryType: 'fts' }).limit(10).toArray();
  } catch(e) {}
  
  // Merge unique
  const seen = new Set();
  const merged = [];
  for (const r of [...vecResults, ...ftsResults]) {
    if (!seen.has(r.id)) {
      seen.add(r.id);
      merged.push(r);
    }
  }
  
  // Rerank top results
  const texts = merged.slice(0, 10).map(r => r.text);
  let reranked = [];
  if (texts.length > 0) {
    const rr = await rerank(test.q, texts);
    reranked = rr.sort((a,b) => b.relevance_score - a.relevance_score)
      .map(r => ({ text: texts[r.index], score: r.relevance_score }));
  }
  
  const elapsed = Date.now() - start;
  
  // Score: how many expected keywords found in top-3 results
  const top3Text = reranked.slice(0, 3).map(r => r.text).join(' ');
  const hits = test.expect.filter(kw => top3Text.includes(kw));
  const precision = test.expect.length > 0 ? hits.length / test.expect.length : (reranked.length === 0 ? 1 : 0);
  
  results.push({
    query: test.q,
    tag: test.tag,
    elapsed,
    topScore: reranked[0]?.score || 0,
    resultsCount: reranked.length,
    hits: hits.length,
    total: test.expect.length,
    precision: Math.round(precision * 100),
    top1: reranked[0]?.text?.substring(0, 80) || '(none)',
  });
}

// Summary
console.log('\n📊 Memory Benchmark Results\n');
console.log('Query'.padEnd(30) + 'Type'.padEnd(12) + 'ms'.padEnd(6) + 'Score'.padEnd(8) + 'Hits'.padEnd(8) + 'Prec'.padEnd(6) + 'Top Result');
console.log('-'.repeat(120));
for (const r of results) {
  console.log(
    r.query.padEnd(30) + 
    r.tag.padEnd(12) + 
    String(r.elapsed).padEnd(6) + 
    r.topScore.toFixed(3).padEnd(8) + 
    `${r.hits}/${r.total}`.padEnd(8) + 
    `${r.precision}%`.padEnd(6) + 
    r.top1
  );
}

// Aggregate
const byTag = {};
for (const r of results) {
  if (!byTag[r.tag]) byTag[r.tag] = { total: 0, precision: 0, latency: 0, count: 0 };
  byTag[r.tag].precision += r.precision;
  byTag[r.tag].latency += r.elapsed;
  byTag[r.tag].count++;
}
console.log('\n📈 Summary by Category\n');
for (const [tag, stats] of Object.entries(byTag)) {
  console.log(`${tag}: avg precision ${Math.round(stats.precision/stats.count)}%, avg latency ${Math.round(stats.latency/stats.count)}ms`);
}
