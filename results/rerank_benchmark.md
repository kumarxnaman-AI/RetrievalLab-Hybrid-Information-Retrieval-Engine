# Reranking experiment

Dataset: datasets/scifact - 5183 documents, 40 test queries.
First stage: BM25 + HNSW fused with RRF (k=60), 100 candidates per retriever.
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Quality uses all 40 test queries; latency uses the first
10 (end-to-end: query text in, top-10 documents out, single thread, CPU).

| System | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | p50 latency | p95 latency | vs. hybrid |
|---|---|---|---|---|---|---|---|
| Hybrid (RRF) | 0.805 | 0.783 | 0.828 | 0.896 | 39 ms | 48 ms | - |
| Hybrid + reranker (top 10) | 0.785 | 0.755 | 0.847 | 0.896 | 1024 ms | 1165 ms | -2.5% nDCG, +1117 ms |
| Hybrid + reranker (top 20) | 0.779 | 0.747 | 0.778 | 0.890 | 2082 ms | 2312 ms | -3.2% nDCG, +2264 ms |
| Hybrid + reranker (top 50) | 0.771 | 0.744 | 0.778 | 0.865 | 5366 ms | 6089 ms | -4.2% nDCG, +6041 ms |
| Hybrid + reranker (top 100) | 0.771 | 0.744 | 0.778 | 0.865 | 9839 ms | 10485 ms | -4.2% nDCG, +10437 ms |

## Headline

Reranking the top 10 hybrid candidates (the best pool size) reduced nDCG@10 from 0.805 to
0.785 (-2.5%), while adding about
1117 ms at p95 (48 ms -> 1165 ms).

**Caveat:** this is a 40-query sample, so differences of a few points are within noise
(per-query: 6 improved, 8 hurt, 26 unchanged). The ms-marco cross-encoder is also trained on web
search, not scientific claims, which is a plausible reason it does not help on SciFact. A full 300-query
run is `python -m searchlab.rerank_benchmark` (roughly an hour on this CPU).

Per-query effect at top 10 (nDCG@10): improved on **6** queries, hurt on **8**,
unchanged on 26.
