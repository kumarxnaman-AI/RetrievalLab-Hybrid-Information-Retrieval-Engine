# Benchmark

Dataset: datasets/scifact (5183 docs, 19339 chunks, 300 queries)
Chunking: size=500, overlap=50 | BM25: k1=1.5, b=0.75 | Dense model: sentence-transformers/all-MiniLM-L6-v2

| Retrieval | Recall@10 | Precision@10 | HitRate@10 | MRR@10 | nDCG@10 | p50 latency | p95 latency | Index size | Build time |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.756 | 0.083 | 0.777 | 0.589 | 0.624 | 2.20 ms | 6.21 ms | 6.47 MB | 0.5 s |
| Dense Exact | 0.809 | 0.091 | 0.817 | 0.633 | 0.673 | 16.57 ms | 23.54 ms | 29.70 MB | 400.6 s |
