# HNSW vs exact search

Dataset: datasets/scifact (19339 chunk vectors, dim=384, 300 queries)
HNSW: M=16, efConstruction=200, metric=cosine, layers=4, avg layer-0 degree=26.0
ANN recall is the overlap of HNSW's top-10 chunks with the exact top-10 chunks. nDCG@10 is document-level, computed from those top-10 chunks.
Latency is vector search only (queries are embedded beforehand), single thread.

| Search | ANN Recall@10 | nDCG@10 | p50 | p95 | Queries/sec | Build time | Memory |
|---|---|---|---|---|---|---|---|
| Exact KNN | 1.000 | 0.657 | 2.04 ms | 3.87 ms | 443 | 0.0 s | 29.7 MB |
| HNSW ef=10 | 0.908 | 0.637 | 0.43 ms | 1.14 ms | 1922 | 139.0 s | 49.9 MB |
| HNSW ef=20 | 0.959 | 0.646 | 0.47 ms | 1.19 ms | 1755 | 139.0 s | 49.9 MB |
| HNSW ef=50 | 0.989 | 0.653 | 0.90 ms | 1.50 ms | 989 | 139.0 s | 49.9 MB |
| HNSW ef=100 | 0.997 | 0.657 | 1.73 ms | 3.03 ms | 497 | 139.0 s | 49.9 MB |
| HNSW ef=200 | 0.999 | 0.657 | 3.34 ms | 6.05 ms | 260 | 139.0 s | 49.9 MB |
