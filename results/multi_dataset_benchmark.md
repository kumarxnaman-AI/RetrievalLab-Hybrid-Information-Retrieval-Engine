# Multi-dataset benchmark and failure analysis

Same code, same settings (500-char chunks, MiniLM embeddings, HNSW M=16/efConstruction=200/efSearch=100, RRF k=60) on every dataset. Quality is document-level at the top 10.

## Summary

| System | scifact nDCG@10 | nfcorpus nDCG@10 |
|---|---|---|
| BM25 | 0.624 | 0.291 |
| Dense (exact) | 0.673 | 0.323 |
| Dense (HNSW) | 0.673 | 0.324 |
| Hybrid (RRF) | 0.680 | 0.338 |

## scifact

5183 documents (19339 chunks), 300 test queries. Latency: first 50 queries, end to end
(including query embedding), single thread, CPU. Memory is the size of each index structure.

| System | Recall@10 | MRR@10 | nDCG@10 | p50 | p95 | Queries/sec | Index memory | Build time |
|---|---|---|---|---|---|---|---|---|
| BM25 | 0.756 | 0.589 | 0.624 | 3.2 ms | 9.0 ms | 259 | 6.5 MB | 0.9 s |
| Dense (exact) | 0.809 | 0.633 | 0.673 | 18.3 ms | 32.3 ms | 40 | 29.7 MB | n/a (cached) |
| Dense (HNSW) | 0.809 | 0.633 | 0.673 | 26.0 ms | 99.9 ms | 27 | 49.9 MB | 139.0 s |
| Hybrid (RRF) | 0.814 | 0.643 | 0.680 | 28.5 ms | 41.4 ms | 33 | BM25 + HNSW | - |

### Where each system wins (per-query nDCG@10, clear = gap of at least 0.1)

| Pattern | Queries | Share |
|---|---|---|
| BM25 wins | 52 | 17% |
| Dense wins | 83 | 28% |
| Hybrid wins | 12 | 4% |
| Hybrid loses to the best single retriever | 61 | 20% |
| All systems fail | 29 | 10% |

**BM25 wins, largest gaps:**
- "Activation of PPM1D suppresses p53 function." (BM25 1.00, dense 0.00, hybrid 0.57)
- "Arterioles have a larger lumen diameter than venules." (BM25 1.00, dense 0.00, hybrid 0.32)
- "In transgenic mice harboring green florescent protein under the control of the Sox2 promoter, less than ten pe" (BM25 1.00, dense 0.00, hybrid 0.43)
- "Insulin increases risk of severe kidney failure." (BM25 1.00, dense 0.00, hybrid 0.50)

**Dense wins, largest gaps:**
- "AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in the lungs." (BM25 0.00, dense 1.00, hybrid 0.36)
- "Activator-inhibitor pairs are provided dorsally by Admpchordin." (BM25 0.00, dense 1.00, hybrid 1.00)
- "Hypertension is frequently observed in type 1 diabetes patients." (BM25 0.00, dense 1.00, hybrid 0.00)
- "Integrating classroom-based collaborative learning with Web-based collaborative learning leads to subpar class" (BM25 0.00, dense 1.00, hybrid 0.63)

**Hybrid wins, largest gaps:**
- "CCL19 is absent within dLNs." (BM25 0.00, dense 0.00, hybrid 0.43)
- "IFIT1 restricts viral replication by sequestrating mis-capped viral RNAs." (BM25 0.00, dense 0.00, hybrid 0.43)
- "Glycolysis is one of the primary glycometabolic pathways in cells." (BM25 0.00, dense 0.00, hybrid 0.39)
- "Cells lacking clpC have a defect in sporulation efficiency in Bacillus subtilis." (BM25 0.63, dense 0.63, hybrid 1.00)

**Hybrid loses to the best single retriever:**
- "Hypertension is frequently observed in type 1 diabetes patients." (BM25 0.00, dense 1.00, hybrid 0.00)
- "The myocardial lineage develops from cardiac progenitors of mesodermal origin." (BM25 0.00, dense 1.00, hybrid 0.00)
- "The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune events." (BM25 0.00, dense 1.00, hybrid 0.00)
- "The treatment of cancer patients with co-IR blockade precipitates adverse autoimmune events." (BM25 0.00, dense 1.00, hybrid 0.00)

### Query characteristics behind those wins

| Queries where... | Count | With an acronym / number / entity | Mean words |
|---|---|---|---|
| BM25 wins | 52 | 32/52 | 10.9 |
| Dense wins | 83 | 53/83 | 13.4 |
| All queries | 300 | 191/300 | 12.6 |

### Where reranking hurts (top-20 pool, first 15 queries)

Mean nDCG@10 of hybrid on this sample: 0.686; after reranking, queries improved: 2, hurt: 4, unchanged: 9.
Mean Δ nDCG@10: -0.030. Reranking added 1973 ms per query on average.

Worst cases:
- "Activation of PPM1D suppresses p53 function." (Δ nDCG -0.57)
- "ALDH1 expression is associated with better breast cancer outcomes." (Δ nDCG -0.37)
- "A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection." (Δ nDCG -0.13)

## nfcorpus

3633 documents (14252 chunks), 323 test queries. Latency: first 50 queries, end to end
(including query embedding), single thread, CPU. Memory is the size of each index structure.

| System | Recall@10 | MRR@10 | nDCG@10 | p50 | p95 | Queries/sec | Index memory | Build time |
|---|---|---|---|---|---|---|---|---|
| BM25 | 0.143 | 0.488 | 0.291 | 1.1 ms | 3.4 ms | 748 | 4.9 MB | 0.5 s |
| Dense (exact) | 0.160 | 0.508 | 0.323 | 12.6 ms | 22.3 ms | 71 | 21.9 MB | 339.6 s |
| Dense (HNSW) | 0.161 | 0.509 | 0.324 | 13.3 ms | 20.1 ms | 72 | 36.9 MB | 79.7 s |
| Hybrid (RRF) | 0.169 | 0.547 | 0.338 | 21.8 ms | 32.5 ms | 40 | BM25 + HNSW | - |

### Where each system wins (per-query nDCG@10, clear = gap of at least 0.1)

| Pattern | Queries | Share |
|---|---|---|
| BM25 wins | 47 | 15% |
| Dense wins | 80 | 25% |
| Hybrid wins | 19 | 6% |
| Hybrid loses to the best single retriever | 57 | 18% |
| All systems fail | 90 | 28% |

**BM25 wins, largest gaps:**
- "kohlrabi" (BM25 1.00, dense 0.00, hybrid 1.00)
- "maple syrup" (BM25 1.00, dense 0.50, hybrid 0.69)
- "cadaverine" (BM25 0.64, dense 0.15, hybrid 0.61)
- "Pharmacists Versus Health Food Store Employees: Who Gives Better Advice?" (BM25 0.48, dense 0.00, hybrid 0.33)

**Dense wins, largest gaps:**
- "viral infections" (BM25 0.20, dense 1.00, hybrid 0.60)
- "Too Much Iodine Can Be as Bad as Too Little" (BM25 0.17, dense 0.75, hybrid 0.57)
- "cocaine" (BM25 0.22, dense 0.79, hybrid 0.79)
- "Is Milk Good for Our Bones?" (BM25 0.17, dense 0.67, hybrid 0.44)

**Hybrid wins, largest gaps:**
- "Neurobiology of Artificial Sweeteners" (BM25 0.40, dense 0.40, hybrid 0.64)
- "Does Cholesterol Size Matter?" (BM25 0.36, dense 0.14, hybrid 0.60)
- "Sexually Transmitted Fish Toxin" (BM25 0.08, dense 0.15, hybrid 0.33)
- "What Do Meat Purge and Cola Have in Common?" (BM25 0.13, dense 0.19, hybrid 0.35)

**Hybrid loses to the best single retriever:**
- "ultra-processed foods" (BM25 0.45, dense 0.00, hybrid 0.00)
- "viral infections" (BM25 0.20, dense 1.00, hybrid 0.60)
- "Apthous Ulcer Mystery Solved" (BM25 0.00, dense 0.47, hybrid 0.10)
- "Preventing Cataracts with Diet" (BM25 0.00, dense 0.50, hybrid 0.14)

### Query characteristics behind those wins

| Queries where... | Count | With an acronym / number / entity | Mean words |
|---|---|---|---|
| BM25 wins | 47 | 18/47 | 2.6 |
| Dense wins | 80 | 50/80 | 4.2 |
| All queries | 323 | 154/323 | 3.3 |

### Where reranking hurts (top-20 pool, first 15 queries)

Mean nDCG@10 of hybrid on this sample: 0.374; after reranking, queries improved: 10, hurt: 5, unchanged: 0.
Mean Δ nDCG@10: +0.048. Reranking added 1918 ms per query on average.

Worst cases:
- "What Do Meat Purge and Cola Have in Common?" (Δ nDCG -0.14)
- "Who Should be Careful About Curcumin?" (Δ nDCG -0.10)
- "What is Actually in Chicken Nuggets?" (Δ nDCG -0.07)
