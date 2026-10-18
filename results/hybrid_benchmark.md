# Hybrid retrieval experiment

Dataset: datasets/scifact - 5183 documents, 300 test queries, 809 train queries.
Each retriever contributes its top 100 documents; the dense side is our HNSW index
(M=16, efConstruction=200) unless stated. Fusion settings were **tuned on the train queries and
reported on the test queries**.

Tuned weighted RRF: `{'method': 'weighted_rrf', 'rrf_k': 10, 'bm25_weight': 0.3, 'dense_weight': 0.7}` (train nDCG@10 0.702)
Tuned score fusion: `{'method': 'score_fusion', 'alpha': 0.6}` (train nDCG@10 0.718)

## Main comparison (test queries)

| Method | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.677 | 0.756 | 0.589 | 0.624 |
| Dense (exact) | 0.723 | 0.809 | 0.633 | 0.673 |
| Dense (HNSW) | 0.723 | 0.809 | 0.633 | 0.673 |
| RRF (k=60) | 0.743 | 0.814 | 0.643 | 0.680 |
| Weighted RRF (tuned) | 0.749 | 0.827 | 0.657 | 0.695 |
| Score fusion (tuned) | 0.759 | 0.829 | 0.657 | 0.694 |

## RRF k sweep (test queries)

| RRF k | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| 1 | 0.754 | 0.820 | 0.657 | 0.694 |
| 10 | 0.736 | 0.838 | 0.646 | 0.688 |
| 30 | 0.736 | 0.828 | 0.648 | 0.685 |
| 60 | 0.743 | 0.814 | 0.643 | 0.680 |
| 100 | 0.746 | 0.810 | 0.642 | 0.678 |
| 200 | 0.746 | 0.801 | 0.641 | 0.675 |

## Score fusion alpha sweep

| alpha (dense weight) | train nDCG@10 | test nDCG@10 |
|---|---|---|
| 0.0 | 0.632 | 0.624 |
| 0.1 | 0.647 | 0.643 |
| 0.2 | 0.664 | 0.658 |
| 0.3 | 0.681 | 0.669 |
| 0.4 | 0.693 | 0.683 |
| 0.5 | 0.711 | 0.694 |
| 0.6 | 0.718 | 0.694 |
| 0.7 | 0.715 | 0.696 |
| 0.8 | 0.708 | 0.698 |
| 0.9 | 0.695 | 0.687 |
| 1.0 | 0.684 | 0.673 |

## End-to-end latency (100 test queries, includes query embedding, single thread)

| Pipeline | p50 | p95 |
|---|---|---|
| BM25 | 2.7 ms | 6.5 ms |
| Dense (HNSW, ef=100) | 17.3 ms | 21.1 ms |
| Hybrid RRF (BM25 + HNSW) | 29.4 ms | 42.1 ms |
