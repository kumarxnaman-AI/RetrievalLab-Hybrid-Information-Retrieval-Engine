# Query transformation experiment

Dataset: datasets/scifact, a random sample of 100 test queries (seed 0). LLM: `qwen/qwen3.8-27b`.
Retrieval: 500-char chunks, BM25 + exact dense (MiniLM) + RRF hybrid (k=60). Quality at the document level.
"helped / neutral / harmed" counts queries whose nDCG@10 changed by more than 0.01 versus the original query.

- **original**: the query unchanged.
- **rewrite**: the LLM rewrites the query for both retrievers.
- **hyde**: the dense retriever embeds an LLM-written hypothetical passage; BM25 keeps the original query
  (so the BM25 rows for `hyde` equal `original` by construction).
- **expansion**: the query plus 8 LLM-suggested related terms, for both retrievers.

| System | Query | Recall@10 | MRR@10 | nDCG@10 | Δ nDCG@10 | helped / neutral / harmed |
|---|---|---|---|---|---|---|
| BM25 | original | 0.747 | 0.565 | 0.603 | - | - |
| BM25 | rewrite | 0.822 | 0.579 | 0.631 | +0.028 | 16 / 77 / 7 |
| BM25 | hyde | 0.747 | 0.565 | 0.603 | +0.000 | 0 / 100 / 0 |
| BM25 | expansion | 0.797 | 0.571 | 0.623 | +0.020 | 25 / 50 / 25 |
| Dense | original | 0.761 | 0.559 | 0.605 | - | - |
| Dense | rewrite | 0.762 | 0.574 | 0.617 | +0.011 | 12 / 74 / 14 |
| Dense | hyde | 0.891 | 0.646 | 0.705 | +0.100 | 34 / 49 / 17 |
| Dense | expansion | 0.823 | 0.629 | 0.671 | +0.066 | 27 / 61 / 12 |
| Hybrid | original | 0.794 | 0.603 | 0.645 | - | - |
| Hybrid | rewrite | 0.777 | 0.614 | 0.650 | +0.006 | 11 / 78 / 11 |
| Hybrid | hyde | 0.871 | 0.666 | 0.710 | +0.065 | 30 / 66 / 4 |
| Hybrid | expansion | 0.836 | 0.640 | 0.680 | +0.035 | 24 / 58 / 18 |

With 100 queries, differences of about a point are within noise; look at the helped/harmed counts as well as the mean.
