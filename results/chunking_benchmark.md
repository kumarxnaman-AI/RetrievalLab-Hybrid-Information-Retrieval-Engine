# Chunking experiment

Dataset: datasets/scifact - **demo scale**: 1500 of 5183 documents (every relevant document for the
60 sampled test queries, plus random distractors) and 60 test queries. A smaller corpus is easier
to search, so absolute scores are higher than on the full corpus; compare strategies with each other, not with other tables.
Target chunk size 500 characters.
Quality is document-level (chunk results collapse to the best chunk per document). Hybrid = BM25 + exact dense,
RRF k=60, 100 candidates each. Dense model: sentence-transformers/all-MiniLM-L6-v2.

Strategies: `fixed` (no overlap), `sliding` (stride = half the window), `recursive` (sentence-packing that respects
paragraph breaks), `semantic` (cuts where adjacent-sentence embedding similarity drops, max 800 chars).
SciFact abstracts are single paragraphs, so `recursive` reduces to sentence-boundary packing here.

## nDCG@10 by system and chunking

| nDCG@10 | fixed | sliding | recursive | semantic |
|---|---|---|---|---|
| BM25 | 0.668 | 0.669 | 0.663 | 0.673 |
| Dense | 0.712 | 0.731 | 0.745 | 0.705 |
| Hybrid | 0.724 | 0.712 | 0.728 | 0.716 |

## Best and worst per system

- **BM25**: best `semantic` (0.673), worst `recursive` (0.663), spread 0.010 nDCG.
- **Dense**: best `recursive` (0.745), worst `semantic` (0.705), spread 0.040 nDCG.
- **Hybrid**: best `recursive` (0.728), worst `sliding` (0.712), spread 0.016 nDCG.

## Chunk statistics

| Strategy | Chunks | Avg chars | p90 chars |
|---|---|---|---|
| fixed | 5239 | 426 | 500 |
| sliding | 8187 | 477 | 500 |
| recursive | 5978 | 373 | 481 |
| semantic | 4731 | 472 | 749 |

## Full metrics

| System | Chunking | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 | fixed | 0.814 | 0.632 | 0.668 |
| BM25 | sliding | 0.814 | 0.633 | 0.669 |
| BM25 | recursive | 0.797 | 0.633 | 0.663 |
| BM25 | semantic | 0.803 | 0.642 | 0.673 |
| Dense | fixed | 0.833 | 0.673 | 0.712 |
| Dense | sliding | 0.833 | 0.699 | 0.731 |
| Dense | recursive | 0.850 | 0.713 | 0.745 |
| Dense | semantic | 0.833 | 0.663 | 0.705 |
| Hybrid | fixed | 0.819 | 0.704 | 0.724 |
| Hybrid | sliding | 0.803 | 0.693 | 0.712 |
| Hybrid | recursive | 0.797 | 0.721 | 0.728 |
| Hybrid | semantic | 0.786 | 0.703 | 0.716 |
