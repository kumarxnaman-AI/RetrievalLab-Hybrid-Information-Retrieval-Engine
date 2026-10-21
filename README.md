# RetrievalLab-Hybrid-Information-Retrieval-Engine
Built a search engine implementing BM25 inverted indexing, HNSW approximate nearest-neighbor search, reciprocal-rank fusion and cross-encoder reranking. Developed an evaluation framework comparing sparse, dense and hybrid retrieval using Recall@K, MRR, nDCG and latency benchmarks, with an interactive dashboard for inspecting retrieval behavior.

Everything the retrieval stack depends on is written from scratch in this repo: the inverted index, BM25
scoring, cosine similarity and exact KNN, the HNSW graph, the fusion methods, and the evaluation metrics.
Only the embedding model and the cross-encoder come from existing libraries.

## Pipeline

```
                        Query
                          |
            +-------------+--------------+
            v                            v
   Custom BM25 (inverted index)   Dense retrieval (MiniLM embeddings)
                                  exact KNN  |  our HNSW graph
            |                            |
            +-------------+--------------+
                          v
        Fusion: RRF | weighted RRF | normalized score fusion
                          v
              Cross-encoder reranker (optional)
                          v
                     Ranked documents
```

## Results

All numbers are on [BEIR SciFact](https://github.com/beir-cellar/beir): 5,183 scientific abstracts
(split into 19,339 chunks of 500 characters, 50 overlap) and 300 labeled test queries. Single thread, CPU.
Quality is measured at the document level; chunk results are collapsed to their best chunk per document.
The full tables are in [results/](results/).

### 1. BM25 vs exact dense search

| Retrieval | Recall@10 | MRR@10 | nDCG@10 | p95 latency | Index size |
|---|---|---|---|---|---|
| BM25 (k1=1.5, b=0.75) | 0.756 | 0.589 | 0.624 | 6.2 ms | 6.5 MB |
| Dense exact (all-MiniLM-L6-v2) | 0.809 | 0.633 | 0.673 | 23.5 ms* | 29.7 MB |

\*includes embedding the query. BM25 is the stronger choice on speed, size and build time (0.5 s vs 400 s
to embed the corpus); dense retrieval is about 5 points better on nDCG.

### 2. HNSW (from scratch) vs exact KNN

M=16, efConstruction=200, cosine. ANN recall = overlap of HNSW's top-10 with the exact top-10. Vector search
only (queries are pre-embedded).

| Search | ANN Recall@10 | p50 | p95 | Queries/sec | Build | Memory |
|---|---|---|---|---|---|---|
| Exact KNN | 1.000 | 2.04 ms | 3.87 ms | 443 | - | 29.7 MB |
| HNSW ef=10 | 0.908 | 0.43 ms | 1.14 ms | 1922 | 139 s | 49.9 MB |
| HNSW ef=20 | 0.959 | 0.47 ms | 1.19 ms | 1755 | 139 s | 49.9 MB |
| HNSW ef=50 | 0.989 | 0.90 ms | 1.50 ms | 989 | 139 s | 49.9 MB |
| HNSW ef=100 | 0.997 | 1.73 ms | 3.03 ms | 497 | 139 s | 49.9 MB |
| HNSW ef=200 | 0.999 | 3.34 ms | 6.05 ms | 260 | 139 s | 49.9 MB |

![HNSW recall vs latency](results/hnsw_recall_vs_latency.png)

**Reading this honestly:** the graph delivers the expected recall/latency trade-off (ef=20 gives 0.96 recall at
about 4x the throughput of exact search). But at ef=100 and above it is no faster than brute force. At 19k vectors,
exact search is a single NumPy matrix multiply, which is very hard to beat, and this HNSW is pure Python. HNSW's
advantage is that its cost grows roughly logarithmically with corpus size while brute force grows linearly,
so it should pay off at much larger scale. This repo does not demonstrate that scale.
It also costs about 1.7x the memory and 139 s to build.

### 3. Hybrid retrieval

BM25 + our HNSW, top 100 documents from each. Fusion settings (RRF k, weights, alpha) were **tuned on SciFact's
separate train queries** and evaluated on the test queries, so tuning does not leak into the scores.

| Method | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.677 | 0.756 | 0.589 | 0.624 |
| Dense (exact) | 0.723 | 0.809 | 0.633 | 0.673 |
| Dense (HNSW) | 0.723 | 0.809 | 0.633 | 0.673 |
| RRF (k=60) | 0.743 | 0.814 | 0.643 | 0.680 |
| Weighted RRF (tuned: k=10, BM25 0.3 / dense 0.7) | 0.749 | 0.827 | 0.657 | 0.695 |
| Score fusion (tuned: alpha=0.6) | 0.759 | 0.829 | 0.657 | 0.694 |

Fusion beats either retriever alone, by about +0.7 nDCG points untuned and +2.2 points tuned over dense search.
HNSW at ef=100 retrieves the same candidates as exact search, so swapping it in costs no quality here.
End-to-end hybrid latency (including query embedding): p50 29 ms, p95 42 ms.

### 4. Cross-encoder reranking

Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top N hybrid candidates. **Measured on a 40-query
sample** because the cross-encoder runs at roughly 10 pairs/second on this CPU.

| System | nDCG@10 | MRR@10 | p95 latency |
|---|---|---|---|
| Hybrid (RRF) | 0.805 | 0.783 | 48 ms |
| Hybrid + reranker (top 10) | 0.785 | 0.755 | 1,165 ms |
| Hybrid + reranker (top 20) | 0.779 | 0.747 | 2,312 ms |
| Hybrid + reranker (top 50) | 0.771 | 0.744 | 6,089 ms |
| Hybrid + reranker (top 100) | 0.771 | 0.744 | 10,485 ms |

**On this setup reranking did not help**: nDCG fell 2 to 4 points while latency rose from tens of milliseconds
to seconds. With 40 queries a difference this size is within noise (6 queries improved, 8 got worse, 26 unchanged),
so the fair conclusion is "no evidence of a gain", not "reranking hurts". A likely reason is domain mismatch:
this model is trained on web search passages, while SciFact is scientific claim retrieval. A full 300-query run
(`python -m searchlab.rerank_benchmark`) takes roughly an hour on this machine.

### 5. LLM query transformations (Phase 6)

Each of 100 randomly sampled SciFact test queries is transformed by an LLM (`qwen/qwen3.8-27b` through Groq's free
tier), then retrieved with 500-char chunks. nDCG@10, with the change from the original query in brackets:

| Query sent to the retrievers | BM25 | Dense | Hybrid (RRF) |
|---|---|---|---|
| Original | 0.603 | 0.605 | 0.645 |
| LLM rewrite | 0.631 (+0.028) | 0.617 (+0.011) | 0.650 (+0.006) |
| **HyDE** (dense embeds an LLM-written hypothetical abstract) | 0.603 (BM25 unchanged) | **0.705 (+0.100)** | **0.710 (+0.065)** |
| Expansion (+8 related terms) | 0.623 (+0.020) | 0.671 (+0.066) | 0.680 (+0.035) |

HyDE was the clear winner: with hybrid retrieval it helped 30 queries and hurt 4. Expansion also helped on average but
hurt more queries (24 helped, 18 hurt). Plain rewriting was close to neutral overall (11 helped, 11 hurt). Each
transformation costs one extra LLM call per query (1 to 3 seconds in my tests, and rate-limited on a free tier). One
model, one dataset and 100 queries: treat gaps of a point or two as noise. SciFact queries are full claim sentences and
its documents are abstracts, which plausibly suits HyDE, but that explanation was not tested. Full table:
[results/query_transform_benchmark.md](results/query_transform_benchmark.md).

### 6. When does query rewriting hurt? (Phase 10)

For the hybrid system, the rewrite changed nDCG@10 by more than 0.01 on 22 of 100 queries: **11 helped (mean +0.32),
11 harmed (mean -0.27)**, and 78 were unchanged. I tested five suspected causes with simple heuristics (an entity, a
constraint word or a number disappearing, a generalized rewrite, or a large embedding shift). **None of them predicted
harm**: the harm rate was about 11% whether a query was flagged or not (0% to 20% across flags, from groups of 4 to 34
queries). The most harmed rewrites mostly turned a full claim sentence into a keyword list ("Autophagy declines in aged
organisms." became "autophagy decline aging"), but so did several of the most helped. So in this experiment, rewriting
looks high-variance rather than systematically harmful, and the heuristics did not find a rule for when it hurts. That
is a negative result, and the groups are small. Details and examples: [results/rewrite_analysis.md](results/rewrite_analysis.md).

### 7. Grounded RAG answers (Phase 8)

`POST /answer` (and the dashboard's Ask tab) retrieves with the hybrid settings, gives the LLM numbered sources within
a token budget, and then checks every answer sentence for a valid citation that supports it. A real run on SciFact:

- *"Do statins reduce the risk of cardiovascular disease?"* gave a 3-sentence answer citing sources [1] and [2], with
  100% of sentences supported by their citations, using 615 of 1,500 context tokens.
- *"What is the capital of France?"* and a BRCA1-mechanism question the retrieved abstracts do not answer both got
  "I cannot answer from the provided sources." instead of an invented answer.

The grounding check is lexical (at least half of a sentence's content words must appear in the passages it cites). It
catches uncited and clearly unsupported claims, not subtle paraphrase errors. I tried three questions by hand; this is
a demo, not an evaluation of answer quality.

### 8. Chunking strategies (Phase 7)

Four strategies, each crossed with BM25, exact dense search and hybrid RRF. This ran at **demo scale**: a 1,500-document
corpus (every relevant document for 60 sampled test queries plus random distractors), so scores are higher than on the full
corpus and should only be compared with each other. nDCG@10:

| | fixed | sliding (50% overlap) | recursive (sentence packing) | semantic (topic-shift cuts) |
|---|---|---|---|---|
| BM25 | 0.668 | 0.669 | 0.663 | 0.673 |
| Dense | 0.712 | 0.731 | **0.745** | 0.705 |
| Hybrid | **0.724** | 0.712 | **0.728** | 0.716 |

**The honest reading: chunking barely mattered here.** The spread across strategies is 0.01 for BM25, 0.04 for dense and
0.016 for hybrid, and with 60 queries gaps this size are within noise. Sentence-aware `recursive` chunking looked best
for dense search (+0.033 over fixed windows), but the more sophisticated `semantic` chunking was the worst for dense
and no better than plain fixed windows for hybrid. SciFact abstracts are single paragraphs, so `recursive` can only
pack sentences here, and it would need documents with real structure to show its value. Full tables, including chunk
statistics: [results/chunking_benchmark.md](results/chunking_benchmark.md).

### 9. Two datasets and failure analysis (Phase 9)

The same pipeline (500-char chunks, MiniLM, HNSW M=16 / efSearch=100, RRF k=60) on SciFact and on
[NFCorpus](https://www.cl.uni-heidelberg.de/statnlp/research/nfcorpus/) (3,633 medical and nutrition documents, 323 test
queries, many of them short keyword queries).

| System | SciFact nDCG@10 | NFCorpus nDCG@10 | NFCorpus p50 / p95 latency | NFCorpus index | NFCorpus build |
|---|---|---|---|---|---|
| BM25 | 0.624 | 0.291 | 1.1 / 3.4 ms | 4.9 MB | 0.5 s |
| Dense (exact) | 0.673 | 0.323 | 12.6 / 22.3 ms | 21.9 MB | 340 s |
| Dense (HNSW) | 0.673 | 0.324 | 13.3 / 20.1 ms | 36.9 MB | 80 s |
| **Hybrid (RRF)** | **0.680** | **0.338** | 21.8 / 32.5 ms | both | - |

Hybrid has the best average on both datasets, and the ranking BM25 < dense < hybrid is the same on each. Latency includes
embedding the query; throughput for the same runs is in the full report. On SciFact, HNSW was slower than exact search end
to end (26 vs 18 ms p50), which matches the Phase 2 finding that this pure-Python HNSW does not beat NumPy brute force
at this scale.

**Where each system wins** (per-query nDCG@10 gap of at least 0.1):

| | SciFact (300 queries) | NFCorpus (323 queries) |
|---|---|---|
| BM25 clearly better than dense | 17% | 15% |
| Dense clearly better than BM25 | 28% | 25% |
| Hybrid clearly better than both | 4% | 6% |
| Hybrid clearly worse than the better single retriever | 20% | 18% |
| Every system scores under 0.1 | 10% | 28% |

The "hybrid worse than the better single retriever" row compares hybrid against the per-query best of two systems,
which is a demanding bar: fusion is a compromise, so it rarely matches the best one on every query. Its gain comes from
being decent on both kinds of query. Some patterns I could see:

- **Short keyword queries favor BM25 on NFCorpus**: BM25 wins averaged 2.6 words versus 4.2 for dense wins ("kohlrabi",
  "maple syrup", "cadaverine" were BM25 wins; "viral infections" and "Is Milk Good for Our Bones?" were dense wins).
- **The "acronym / number / entity queries favor BM25" hypothesis was not supported.** On SciFact, 62% of BM25-win
  queries contained such a term versus 64% of all queries (on NFCorpus the heuristic is noisy because titles are in
  title case).
- **Reranking** (top-20 pool, first 15 queries per dataset): it hurt on SciFact (2 improved, 4 hurt, mean change -0.03)
  and helped on NFCorpus (10 improved, 5 hurt, mean +0.05), at about 2 seconds of added CPU latency per query. Fifteen
  queries is too few to be confident, but it is consistent with the earlier 40-query SciFact result.
- NFCorpus is hard: 28% of queries are missed by every system.

Not done: a "real-world" demo dataset such as government schemes. That needs real documents and relevance labels, which
this repo does not contain; a dataset in BEIR format can be added under `datasets/<name>` and passed with `--datasets`.
Failure analysis for query rewriting is in section 6. Full report with examples:
[results/multi_dataset_benchmark.md](results/multi_dataset_benchmark.md).

## Run it

Requires Python 3.10+. From the project root, using a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Download the benchmark dataset (not committed)
Invoke-WebRequest https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -OutFile datasets\scifact.zip
Expand-Archive datasets\scifact.zip -DestinationPath datasets

.venv\Scripts\python.exe -m pytest                       # 177 tests, no model downloads or API key needed
.venv\Scripts\python.exe -m searchlab.demo "your query"  # BM25 vs dense side by side
```

The first run that touches the models downloads them, and the first corpus embedding takes several minutes on CPU
(it is cached in `.cache/` afterwards, as is the HNSW graph).

Experiments (each writes a report to `results/`):

```powershell
.venv\Scripts\python.exe -m searchlab.benchmark          # BM25 vs exact dense
.venv\Scripts\python.exe -m searchlab.hnsw_benchmark     # HNSW vs exact KNN (+ graph)
.venv\Scripts\python.exe -m searchlab.hybrid_benchmark   # BM25 / dense / RRF / weighted RRF / score fusion
.venv\Scripts\python.exe -m searchlab.rerank_benchmark --max-queries 40 --latency-queries 10
```

Dashboard (two terminals):

```powershell
.venv\Scripts\python.exe -m uvicorn searchlab.api:app --port 8765
.venv\Scripts\python.exe -m streamlit run src/searchlab/dashboard.py
```

The dashboard shows BM25, dense, hybrid and hybrid + reranker side by side with scores and latency, and lets
you change top-K, BM25 `k1`/`b`, HNSW `efSearch`, exact vs HNSW, fusion method, RRF `k`, weights, alpha, and
reranking on/off. The "Ask (RAG)" tab answers a question with numbered citations and flags unsupported sentences
(needs the LLM `.env` described below). The last tab shows the experiment tables and the recall-vs-latency graph. The API is
configurable with `SEARCHLAB_DATASET` (default `datasets/scifact`); the dashboard with `SEARCHLAB_API_URL`.

## Code map

| Piece | File |
|---|---|
| Config, data models, loaders, cleaning, chunking | `config.py`, `models.py`, `loaders.py`, `cleaning.py`, `chunking.py` |
| Tokenizer, inverted index (postings, TF, DF, IDF) | `tokenizer.py`, `index/inverted_index.py` |
| BM25 | `retrieval/bm25.py` |
| Embeddings, cosine similarity, exact top-K | `retrieval/dense.py` |
| HNSW graph (layers, greedy + beam search, neighbor selection) | `index/hnsw.py`, `retrieval/hnsw_retriever.py` |
| RRF, weighted RRF, score fusion, hybrid retriever | `retrieval/fusion.py`, `retrieval/hybrid.py` |
| Cross-encoder reranker | `retrieval/reranker.py` |
| Metrics (Recall, Precision, MRR, nDCG, Hit Rate) and dataset loader | `evaluation/metrics.py`, `evaluation/dataset.py` |
| Engine, API, dashboard | `engine.py`, `api.py`, `dashboard.py` |
| Chunking strategies and experiment (Phase 7) | `chunkers.py`, `chunking_benchmark.py` |
| LLM client, query transformations, experiment, harm analysis (Phases 6, 10) | `llm.py`, `query_transform.py`, `query_benchmark.py`, `rewrite_analysis.py` |
| RAG answers with citations and grounding check (Phase 8) | `rag.py` |
| Multi-dataset benchmark and failure analysis (Phase 9) | `multi_benchmark.py` |

## Status

| Phase | |
|---|---|
| 1. BM25 + exact dense + evaluation | done |
| 2. HNSW from scratch | done |
| 3. Hybrid search (RRF, weighted RRF, score fusion) | done |
| 4. Cross-encoder reranking | done (measured on a 40-query sample) |
| 5. FastAPI + Streamlit dashboard | done |
| 6. Query rewriting, HyDE, expansion | done (100 queries, one LLM) |
| 7. Chunking experiments | done at demo scale (1,500 documents, 60 queries) |
| 8. RAG answer layer | done; tried by hand on 3 real questions (not a quality evaluation) |
| 9. Multi-dataset benchmarking and failure analysis | done for SciFact and NFCorpus; no real-world demo dataset |
| 10. Research extension: when does query rewriting hurt? | done: the heuristics did not predict harm (negative result) |

This is a learning and demo project, so several experiments were deliberately run at small scale (see the limitations).

### Reproducing phases 6 to 10

```powershell
# Phase 7 chunking (demo scale by default: 1,500 documents, 60 queries; use --max-docs 0 --max-queries 0 for the full set)
.venv\Scripts\python.exe -m searchlab.chunking_benchmark

# Phase 9: SciFact + NFCorpus (download NFCorpus the same way as SciFact, from .../datasets/nfcorpus.zip).
# Run it alone, since it measures latency.
.venv\Scripts\python.exe -m searchlab.multi_benchmark --rerank-queries 15 --latency-queries 50

# Phases 6, 8, 10 need an LLM. Create a git-ignored .env file in the project root:
#      LLM_API_KEY=...            (a free Groq or Google AI Studio key works; never commit it)
#      LLM_BASE_URL=https://api.groq.com/openai/v1
#      LLM_MODEL=qwen/qwen3.8-27b
.venv\Scripts\python.exe -m searchlab.query_benchmark --max-queries 100   # Phase 6 (every LLM reply is cached)
.venv\Scripts\python.exe -m searchlab.rewrite_analysis                    # Phase 10
```

The LLM client works with any OpenAI-compatible endpoint. Model names on free tiers change: if you get "model not found",
list the available ones with a `GET {LLM_BASE_URL}/models` request. Avoid small "reasoning" models (for example
`openai/gpt-oss-20b`): they spend the short token budget thinking and return empty answers. Free tiers are rate-limited,
so the Phase 6 run (about 300 LLM calls) took roughly 20 minutes the first time and is instant from the cache afterwards.
Only public benchmark data is sent to the LLM.

## Known limitations

- Two public datasets, both biomedical. Conclusions may not transfer to other domains.
- The HNSW is pure Python: correct and tested against brute force, but built for learning, not production speed.
- Demo-scale experiments: reranking was measured on 15 to 40 queries, chunking on a 1,500-document corpus with 60
  queries, query rewriting on 100 queries with one LLM. Gaps of a point or two are within noise.
- The RAG grounding check is lexical, and the RAG layer was only tried on 3 questions.
- Latencies are single-threaded CPU timings on one machine and vary run to run (p95 especially, over few queries).
- No real-world demo dataset (it needs documents and relevance labels that this repo does not contain).
