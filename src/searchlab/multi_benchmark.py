"""Phase 9: benchmark every retrieval system on several datasets and analyze failure cases.

Per dataset: quality (Recall/MRR/nDCG), end-to-end latency and throughput, index memory and
build time, then which kinds of queries each system wins or loses on.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from searchlab.config import Config
from searchlab.evaluation.dataset import load_beir_dataset
from searchlab.evaluation.metrics import ndcg_at_k, quality_summary
from searchlab.hybrid_benchmark import table
from searchlab.logging_setup import setup_logging
from searchlab.pipeline import build_hnsw, build_systems
from searchlab.retrieval.aggregate import aggregate_to_documents
from searchlab.retrieval.fusion import fuse_results
from searchlab.retrieval.hybrid import HybridParams, HybridRetriever
from searchlab.retrieval.reranker import CrossEncoderReranker
from searchlab.rewrite_analysis import entity_tokens, words

logger = logging.getLogger(__name__)

TOP = 10
MARGIN = 0.1  # an nDCG@10 gap larger than this counts as a clear win/loss for a query


def categorize(bm25: float, dense: float, hybrid: float, margin: float = MARGIN) -> set[str]:
    """Which clear win/loss patterns does this query show? (labels can overlap)"""
    labels = set()
    if bm25 >= dense + margin:
        labels.add("BM25 wins")
    if dense >= bm25 + margin:
        labels.add("Dense wins")
    if hybrid >= max(bm25, dense) + margin:
        labels.add("Hybrid wins")
    if max(bm25, dense) >= hybrid + margin:
        labels.add("Hybrid loses to the best single retriever")
    if max(bm25, dense, hybrid) < margin:
        labels.add("All systems fail")
    return labels


def has_specific_terms(query: str) -> bool:
    """Does the query contain an acronym, a digit-bearing token or a capitalized entity?"""
    return bool(entity_tokens(query))


def pctl(values, q):
    return float(np.percentile(values, q))


def run_dataset(name: str, path: Path, rerank_queries: int, latency_queries: int):
    logger.info("##### dataset: %s", name)
    config = Config()
    docs, queries, qrels = load_beir_dataset(path, "test")
    doc_texts = {d.doc_id: d.text for d in docs}
    systems = build_systems(docs, config)
    hnsw = build_hnsw(systems.dense, config)
    hybrid = HybridRetriever(systems.bm25, lambda v, n, ef: hnsw.search_vector(v, n, ef), systems.dense.encode)
    params = HybridParams(candidates=100)
    n_chunks = params.candidates * hybrid.CHUNK_OVERFETCH

    qids = list(queries)
    qvecs = systems.dense.encode([queries[q] for q in qids])
    runs = {k: {} for k in ("BM25", "Dense (exact)", "Dense (HNSW)", "Hybrid (RRF)")}
    fused_by_query = {}
    for qid, qvec in zip(qids, qvecs):
        bm25_docs, hnsw_docs = hybrid.candidates(queries[qid], params, qvec)
        exact_docs = aggregate_to_documents(systems.dense.search_vector(qvec, n_chunks), params.candidates)
        fused = fuse_results(bm25_docs, hnsw_docs, method="rrf", rrf_k=60)
        fused_by_query[qid] = fused
        for key, results in (("BM25", bm25_docs), ("Dense (exact)", exact_docs), ("Dense (HNSW)", hnsw_docs), ("Hybrid (RRF)", fused)):
            runs[key][qid] = [r.doc_id for r in results[:TOP]]
    quality = {k: quality_summary(run, qrels) for k, run in runs.items()}
    per_query = {
        k: {qid: ndcg_at_k(run[qid], qrels[qid], TOP) for qid in qids} for k, run in runs.items()
    }

    # End-to-end latency: query text in, top-10 documents out (includes embedding the query).
    sample = [queries[q] for q in qids[:latency_queries]]
    hybrid.search(sample[0], TOP, params)
    systems.dense.search(sample[0], 5)
    latency = {}
    for key, fn in (
        ("BM25", lambda t: aggregate_to_documents(systems.bm25.search(t, 50), TOP)),
        ("Dense (exact)", lambda t: aggregate_to_documents(systems.dense.search(t, 50), TOP)),
        ("Dense (HNSW)", lambda t: aggregate_to_documents(hnsw.search(t, 50, 100), TOP)),
        ("Hybrid (RRF)", lambda t: hybrid.search(t, TOP, params)),
    ):
        times = []
        for text in sample:
            start = time.perf_counter()
            fn(text)
            times.append((time.perf_counter() - start) * 1000)
        latency[key] = times

    # Reranking on a paired sample: the same queries, with and without the cross-encoder.
    reranker = CrossEncoderReranker()
    rerank_ids = qids[:rerank_queries]
    rerank_delta, rerank_ms = {}, []
    for qid in rerank_ids:
        pool = fused_by_query[qid][:20]
        start = time.perf_counter()
        reranked = reranker.rerank(queries[qid], pool, doc_texts, TOP)
        rerank_ms.append((time.perf_counter() - start) * 1000)
        after = ndcg_at_k([r.doc_id for r in reranked], qrels[qid], TOP)
        rerank_delta[qid] = after - per_query["Hybrid (RRF)"][qid]

    return {
        "name": name,
        "docs": len(docs),
        "chunks": len(systems.chunks),
        "queries": queries,
        "qrels": qrels,
        "quality": quality,
        "per_query": per_query,
        "latency": latency,
        "memory": {
            "BM25": systems.bm25.index.size_bytes(),
            "Dense (exact)": systems.dense.size_bytes(),
            "Dense (HNSW)": hnsw.size_bytes(),
        },
        "build": {
            "BM25": systems.bm25_build_seconds,
            "Dense (exact)": systems.dense.embed_seconds,
            "Dense (HNSW)": hnsw.build_seconds,
        },
        "rerank_delta": rerank_delta,
        "rerank_ms": rerank_ms,
        "rerank_hybrid_ndcg": float(np.mean([per_query["Hybrid (RRF)"][q] for q in rerank_ids])),
    }


def fmt_seconds(value):
    return "n/a (cached)" if value is None else f"{value:.1f} s"


def dataset_section(r: dict) -> str:
    systems = list(r["quality"])
    quality_table = table(
        ["System", "Recall@10", "MRR@10", "nDCG@10", "p50", "p95", "Queries/sec", "Index memory", "Build time"],
        [
            [
                s,
                f"{r['quality'][s]['Recall@10']:.3f}",
                f"{r['quality'][s]['MRR@10']:.3f}",
                f"{r['quality'][s]['nDCG@10']:.3f}",
                f"{pctl(r['latency'][s], 50):.1f} ms",
                f"{pctl(r['latency'][s], 95):.1f} ms",
                f"{1000 / np.mean(r['latency'][s]):.0f}",
                f"{r['memory'][s] / 1e6:.1f} MB" if s in r["memory"] else "BM25 + HNSW",
                fmt_seconds(r["build"][s]) if s in r["build"] else "-",
            ]
            for s in systems
        ],
    )

    cats: dict[str, list[str]] = {}
    for qid in r["queries"]:
        labels = categorize(r["per_query"]["BM25"][qid], r["per_query"]["Dense (exact)"][qid], r["per_query"]["Hybrid (RRF)"][qid])
        for label in labels:
            cats.setdefault(label, []).append(qid)
    n = len(r["queries"])
    order = ["BM25 wins", "Dense wins", "Hybrid wins", "Hybrid loses to the best single retriever", "All systems fail"]
    cat_table = table(["Pattern", "Queries", "Share"], [[c, len(cats.get(c, [])), f"{100 * len(cats.get(c, [])) / n:.0f}%"] for c in order])

    def examples(label, limit=4):
        qids = cats.get(label, [])
        key = (lambda q: r["per_query"]["BM25"][q] - r["per_query"]["Dense (exact)"][q]) if label == "BM25 wins" else (
            (lambda q: r["per_query"]["Dense (exact)"][q] - r["per_query"]["BM25"][q]) if label == "Dense wins" else
            (lambda q: abs(r["per_query"]["Hybrid (RRF)"][q] - max(r["per_query"]["BM25"][q], r["per_query"]["Dense (exact)"][q])))
        )
        lines = []
        for q in sorted(qids, key=key, reverse=True)[:limit]:
            pq = r["per_query"]
            lines.append(
                f"- \"{r['queries'][q][:110]}\" (BM25 {pq['BM25'][q]:.2f}, dense {pq['Dense (exact)'][q]:.2f}, hybrid {pq['Hybrid (RRF)'][q]:.2f})"
            )
        return "\n".join(lines) or "_none_"

    bm_q, de_q = cats.get("BM25 wins", []), cats.get("Dense wins", [])

    def share_specific(qids):
        return f"{sum(has_specific_terms(r['queries'][q]) for q in qids)}/{len(qids)}" if qids else "n/a"

    def mean_len(qids):
        return f"{np.mean([len(words(r['queries'][q])) for q in qids]):.1f}" if qids else "n/a"

    pattern_table = table(
        ["Queries where...", "Count", "With an acronym / number / entity", "Mean words"],
        [
            ["BM25 wins", len(bm_q), share_specific(bm_q), mean_len(bm_q)],
            ["Dense wins", len(de_q), share_specific(de_q), mean_len(de_q)],
            ["All queries", n, share_specific(list(r["queries"])), mean_len(list(r["queries"]))],
        ],
    )

    deltas = list(r["rerank_delta"].values())
    helped, hurt = sum(d > 0.01 for d in deltas), sum(d < -0.01 for d in deltas)
    worst = sorted(r["rerank_delta"], key=r["rerank_delta"].get)[:3]
    rerank_lines = "\n".join(f"- \"{r['queries'][q][:110]}\" (Δ nDCG {r['rerank_delta'][q]:+.2f})" for q in worst if r["rerank_delta"][q] < -0.01) or "_none_"

    return f"""## {r['name']}

{r['docs']} documents ({r['chunks']} chunks), {n} test queries. Latency: first {len(r['latency']['BM25'])} queries, end to end
(including query embedding), single thread, CPU. Memory is the size of each index structure.

{quality_table}

### Where each system wins (per-query nDCG@10, clear = gap of at least {MARGIN})

{cat_table}

**BM25 wins, largest gaps:**
{examples("BM25 wins")}

**Dense wins, largest gaps:**
{examples("Dense wins")}

**Hybrid wins, largest gaps:**
{examples("Hybrid wins")}

**Hybrid loses to the best single retriever:**
{examples("Hybrid loses to the best single retriever")}

### Query characteristics behind those wins

{pattern_table}

### Where reranking hurts (top-20 pool, first {len(deltas)} queries)

Mean nDCG@10 of hybrid on this sample: {r['rerank_hybrid_ndcg']:.3f}; after reranking, queries improved: {helped}, hurt: {hurt}, unchanged: {len(deltas) - helped - hurt}.
Mean Δ nDCG@10: {np.mean(deltas):+.3f}. Reranking added {np.mean(r['rerank_ms']):.0f} ms per query on average.

Worst cases:
{rerank_lines}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-dataset benchmark and failure analysis")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--rerank-queries", type=int, default=30)
    parser.add_argument("--latency-queries", type=int, default=100)
    parser.add_argument("--out", default="results/multi_dataset_benchmark.md")
    args = parser.parse_args()

    setup_logging("INFO")
    results = [
        run_dataset(name, Path(args.root) / name, args.rerank_queries, args.latency_queries)
        for name in args.datasets
    ]

    summary = table(
        ["System"] + [f"{r['name']} nDCG@10" for r in results],
        [[s] + [f"{r['quality'][s]['nDCG@10']:.3f}" for r in results] for s in results[0]["quality"]],
    )
    report = (
        "# Multi-dataset benchmark and failure analysis\n\n"
        "Same code, same settings (500-char chunks, MiniLM embeddings, HNSW M=16/efConstruction=200/efSearch=100, "
        "RRF k=60) on every dataset. Quality is document-level at the top 10.\n\n"
        f"## Summary\n\n{summary}\n\n" + "\n".join(dataset_section(r) for r in results)
    )
    print("\n" + report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
