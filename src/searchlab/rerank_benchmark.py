"""Phase 4 experiment: hybrid retrieval vs hybrid + cross-encoder reranking.

Quality: every test query's top-100 hybrid candidates are scored once by the cross-encoder;
each candidate-pool size N then reranks only the top N of those (so quality for N=10/20/50/100
comes from the same scores). Latency: measured separately, for real, on a query sample.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from searchlab.config import Config
from searchlab.evaluation.dataset import load_beir_dataset
from searchlab.evaluation.metrics import ndcg_at_k, quality_summary
from searchlab.hybrid_benchmark import collect_candidates, table
from searchlab.logging_setup import setup_logging
from searchlab.pipeline import build_hnsw, build_systems
from searchlab.retrieval.fusion import fuse_results
from searchlab.retrieval.hybrid import HybridParams, HybridRetriever
from searchlab.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

TOP = 10
POOL_SIZES = (10, 20, 50, 100)


def pct(a, b):
    return f"{(a - b) / b * 100:+.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reranking experiment")
    parser.add_argument("--dataset", default="datasets/scifact")
    parser.add_argument("--latency-queries", type=int, default=30)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--out", default="results/rerank_benchmark.md")
    args = parser.parse_args()

    config = Config()
    setup_logging(config.log_level)
    docs, queries, qrels = load_beir_dataset(Path(args.dataset), "test")
    if args.max_queries:
        queries = dict(list(queries.items())[: args.max_queries])
        qrels = {q: qrels[q] for q in queries}
    doc_texts = {d.doc_id: d.text for d in docs}

    systems = build_systems(docs, config)
    hnsw = build_hnsw(systems.dense, config)
    hybrid = HybridRetriever(systems.bm25, lambda v, n, ef: hnsw.search_vector(v, n, ef), systems.dense.encode)
    params = HybridParams(method="rrf", rrf_k=60, candidates=100)
    reranker = CrossEncoderReranker()

    # --- quality: hybrid top-100 for every query, scored once by the cross-encoder ---------
    candidates = collect_candidates(hybrid, systems, queries, params)
    fused = {
        qid: fuse_results(bm25, hnsw_docs, method="rrf", rrf_k=60)[:100]
        for qid, (bm25, hnsw_docs, _) in candidates.items()
    }
    logger.info("scoring %d query-document pairs with the cross-encoder", sum(len(v) for v in fused.values()))
    scored = {}  # qid -> [(doc_id, rerank_score)] in hybrid order
    for i, (qid, results) in enumerate(fused.items(), start=1):
        scores = reranker.score(queries[qid], [doc_texts[r.doc_id] for r in results])
        scored[qid] = [(r.doc_id, s) for r, s in zip(results, scores)]
        if i % 50 == 0:
            logger.info("reranked %d / %d queries", i, len(fused))

    def run_for(n):
        return {
            qid: [d for d, _ in sorted(pairs[:n], key=lambda p: -p[1])][:TOP]
            for qid, pairs in scored.items()
        }

    hybrid_run = {qid: [r.doc_id for r in results[:TOP]] for qid, results in fused.items()}
    quality = {"Hybrid (RRF)": quality_summary(hybrid_run, qrels)}
    runs = {}
    for n in POOL_SIZES:
        runs[n] = run_for(n)
        quality[f"Hybrid + reranker (top {n})"] = quality_summary(runs[n], qrels)

    # --- latency: real end-to-end timing on a sample of queries -----------------------------
    sample = list(queries.values())[: args.latency_queries]
    hybrid.search(sample[0], 100, params)
    reranker.rerank(sample[0], hybrid.search(sample[0], 10, params), doc_texts, TOP)  # warm-up
    latency = {}
    base = []
    for text in sample:
        t = time.perf_counter()
        hybrid.search(text, TOP, params)
        base.append((time.perf_counter() - t) * 1000)
    latency["Hybrid (RRF)"] = base
    for n in POOL_SIZES:
        lat = []
        for text in sample:
            t = time.perf_counter()
            pool = hybrid.search(text, n, params)
            reranker.rerank(text, pool, doc_texts, TOP)
            lat.append((time.perf_counter() - t) * 1000)
        latency[f"Hybrid + reranker (top {n})"] = lat

    base_q, base_p95 = quality["Hybrid (RRF)"], float(np.percentile(latency["Hybrid (RRF)"], 95))
    rows = []
    for name, q in quality.items():
        lat = latency[name]
        p95 = float(np.percentile(lat, 95))
        rows.append(
            [
                name,
                f"{q['nDCG@10']:.3f}",
                f"{q['MRR@10']:.3f}",
                f"{q['Recall@5']:.3f}",
                f"{q['Recall@10']:.3f}",
                f"{np.percentile(lat, 50):.0f} ms",
                f"{p95:.0f} ms",
                "-" if name == "Hybrid (RRF)" else f"{pct(q['nDCG@10'], base_q['nDCG@10'])} nDCG, +{p95 - base_p95:.0f} ms",
            ]
        )
    main_table = table(
        ["System", "nDCG@10", "MRR@10", "Recall@5", "Recall@10", "p50 latency", "p95 latency", "vs. hybrid"],
        rows,
    )

    # --- where does reranking help or hurt? (per-query nDCG@10, best pool size) --------------
    best_n = max(POOL_SIZES, key=lambda n: quality[f"Hybrid + reranker (top {n})"]["nDCG@10"])
    better = worse = same = 0
    for qid, rel in qrels.items():
        before = ndcg_at_k(hybrid_run[qid], rel, TOP)
        after = ndcg_at_k(runs[best_n][qid], rel, TOP)
        better += after > before + 1e-9
        worse += after < before - 1e-9
        same += abs(after - before) <= 1e-9
    best_q = quality[f"Hybrid + reranker (top {best_n})"]
    best_p95 = float(np.percentile(latency[f"Hybrid + reranker (top {best_n})"], 95))

    verb = "improved" if best_q["nDCG@10"] >= base_q["nDCG@10"] else "reduced"
    report = f"""# Reranking experiment

Dataset: {args.dataset} - {len(docs)} documents, {len(queries)} test queries.
First stage: BM25 + HNSW fused with RRF (k=60), 100 candidates per retriever.
Reranker: `{reranker.model_name}`. Quality uses all {len(queries)} test queries; latency uses the first
{len(sample)} (end-to-end: query text in, top-{TOP} documents out, single thread, CPU).

{main_table}

## Headline

Reranking the top {best_n} hybrid candidates (the best pool size) {verb} nDCG@10 from {base_q['nDCG@10']:.3f} to
{best_q['nDCG@10']:.3f} ({pct(best_q['nDCG@10'], base_q['nDCG@10'])}), while adding about
{best_p95 - base_p95:.0f} ms at p95 ({base_p95:.0f} ms -> {best_p95:.0f} ms).

Per-query effect at top {best_n} (nDCG@10): improved on **{better}** queries, hurt on **{worse}**,
unchanged on {same}.
"""
    print("\n" + report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
