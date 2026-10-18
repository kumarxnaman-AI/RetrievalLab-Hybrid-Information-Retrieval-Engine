"""Phase 7 experiment: does smarter chunking improve BM25, dense and hybrid retrieval?

Every strategy is evaluated with BM25, exact dense search and RRF-fused hybrid retrieval on
the same queries. Exact (not HNSW) dense search keeps approximation error out of the comparison.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from searchlab.chunkers import STRATEGIES, build_chunks
from searchlab.config import Config
from searchlab.evaluation.dataset import load_beir_dataset
from searchlab.evaluation.metrics import quality_summary
from searchlab.hybrid_benchmark import table
from searchlab.index.inverted_index import InvertedIndex
from searchlab.logging_setup import setup_logging
from searchlab.retrieval.aggregate import aggregate_to_documents
from searchlab.retrieval.bm25 import BM25Retriever
from searchlab.retrieval.dense import DenseRetriever
from searchlab.retrieval.fusion import fuse_results

logger = logging.getLogger(__name__)

TOP = 10
CANDIDATES = 100
OVERFETCH = 3
SYSTEMS = ("BM25", "Dense", "Hybrid")


def evaluate_strategy(strategy, docs, queries, qrels, qvecs, dense_for_encoding, size, cache_dir):
    config = Config()
    logger.info("=== strategy: %s ===", strategy)
    start = time.perf_counter()
    chunks = build_chunks(strategy, docs, size=size, encode=dense_for_encoding.encode)
    chunk_seconds = time.perf_counter() - start

    index = InvertedIndex()
    index.build(chunks)
    bm25 = BM25Retriever(index, config.bm25_k1, config.bm25_b)

    dense = DenseRetriever(config.embedding_model)
    dense.build(chunks, cache_path=cache_dir / f"chunking_{strategy}_{size}_{len(chunks)}.npz")

    n_chunks = CANDIDATES * OVERFETCH
    runs = {name: {} for name in SYSTEMS}
    for (qid, text), qvec in zip(queries.items(), qvecs):
        bm25_docs = aggregate_to_documents(bm25.search(text, n_chunks), CANDIDATES)
        dense_docs = aggregate_to_documents(dense.search_vector(qvec, n_chunks), CANDIDATES)
        hybrid_docs = fuse_results(bm25_docs, dense_docs, method="rrf", rrf_k=60)
        runs["BM25"][qid] = [r.doc_id for r in bm25_docs[:TOP]]
        runs["Dense"][qid] = [r.doc_id for r in dense_docs[:TOP]]
        runs["Hybrid"][qid] = [r.doc_id for r in hybrid_docs[:TOP]]

    lengths = [len(c.text) for c in chunks]
    return {
        "strategy": strategy,
        "chunks": len(chunks),
        "avg_chars": float(np.mean(lengths)),
        "p90_chars": float(np.percentile(lengths, 90)),
        "chunk_seconds": chunk_seconds,
        "quality": {name: quality_summary(run, qrels) for name, run in runs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking strategy experiment")
    parser.add_argument("--dataset", default="datasets/scifact")
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    parser.add_argument("--out", default="results/chunking_benchmark.md")
    args = parser.parse_args()

    setup_logging("INFO")
    docs, queries, qrels = load_beir_dataset(Path(args.dataset), "test")
    cache_dir = Path(".cache")

    encoder = DenseRetriever(Config().embedding_model)
    qvecs = encoder.encode(list(queries.values()))

    results = [
        evaluate_strategy(s, docs, queries, qrels, qvecs, encoder, args.size, cache_dir)
        for s in args.strategies
    ]

    stats_table = table(
        ["Strategy", "Chunks", "Avg chars", "p90 chars"],
        [[r["strategy"], r["chunks"], f"{r['avg_chars']:.0f}", f"{r['p90_chars']:.0f}"] for r in results],
    )
    detail_table = table(
        ["System", "Chunking", "Recall@10", "MRR@10", "nDCG@10"],
        [
            [system, r["strategy"]]
            + [f"{r['quality'][system][m]:.3f}" for m in ("Recall@10", "MRR@10", "nDCG@10")]
            for system in SYSTEMS
            for r in results
        ],
    )
    matrix = table(
        ["nDCG@10"] + [r["strategy"] for r in results],
        [
            [system] + [f"{r['quality'][system]['nDCG@10']:.3f}" for r in results]
            for system in SYSTEMS
        ],
    )

    lines = []
    for system in SYSTEMS:
        best = max(results, key=lambda r: r["quality"][system]["nDCG@10"])
        worst = min(results, key=lambda r: r["quality"][system]["nDCG@10"])
        spread = best["quality"][system]["nDCG@10"] - worst["quality"][system]["nDCG@10"]
        lines.append(
            f"- **{system}**: best `{best['strategy']}` ({best['quality'][system]['nDCG@10']:.3f}), "
            f"worst `{worst['strategy']}` ({worst['quality'][system]['nDCG@10']:.3f}), spread {spread:.3f} nDCG."
        )

    report = f"""# Chunking experiment

Dataset: {args.dataset} - {len(docs)} documents, {len(queries)} test queries. Target chunk size {args.size} characters.
Quality is document-level (chunk results collapse to the best chunk per document). Hybrid = BM25 + exact dense,
RRF k=60, {CANDIDATES} candidates each. Dense model: {Config().embedding_model}.

Strategies: `fixed` (no overlap), `sliding` (stride = half the window), `recursive` (sentence-packing that respects
paragraph breaks), `semantic` (cuts where adjacent-sentence embedding similarity drops, max {int(args.size * 1.6)} chars).
SciFact abstracts are single paragraphs, so `recursive` reduces to sentence-boundary packing here.

## nDCG@10 by system and chunking

{matrix}

## Best and worst per system

{chr(10).join(lines)}

## Chunk statistics

{stats_table}

## Full metrics

{detail_table}
"""
    print("\n" + report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
