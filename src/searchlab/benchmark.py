import argparse
import logging
import time
from pathlib import Path

import numpy as np

from searchlab.config import Config
from searchlab.evaluation.dataset import load_beir_dataset
from searchlab.evaluation.metrics import evaluate_run
from searchlab.logging_setup import setup_logging
from searchlab.pipeline import build_systems
from searchlab.retrieval.aggregate import aggregate_to_documents

logger = logging.getLogger(__name__)

# Retrieve extra chunks so that, after collapsing to documents, we still have k docs.
CHUNK_OVERFETCH = 5


def run_queries(search_fn, queries: dict[str, str], k: int):
    """Run every query; return {query_id: ranked doc IDs} and per-query latencies (ms)."""
    run: dict[str, list[str]] = {}
    latencies_ms: list[float] = []
    for query_id, text in queries.items():
        start = time.perf_counter()
        chunk_results = search_fn(text, k * CHUNK_OVERFETCH)
        doc_results = aggregate_to_documents(chunk_results, k)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        run[query_id] = [r.doc_id for r in doc_results]
    return run, latencies_ms


def format_table(rows: list[dict], k: int) -> str:
    header = (
        f"| Retrieval | Recall@{k} | Precision@{k} | HitRate@{k} | MRR@{k} | nDCG@{k} "
        "| p50 latency | p95 latency | Index size | Build time |"
    )
    divider = "|" + "---|" * 10
    lines = [header, divider]
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {m[f'Recall@{k}']:.3f} | {m[f'Precision@{k}']:.3f} "
            f"| {m[f'HitRate@{k}']:.3f} | {m[f'MRR@{k}']:.3f} | {m[f'nDCG@{k}']:.3f} "
            f"| {r['p50']:.2f} ms | {r['p95']:.2f} ms "
            f"| {r['size_bytes'] / 1e6:.2f} MB | {r['build_seconds']:.1f} s |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BM25 vs exact dense search")
    parser.add_argument("--dataset", default="datasets/scifact")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--out", default="results/benchmark.md")
    args = parser.parse_args()

    config = Config(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        bm25_k1=args.k1,
        bm25_b=args.b,
        top_k=args.k,
    )
    setup_logging(config.log_level)

    docs, queries, qrels = load_beir_dataset(Path(args.dataset))
    if args.max_queries:
        queries = dict(list(queries.items())[: args.max_queries])
        qrels = {qid: qrels[qid] for qid in queries}

    systems = build_systems(docs, config)

    rows = []
    for name, retriever, build_seconds in [
        ("BM25", systems.bm25, systems.bm25_build_seconds),
        ("Dense Exact", systems.dense, systems.dense_build_seconds),
    ]:
        # Warm up once so one-time costs (model load, caches) don't skew latency.
        retriever.search(next(iter(queries.values())), args.k)
        run, latencies = run_queries(retriever.search, queries, args.k)
        size = (
            retriever.index.size_bytes() if name == "BM25" else retriever.size_bytes()
        )
        rows.append(
            {
                "name": name,
                "metrics": evaluate_run(run, qrels, args.k),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "size_bytes": size,
                "build_seconds": build_seconds,
            }
        )

    table = format_table(rows, args.k)
    summary = (
        f"Dataset: {args.dataset} ({len(docs)} docs, {len(systems.chunks)} chunks, "
        f"{len(queries)} queries)\n"
        f"Chunking: size={args.chunk_size}, overlap={args.overlap} | "
        f"BM25: k1={args.k1}, b={args.b} | Dense model: {config.embedding_model}\n"
    )
    print("\n" + summary)
    print(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Benchmark\n\n{summary}\n{table}\n", encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
