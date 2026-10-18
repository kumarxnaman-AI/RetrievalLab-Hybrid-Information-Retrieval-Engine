"""Phase 2 experiment: exact cosine KNN vs our HNSW at several efSearch values."""

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np

from searchlab.config import Config
from searchlab.evaluation.dataset import load_beir_dataset
from searchlab.evaluation.metrics import evaluate_run
from searchlab.logging_setup import setup_logging
from searchlab.pipeline import build_systems
from searchlab.retrieval.aggregate import aggregate_to_documents
from searchlab.retrieval.hnsw_retriever import HNSWRetriever

logger = logging.getLogger(__name__)


def timed_search(search_fn, query_vectors: np.ndarray):
    """Run every query; return per-query results and latencies in ms."""
    results, latencies_ms = [], []
    for vec in query_vectors:
        start = time.perf_counter()
        results.append(search_fn(vec))
        latencies_ms.append((time.perf_counter() - start) * 1000)
    return results, np.array(latencies_ms)


def summarize(name, results, latencies_ms, truth, query_ids, qrels, k, build_s, mem_bytes):
    recall = float(
        np.mean(
            [
                len({r.chunk_id for r in res} & truth[i]) / len(truth[i])
                for i, res in enumerate(results)
            ]
        )
    )
    run = {
        qid: [r.doc_id for r in aggregate_to_documents(res, k)]
        for qid, res in zip(query_ids, results)
    }
    quality = evaluate_run(run, qrels, k)
    return {
        "name": name,
        "recall": recall,
        "ndcg": quality[f"nDCG@{k}"],
        "p50": float(np.percentile(latencies_ms, 50)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "qps": len(latencies_ms) / (latencies_ms.sum() / 1000),
        "build_s": build_s,
        "mem_mb": mem_bytes / 1e6,
    }


def format_table(rows, k) -> str:
    lines = [
        f"| Search | ANN Recall@{k} | nDCG@{k} | p50 | p95 | Queries/sec | Build time | Memory |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['recall']:.3f} | {r['ndcg']:.3f} | {r['p50']:.2f} ms "
            f"| {r['p95']:.2f} ms | {r['qps']:.0f} | {r['build_s']:.1f} s | {r['mem_mb']:.1f} MB |"
        )
    return "\n".join(lines)


def plot_recall_vs_latency(rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hnsw = [r for r in rows if r["name"].startswith("HNSW")]
    exact = next(r for r in rows if r["name"].startswith("Exact"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    recalls = [r["recall"] for r in hnsw]
    ax.plot([r["p50"] for r in hnsw], recalls, "o-", color="tab:blue", label="HNSW p50 latency")
    ax.plot([r["p95"] for r in hnsw], recalls, "^--", color="tab:blue", alpha=0.45, label="HNSW p95 latency")
    for r in hnsw:
        ax.annotate(
            "ef=" + r["name"].split("=")[1], (r["p50"], r["recall"]),
            textcoords="offset points", xytext=(6, -12), fontsize=8,
        )
    ax.scatter([exact["p50"]], [exact["recall"]], marker="s", color="tab:red", label="Exact KNN p50", zorder=3)
    ax.scatter([exact["p95"]], [exact["recall"]], marker="s", color="tab:red", alpha=0.4, label="Exact KNN p95", zorder=3)
    ax.set_xlabel("latency per query (ms)")
    ax.set_ylabel("ANN Recall@10 (vs exact)")
    ax.set_title("HNSW recall vs latency (labeled by efSearch)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact KNN vs HNSW benchmark")
    parser.add_argument("--dataset", default="datasets/scifact")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--efs", type=int, nargs="+", default=[10, 20, 50, 100, 200])
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    config = Config()
    setup_logging(config.log_level)

    docs, queries, qrels = load_beir_dataset(Path(args.dataset))
    systems = build_systems(docs, config)
    dense = systems.dense
    query_ids = list(queries)
    query_vectors = dense.encode([queries[q] for q in query_ids])

    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", config.embedding_model)
    cache = Path(".cache") / (
        f"hnsw_{model_slug}_M{args.M}_efc{args.ef_construction}_{len(systems.chunks)}.pkl"
    )
    hnsw = HNSWRetriever(dense, M=args.M, ef_construction=args.ef_construction)
    hnsw.build(cache_path=cache)

    k = args.k
    dense.search_vector(query_vectors[0], k)  # warm-up
    exact_results, exact_lat = timed_search(lambda v: dense.search_vector(v, k), query_vectors)
    truth = [{r.chunk_id for r in res} for res in exact_results]

    rows = [
        summarize("Exact KNN", exact_results, exact_lat, truth, query_ids, qrels, k, 0.0, dense.size_bytes())
    ]
    for ef in args.efs:
        hnsw.search_vector(query_vectors[0], k, ef)  # warm-up
        res, lat = timed_search(lambda v, ef=ef: hnsw.search_vector(v, k, ef), query_vectors)
        rows.append(
            summarize(f"HNSW ef={ef}", res, lat, truth, query_ids, qrels, k, hnsw.build_seconds, hnsw.size_bytes())
        )

    table = format_table(rows, k)
    stats = hnsw.index.stats()
    summary = (
        f"Dataset: {args.dataset} ({len(systems.chunks)} chunk vectors, dim={dense.vectors.shape[1]}, "
        f"{len(queries)} queries)\n"
        f"HNSW: M={args.M}, efConstruction={args.ef_construction}, metric=cosine, "
        f"layers={stats['layers']}, avg layer-0 degree={stats['avg_layer0_degree']:.1f}\n"
        f"ANN recall is the overlap of HNSW's top-{k} chunks with the exact top-{k} chunks. "
        f"nDCG@{k} is document-level, computed from those top-{k} chunks.\n"
        "Latency is vector search only (queries are embedded beforehand), single thread."
    )
    print("\n" + summary + "\n\n" + table)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hnsw_benchmark.md").write_text(f"# HNSW vs exact search\n\n{summary}\n\n{table}\n", encoding="utf-8")
    plot_recall_vs_latency(rows, out_dir / "hnsw_recall_vs_latency.png")
    logger.info("wrote results to %s", out_dir)


if __name__ == "__main__":
    main()
