"""Phase 3 experiment: BM25 vs dense vs RRF vs weighted RRF vs score fusion.

Fusion parameters are tuned on the dataset's *train* queries and reported on the *test*
queries, so the comparison is not inflated by tuning on the data we score.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from searchlab.config import Config
from searchlab.evaluation.dataset import load_beir_dataset
from searchlab.evaluation.metrics import quality_summary
from searchlab.logging_setup import setup_logging
from searchlab.pipeline import build_hnsw, build_systems
from searchlab.retrieval.aggregate import aggregate_to_documents
from searchlab.retrieval.fusion import fuse_results
from searchlab.retrieval.hybrid import HybridParams, HybridRetriever

logger = logging.getLogger(__name__)

TOP = 10


def collect_candidates(hybrid, systems, queries, params):
    """First-stage candidates for every query: {qid: (bm25_docs, hnsw_docs, exact_docs)}."""
    qids = list(queries)
    qvecs = systems.dense.encode([queries[q] for q in qids])
    n_chunks = params.candidates * hybrid.CHUNK_OVERFETCH
    out = {}
    for qid, qvec in zip(qids, qvecs):
        bm25_docs, hnsw_docs = hybrid.candidates(queries[qid], params, qvec)
        exact_docs = aggregate_to_documents(systems.dense.search_vector(qvec, n_chunks), params.candidates)
        out[qid] = (bm25_docs, hnsw_docs, exact_docs)
    return out


def run_of(candidates, fn):
    return {qid: [r.doc_id for r in fn(*docs)[:TOP]] for qid, docs in candidates.items()}


def evaluate_fusion(candidates, qrels, **fusion_args):
    run = run_of(candidates, lambda bm25, hnsw, exact: fuse_results(bm25, hnsw, **fusion_args))
    return quality_summary(run, qrels)


def tune(candidates, qrels, grid):
    """Pick the fusion settings (from `grid`) with the best train nDCG@10."""
    best, best_ndcg = None, -1.0
    for args in grid:
        ndcg = evaluate_fusion(candidates, qrels, **args)["nDCG@10"]
        if ndcg > best_ndcg:
            best, best_ndcg = args, ndcg
    return best, best_ndcg


def latency_ms(fn, queries):
    lat = []
    for text in queries:
        start = time.perf_counter()
        fn(text)
        lat.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 95))


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def metric_cells(m):
    return [f"{m[k]:.3f}" for k in ("Recall@5", "Recall@10", "MRR@10", "nDCG@10")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval experiment")
    parser.add_argument("--dataset", default="datasets/scifact")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--out", default="results/hybrid_benchmark.md")
    args = parser.parse_args()

    config = Config()
    setup_logging(config.log_level)
    path = Path(args.dataset)
    docs, test_q, test_qrels = load_beir_dataset(path, "test")
    _, train_q, train_qrels = load_beir_dataset(path, "train")

    systems = build_systems(docs, config)
    hnsw = build_hnsw(systems.dense, config)
    hybrid = HybridRetriever(
        systems.bm25,
        lambda vec, n, ef: hnsw.search_vector(vec, n, ef),
        systems.dense.encode,
    )
    params = HybridParams(candidates=args.candidates)

    logger.info("collecting candidates for %d train and %d test queries", len(train_q), len(test_q))
    train_c = collect_candidates(hybrid, systems, train_q, params)
    test_c = collect_candidates(hybrid, systems, test_q, params)

    # --- tune on train ---------------------------------------------------------------
    weighted_grid = [
        dict(method="weighted_rrf", rrf_k=k, bm25_weight=w, dense_weight=round(1 - w, 2))
        for k in (10, 60)
        for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ]
    alpha_grid = [dict(method="score_fusion", alpha=round(a / 10, 1)) for a in range(11)]
    best_w, best_w_train = tune(train_c, train_qrels, weighted_grid)
    best_a, best_a_train = tune(train_c, train_qrels, alpha_grid)
    logger.info("weighted RRF best on train: %s (nDCG %.3f)", best_w, best_w_train)
    logger.info("score fusion best on train: %s (nDCG %.3f)", best_a, best_a_train)

    # --- evaluate on test --------------------------------------------------------------
    systems_eval = {
        "BM25": quality_summary(run_of(test_c, lambda b, h, e: b), test_qrels),
        "Dense (exact)": quality_summary(run_of(test_c, lambda b, h, e: e), test_qrels),
        "Dense (HNSW)": quality_summary(run_of(test_c, lambda b, h, e: h), test_qrels),
        "RRF (k=60)": evaluate_fusion(test_c, test_qrels, method="rrf", rrf_k=60),
        "Weighted RRF (tuned)": evaluate_fusion(test_c, test_qrels, **best_w),
        "Score fusion (tuned)": evaluate_fusion(test_c, test_qrels, **best_a),
    }
    main_table = table(
        ["Method", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"],
        [[name] + metric_cells(m) for name, m in systems_eval.items()],
    )

    rrf_sweep = table(
        ["RRF k", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"],
        [
            [k] + metric_cells(evaluate_fusion(test_c, test_qrels, method="rrf", rrf_k=k))
            for k in (1, 10, 30, 60, 100, 200)
        ],
    )
    alpha_sweep = table(
        ["alpha (dense weight)", "train nDCG@10", "test nDCG@10"],
        [
            [
                f"{g['alpha']:.1f}",
                f"{evaluate_fusion(train_c, train_qrels, **g)['nDCG@10']:.3f}",
                f"{evaluate_fusion(test_c, test_qrels, **g)['nDCG@10']:.3f}",
            ]
            for g in alpha_grid
        ],
    )

    # --- end-to-end latency (query text in, top-10 documents out, includes embedding) ----
    sample = list(test_q.values())[:100]
    hybrid.search(sample[0], TOP, params)  # warm-up
    lat_rows = []
    for name, fn in [
        ("BM25", lambda t: aggregate_to_documents(systems.bm25.search(t, 50), TOP)),
        (
            "Dense (HNSW, ef=100)",
            lambda t: aggregate_to_documents(hnsw.search(t, 50, 100), TOP),
        ),
        ("Hybrid RRF (BM25 + HNSW)", lambda t: hybrid.search(t, TOP, params)),
    ]:
        p50, p95 = latency_ms(fn, sample)
        lat_rows.append([name, f"{p50:.1f} ms", f"{p95:.1f} ms"])
    latency_table = table(["Pipeline", "p50", "p95"], lat_rows)

    report = f"""# Hybrid retrieval experiment

Dataset: {args.dataset} - {len(docs)} documents, {len(test_q)} test queries, {len(train_q)} train queries.
Each retriever contributes its top {args.candidates} documents; the dense side is our HNSW index
(M=16, efConstruction=200) unless stated. Fusion settings were **tuned on the train queries and
reported on the test queries**.

Tuned weighted RRF: `{best_w}` (train nDCG@10 {best_w_train:.3f})
Tuned score fusion: `{best_a}` (train nDCG@10 {best_a_train:.3f})

## Main comparison (test queries)

{main_table}

## RRF k sweep (test queries)

{rrf_sweep}

## Score fusion alpha sweep

{alpha_sweep}

## End-to-end latency (100 test queries, includes query embedding, single thread)

{latency_table}
"""
    print("\n" + report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
