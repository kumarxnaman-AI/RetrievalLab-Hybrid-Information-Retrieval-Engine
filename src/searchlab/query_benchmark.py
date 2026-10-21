"""Phase 6 experiment: does LLM query transformation improve retrieval?

Compares original / rewrite / HyDE / expansion across BM25, exact dense search and RRF hybrid on
a sample of test queries, and saves per-query results for the Phase 10 harm analysis.
"""

import argparse
import json
import logging
import random
from pathlib import Path

from searchlab.config import Config
from searchlab.evaluation.dataset import load_beir_dataset
from searchlab.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from searchlab.hybrid_benchmark import table
from searchlab.llm import LLMClient
from searchlab.logging_setup import setup_logging
from searchlab.pipeline import build_systems
from searchlab.query_transform import STRATEGY_NAMES, transform
from searchlab.report import save_and_print
from searchlab.retrieval.aggregate import aggregate_to_documents
from searchlab.retrieval.fusion import fuse_results

logger = logging.getLogger(__name__)

TOP = 10
CANDIDATES = 100
OVERFETCH = 3
SYSTEMS = ("BM25", "Dense", "Hybrid")
EPSILON = 0.01  # nDCG changes smaller than this count as "neutral"


def sample_queries(queries: dict[str, str], n: int | None, seed: int = 0) -> dict[str, str]:
    if not n or n >= len(queries):
        return dict(queries)
    chosen = random.Random(seed).sample(sorted(queries), n)
    return {qid: queries[qid] for qid in chosen}


def retrieve(systems, texts: dict[str, tuple[str, str]]) -> dict[str, dict[str, list[str]]]:
    """For {qid: (bm25_text, dense_text)} return {system: {qid: ranked doc ids}}."""
    qids = list(texts)
    dense_vecs = systems.dense.encode([texts[q][1] for q in qids])
    n_chunks = CANDIDATES * OVERFETCH
    runs = {name: {} for name in SYSTEMS}
    for qid, dvec in zip(qids, dense_vecs):
        bm25_docs = aggregate_to_documents(systems.bm25.search(texts[qid][0], n_chunks), CANDIDATES)
        dense_docs = aggregate_to_documents(systems.dense.search_vector(dvec, n_chunks), CANDIDATES)
        hybrid = fuse_results(bm25_docs, dense_docs, method="rrf", rrf_k=60)
        runs["BM25"][qid] = [r.doc_id for r in bm25_docs[:TOP]]
        runs["Dense"][qid] = [r.doc_id for r in dense_docs[:TOP]]
        runs["Hybrid"][qid] = [r.doc_id for r in hybrid[:TOP]]
    return runs


def per_query_metrics(run, qrels) -> dict[str, dict[str, float]]:
    return {
        qid: {
            "ndcg": ndcg_at_k(run[qid], rel, TOP),
            "recall": recall_at_k(run[qid], rel, TOP),
            "mrr": reciprocal_rank(run[qid], rel, TOP),
        }
        for qid, rel in qrels.items()
    }


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query transformation experiment")
    parser.add_argument("--dataset", default="datasets/scifact")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--out", default="results/query_transform_benchmark.md")
    parser.add_argument("--details", default="results/query_transform_details.json")
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="only fetch and cache the LLM transformations (light on CPU), then exit",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    llm = LLMClient.from_env()
    docs, queries, qrels = load_beir_dataset(Path(args.dataset), "test")
    queries = sample_queries(queries, args.max_queries)
    qrels = {q: qrels[q] for q in queries}

    if args.llm_only:
        for strategy in STRATEGY_NAMES:
            for i, text in enumerate(queries.values(), start=1):
                transform(llm, text, strategy)
                if i % 20 == 0:
                    logger.info("%s: %d / %d queries done", strategy, i, len(queries))
        logger.info("LLM calls made: %d (cache hits: %d)", llm.calls_made, llm.cache_hits)
        return

    systems = build_systems(docs, Config())

    results = {}  # strategy -> system -> qid -> metrics
    rewritten = {}  # strategy -> qid -> texts used
    for strategy in STRATEGY_NAMES:
        logger.info("strategy %s: transforming %d queries", strategy, len(queries))
        transformed = {qid: transform(llm, text, strategy) for qid, text in queries.items()}
        rewritten[strategy] = {
            qid: {"bm25_text": t.bm25_text, "dense_text": t.dense_text} for qid, t in transformed.items()
        }
        runs = retrieve(systems, {qid: (t.bm25_text, t.dense_text) for qid, t in transformed.items()})
        results[strategy] = {name: per_query_metrics(run, qrels) for name, run in runs.items()}
    logger.info("LLM calls made: %d (cache hits: %d)", llm.calls_made, llm.cache_hits)

    rows = []
    for system in SYSTEMS:
        base = results["original"][system]
        for strategy in STRATEGY_NAMES:
            current = results[strategy][system]
            d_ndcg = {q: current[q]["ndcg"] - base[q]["ndcg"] for q in qrels}
            helped = sum(d > EPSILON for d in d_ndcg.values())
            harmed = sum(d < -EPSILON for d in d_ndcg.values())
            rows.append(
                [
                    system,
                    strategy,
                    f"{mean(current[q]['recall'] for q in qrels):.3f}",
                    f"{mean(current[q]['mrr'] for q in qrels):.3f}",
                    f"{mean(current[q]['ndcg'] for q in qrels):.3f}",
                    "-" if strategy == "original" else f"{mean(d_ndcg.values()):+.3f}",
                    "-" if strategy == "original" else f"{helped} / {len(qrels) - helped - harmed} / {harmed}",
                ]
            )
    main_table = table(
        ["System", "Query", "Recall@10", "MRR@10", "nDCG@10", "Δ nDCG@10", "helped / neutral / harmed"], rows
    )

    report = f"""# Query transformation experiment

Dataset: {args.dataset}, a random sample of {len(queries)} test queries (seed 0). LLM: `{llm.model}`.
Retrieval: 500-char chunks, BM25 + exact dense (MiniLM) + RRF hybrid (k=60). Quality at the document level.
"helped / neutral / harmed" counts queries whose nDCG@10 changed by more than {EPSILON} versus the original query.

- **original**: the query unchanged.
- **rewrite**: the LLM rewrites the query for both retrievers.
- **hyde**: the dense retriever embeds an LLM-written hypothetical passage; BM25 keeps the original query
  (so the BM25 rows for `hyde` equal `original` by construction).
- **expansion**: the query plus 8 LLM-suggested related terms, for both retrievers.

{main_table}

With {len(queries)} queries, differences of about a point are within noise; look at the helped/harmed counts as well as the mean.
"""
    save_and_print(report, args.out)
    Path(args.details).write_text(
        json.dumps(
            {
                "model": llm.model,
                "queries": queries,
                "transformed": rewritten,
                "metrics": results,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
