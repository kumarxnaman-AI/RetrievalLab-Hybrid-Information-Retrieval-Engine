"""Ways to merge ranked lists from different retrievers into one ranking."""

from collections import defaultdict

from searchlab.models import SearchResult

METHODS = ("rrf", "weighted_rrf", "score_fusion")


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """RRF(d) = sum over rankers i of  w_i / (k + rank_i(d)),  ranks start at 1.

    Only ranks matter, not scores, so it needs no normalization between BM25 scores
    (unbounded) and cosine similarities (0 to 1). Larger k flattens the advantage of
    top ranks. Documents missing from a ranking contribute nothing for that ranker.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    weights = weights if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("need exactly one weight per ranking")

    scores: dict[str, float] = defaultdict(float)
    for weight, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Rescale scores to [0, 1] so different scoring scales become comparable."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {doc_id: 1.0 for doc_id in scores}
    return {doc_id: (s - low) / (high - low) for doc_id, s in scores.items()}


def score_fusion(
    dense_scores: dict[str, float], bm25_scores: dict[str, float], alpha: float = 0.5
) -> list[tuple[str, float]]:
    """alpha * dense + (1 - alpha) * bm25, after min-max normalizing each list.

    A document missing from one list gets 0 from that list.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    dense_n, bm25_n = min_max_normalize(dense_scores), min_max_normalize(bm25_scores)
    fused = {
        doc_id: alpha * dense_n.get(doc_id, 0.0) + (1 - alpha) * bm25_n.get(doc_id, 0.0)
        for doc_id in dense_n.keys() | bm25_n.keys()
    }
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def fuse_results(
    bm25_docs: list[SearchResult],
    dense_docs: list[SearchResult],
    method: str = "rrf",
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    alpha: float = 0.5,
) -> list[SearchResult]:
    """Fuse two document-level result lists (each sorted best first)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    if method == "rrf":
        ranked = reciprocal_rank_fusion([[r.doc_id for r in bm25_docs], [r.doc_id for r in dense_docs]], rrf_k)
    elif method == "weighted_rrf":
        ranked = reciprocal_rank_fusion(
            [[r.doc_id for r in bm25_docs], [r.doc_id for r in dense_docs]],
            rrf_k,
            weights=[bm25_weight, dense_weight],
        )
    else:
        ranked = score_fusion(
            {r.doc_id: r.score for r in dense_docs},
            {r.doc_id: r.score for r in bm25_docs},
            alpha,
        )

    # Keep a representative chunk per document (prefer the dense system's best chunk).
    chunk_of = {r.doc_id: r.chunk_id for r in bm25_docs}
    chunk_of.update({r.doc_id: r.chunk_id for r in dense_docs})
    return [SearchResult(chunk_of[doc_id], doc_id, score) for doc_id, score in ranked]
