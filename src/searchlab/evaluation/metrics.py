import math

# `ranked` is a list of document IDs, best first.
# `relevance` maps document ID -> relevance grade (>0 means relevant).


def _relevant_set(relevance: dict[str, int]) -> set[str]:
    return {doc_id for doc_id, grade in relevance.items() if grade > 0}


def recall_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    relevant = _relevant_set(relevance)
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    relevant = _relevant_set(relevance)
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / k


def hit_rate_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    relevant = _relevant_set(relevance)
    return 1.0 if any(doc_id in relevant for doc_id in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    relevant = _relevant_set(relevance)
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    def dcg(grades: list[int]) -> float:
        return sum(g / math.log2(position + 2) for position, g in enumerate(grades))

    gains = [max(relevance.get(doc_id, 0), 0) for doc_id in ranked[:k]]
    ideal = sorted((g for g in relevance.values() if g > 0), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_run(
    run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 10
) -> dict[str, float]:
    """Average every metric over all queries in `qrels`."""
    metrics = {
        f"Recall@{k}": recall_at_k,
        f"Precision@{k}": precision_at_k,
        f"HitRate@{k}": hit_rate_at_k,
        f"MRR@{k}": reciprocal_rank,
        f"nDCG@{k}": ndcg_at_k,
    }
    totals = {name: 0.0 for name in metrics}
    for query_id, relevance in qrels.items():
        ranked = run.get(query_id, [])
        for name, fn in metrics.items():
            totals[name] += fn(ranked, relevance, k)
    n = len(qrels)
    return {name: total / n for name, total in totals.items()}
