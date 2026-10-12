from searchlab.models import SearchResult


def aggregate_to_documents(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Collapse ranked chunk results into ranked documents.

    Each document keeps its best-scoring chunk. Input must be sorted best-first.
    """
    seen: set[str] = set()
    docs: list[SearchResult] = []
    for result in results:
        if result.doc_id in seen:
            continue
        seen.add(result.doc_id)
        docs.append(result)
        if len(docs) == top_k:
            break
    return docs
