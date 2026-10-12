import heapq
from collections import defaultdict

from searchlab.index.inverted_index import InvertedIndex
from searchlab.models import SearchResult
from searchlab.tokenizer import tokenize


class BM25Retriever:
    """BM25 scoring over an InvertedIndex.

    score(q, d) = sum over query terms t of
        idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.index = index
        self.k1 = k1
        self.b = b

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        index = self.index
        avg_len = index.avg_chunk_length or 1.0
        scores: dict[str, float] = defaultdict(float)

        # Repeated query terms count once; only chunks sharing a term are scored.
        for term in set(tokenize(query)):
            postings = index.postings.get(term)
            if not postings:
                continue
            idf = index.idf(term)
            for chunk_id, tf in postings.items():
                length_norm = 1 - self.b + self.b * index.chunk_lengths[chunk_id] / avg_len
                scores[chunk_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)

        top = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [
            SearchResult(chunk_id=cid, doc_id=index.chunk_to_doc[cid], score=score)
            for cid, score in top
        ]
