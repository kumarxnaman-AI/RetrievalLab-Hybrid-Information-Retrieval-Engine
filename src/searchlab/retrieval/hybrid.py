from dataclasses import dataclass
from typing import Callable

import numpy as np

from searchlab.models import SearchResult
from searchlab.retrieval.aggregate import aggregate_to_documents
from searchlab.retrieval.bm25 import BM25Retriever
from searchlab.retrieval.fusion import fuse_results


@dataclass
class HybridParams:
    method: str = "rrf"  # rrf | weighted_rrf | score_fusion
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    alpha: float = 0.5  # score fusion: weight of the dense score
    candidates: int = 100  # documents kept from each retriever before fusion
    ef_search: int | None = None  # forwarded to the dense search if it supports it


class HybridRetriever:
    """BM25 + dense search, merged with a fusion method.

    `dense_search(query_vector, n_chunks, ef_search)` returns chunk results, nearest first.
    `encode(texts)` embeds queries. Both are injected, so the dense side can be exact or HNSW.
    """

    # Dense/BM25 searches fetch extra chunks, since several chunks can come from one document.
    CHUNK_OVERFETCH = 3

    def __init__(
        self,
        bm25: BM25Retriever,
        dense_search: Callable[[np.ndarray, int, int | None], list[SearchResult]],
        encode: Callable[[list[str]], np.ndarray],
    ) -> None:
        self.bm25 = bm25
        self.dense_search = dense_search
        self.encode = encode

    def candidates(
        self, query: str, params: HybridParams, query_vec: np.ndarray | None = None
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """First-stage document candidates from each retriever: (bm25_docs, dense_docs)."""
        n_chunks = params.candidates * self.CHUNK_OVERFETCH
        if query_vec is None:
            query_vec = self.encode([query])[0]
        bm25_docs = aggregate_to_documents(self.bm25.search(query, n_chunks), params.candidates)
        ef = max(params.ef_search or 0, n_chunks)
        dense_docs = aggregate_to_documents(
            self.dense_search(query_vec, n_chunks, ef), params.candidates
        )
        return bm25_docs, dense_docs

    def search(
        self,
        query: str,
        top_k: int = 10,
        params: HybridParams | None = None,
        query_vec: np.ndarray | None = None,
    ) -> list[SearchResult]:
        params = params or HybridParams()
        bm25_docs, dense_docs = self.candidates(query, params, query_vec)
        fused = fuse_results(
            bm25_docs,
            dense_docs,
            method=params.method,
            rrf_k=params.rrf_k,
            bm25_weight=params.bm25_weight,
            dense_weight=params.dense_weight,
            alpha=params.alpha,
        )
        return fused[:top_k]
