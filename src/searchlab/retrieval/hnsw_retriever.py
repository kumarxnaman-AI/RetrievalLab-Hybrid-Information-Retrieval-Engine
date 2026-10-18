import json
import logging
import time
from pathlib import Path

import numpy as np

from searchlab.index.hnsw import HNSWIndex
from searchlab.models import SearchResult
from searchlab.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


class HNSWRetriever:
    """Approximate semantic search: the HNSW graph over a DenseRetriever's embeddings."""

    def __init__(
        self,
        dense: DenseRetriever,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        self.dense = dense
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self.index: HNSWIndex | None = None
        self._chunk_to_doc = dict(zip(dense.chunk_ids, dense.doc_ids))
        self.build_seconds = 0.0

    def build(self, cache_path: Path | None = None) -> float:
        """Build the graph (or load a matching cached one). Returns build seconds."""
        meta_path = cache_path.with_suffix(".json") if cache_path else None
        if cache_path is not None and cache_path.exists():
            index = HNSWIndex.load(cache_path)
            if [n.document_id for n in index.nodes] == self.dense.chunk_ids:
                self.index = index
                self.index.ef_search = self.ef_search
                if meta_path and meta_path.exists():
                    self.build_seconds = json.loads(meta_path.read_text())["build_seconds"]
                logger.info("loaded cached HNSW index (%d nodes)", len(index))
                return self.build_seconds

        vectors = self.dense.vectors
        start = time.perf_counter()
        index = HNSWIndex(
            dim=vectors.shape[1],
            max_elements=len(vectors),
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            seed=self.seed,
        )
        index.build(vectors, self.dense.chunk_ids)
        self.build_seconds = time.perf_counter() - start
        self.index = index
        logger.info("built HNSW index in %.1fs: %s", self.build_seconds, index.stats())

        if cache_path is not None:
            index.save(cache_path)
            meta_path.write_text(json.dumps({"build_seconds": self.build_seconds}))
        return self.build_seconds

    def search_vector(
        self, query_vec: np.ndarray, top_k: int = 10, ef_search: int | None = None
    ) -> list[SearchResult]:
        if self.index is None:
            raise RuntimeError("call build() before search()")
        neighbours = self.index.search(query_vec, k=top_k, ef_search=ef_search)
        return [
            SearchResult(n.document_id, self._chunk_to_doc[n.document_id], 1.0 - n.distance)
            for n in neighbours
        ]

    def search(self, query: str, top_k: int = 10, ef_search: int | None = None) -> list[SearchResult]:
        return self.search_vector(self.dense.encode([query])[0], top_k, ef_search)

    def size_bytes(self) -> int:
        return self.index.memory_bytes() if self.index else 0
