import logging
import time
from pathlib import Path

import numpy as np

from searchlab.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


def cosine_similarity(
    query: np.ndarray, matrix: np.ndarray, matrix_norms: np.ndarray | None = None
) -> np.ndarray:
    """Cosine similarity between one query vector and every row of a matrix.

    cos(q, v) = (q . v) / (|q| * |v|)
    """
    if matrix_norms is None:
        matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = matrix_norms * np.linalg.norm(query)
    return (matrix @ query) / np.maximum(denom, 1e-12)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first (exact, no approximation)."""
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    partial = np.argpartition(-scores, k - 1)[:k]
    return partial[np.argsort(-scores[partial])]


class DenseRetriever:
    """Exact (brute-force) semantic search over chunk embeddings."""

    def __init__(self, model_name: str, encoder=None) -> None:
        # `encoder` (list[str] -> ndarray) can replace the real model, e.g. in tests.
        self.model_name = model_name
        self._encoder = encoder
        self._model = None
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.norms = np.zeros(0, dtype=np.float32)
        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if self._encoder is not None:
            return np.asarray(self._encoder(texts), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)

    def build(self, chunks: list[Chunk], cache_path: Path | None = None) -> float:
        """Embed all chunks (or load a matching cache). Returns seconds spent."""
        start = time.perf_counter()
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.doc_ids = [c.doc_id for c in chunks]

        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            if list(cached["chunk_ids"]) == self.chunk_ids:
                self.vectors = cached["vectors"]
                logger.info("loaded %d cached embeddings", len(self.chunk_ids))
                self.norms = np.linalg.norm(self.vectors, axis=1)
                return time.perf_counter() - start

        logger.info("embedding %d chunks", len(chunks))
        self.vectors = self.encode([c.text for c in chunks], show_progress=True)
        self.norms = np.linalg.norm(self.vectors, axis=1)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                vectors=self.vectors,
                chunk_ids=np.array(self.chunk_ids),
            )
        return time.perf_counter() - start

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        return self.search_vector(self.encode([query])[0], top_k)

    def search_vector(self, query_vec: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Exact top-K for an already-embedded query (no model call)."""
        sims = cosine_similarity(query_vec, self.vectors, self.norms)
        return [
            SearchResult(self.chunk_ids[i], self.doc_ids[i], float(sims[i]))
            for i in top_k_indices(sims, top_k)
        ]

    def size_bytes(self) -> int:
        return int(self.vectors.nbytes)
