import logging
from typing import Callable

from searchlab.models import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Second-stage reranking with a cross-encoder.

    A bi-encoder (our dense retriever) embeds the query and each document separately, so it
    can search millions of documents quickly but only compares two fixed vectors. A
    cross-encoder reads the query and one document *together* and outputs a relevance score.
    It is far more accurate but costs one model run per (query, document) pair, so we only
    apply it to a short list of first-stage candidates.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        max_length: int = 512,
        scorer: Callable[[list[tuple[str, str]]], list[float]] | None = None,
    ) -> None:
        # `scorer` can replace the real model, e.g. in tests.
        self.model_name = model_name
        self.max_length = max_length
        self._scorer = scorer
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading cross-encoder %s", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score for each (query, text) pair. Higher is more relevant."""
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        if self._scorer is not None:
            return [float(s) for s in self._scorer(pairs)]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        doc_texts: dict[str, str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Re-sort candidates by cross-encoder score and keep the best top_k."""
        scores = self.score(query, [doc_texts[c.doc_id] for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda pair: -pair[0])
        return [SearchResult(c.chunk_id, c.doc_id, s) for s, c in ranked[:top_k]]
