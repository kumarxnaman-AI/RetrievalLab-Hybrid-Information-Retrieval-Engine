"""The retrieval engine used by the API and dashboard: every retriever behind one object."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from searchlab.config import Config
from searchlab.evaluation.dataset import load_beir_dataset
from searchlab.models import Document, SearchResult
from searchlab.pipeline import build_hnsw, build_systems
from searchlab.retrieval.aggregate import aggregate_to_documents
from searchlab.retrieval.bm25 import BM25Retriever
from searchlab.retrieval.hybrid import HybridParams, HybridRetriever
from searchlab.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 280


@dataclass
class SearchParams:
    top_k: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    ef_search: int = 100
    dense_backend: str = "hnsw"  # hnsw | exact
    fusion_method: str = "rrf"  # rrf | weighted_rrf | score_fusion
    rrf_k: int = 60
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    alpha: float = 0.5
    candidates: int = 100
    rerank: bool = False
    rerank_pool: int = 20


@dataclass
class Panel:
    name: str
    score_label: str
    latency_ms: float
    results: list[dict] = field(default_factory=list)


class SearchEngine:
    def __init__(
        self,
        docs: list[Document],
        bm25: BM25Retriever,
        dense,
        hnsw,
        reranker: CrossEncoderReranker,
        n_chunks: int,
    ) -> None:
        self.docs = {d.doc_id: d for d in docs}
        self.bm25 = bm25
        self.dense = dense
        self.hnsw = hnsw
        self.reranker = reranker
        self.n_chunks = n_chunks
        self.doc_texts = {d.doc_id: d.text for d in docs}

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_documents(
        cls,
        docs: list[Document],
        config: Config | None = None,
        cache_dir: Path | None = None,
        encoder=None,
        reranker: CrossEncoderReranker | None = None,
        M: int = 16,
        ef_construction: int = 200,
    ) -> "SearchEngine":
        config = config or Config()
        systems = build_systems(docs, config, cache_dir=cache_dir, encoder=encoder)
        hnsw = build_hnsw(
            systems.dense, config, M=M, ef_construction=ef_construction, cache_dir=cache_dir
        )
        return cls(
            docs, systems.bm25, systems.dense, hnsw, reranker or CrossEncoderReranker(), len(systems.chunks)
        )

    @classmethod
    def from_dataset(cls, dataset_dir: Path, cache_dir: Path = Path(".cache")) -> "SearchEngine":
        docs, _, _ = load_beir_dataset(dataset_dir)
        return cls.from_documents(docs, Config(), cache_dir=cache_dir)

    # ------------------------------------------------------------------ helpers

    def _card(self, rank: int, result: SearchResult) -> dict:
        doc = self.docs[result.doc_id]
        text = doc.text
        return {
            "rank": rank,
            "doc_id": result.doc_id,
            "title": doc.metadata.get("title") or result.doc_id,
            "snippet": text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else ""),
            "score": round(result.score, 4),
        }

    def _panel(self, name, label, latency_ms, results, top_k) -> Panel:
        return Panel(name, label, round(latency_ms, 2), [self._card(i, r) for i, r in enumerate(results[:top_k], 1)])

    def _dense_search(self, params: SearchParams):
        if params.dense_backend == "exact":
            return lambda vec, n, ef: self.dense.search_vector(vec, n)
        return lambda vec, n, ef: self.hnsw.search_vector(vec, n, ef)

    def _hybrid_params(self, p: SearchParams) -> HybridParams:
        return HybridParams(
            method=p.fusion_method,
            rrf_k=p.rrf_k,
            bm25_weight=p.bm25_weight,
            dense_weight=p.dense_weight,
            alpha=p.alpha,
            candidates=p.candidates,
            ef_search=p.ef_search,
        )

    # ------------------------------------------------------------------ search

    def compare(self, query: str, params: SearchParams | None = None) -> list[Panel]:
        """BM25, dense and hybrid side by side (plus hybrid + reranker when enabled).

        The query is embedded once; that time is added to every panel that needs the
        embedding so each latency is what that pipeline would cost on its own.
        """
        p = params or SearchParams()
        bm25 = BM25Retriever(self.bm25.index, k1=p.bm25_k1, b=p.bm25_b)
        overfetch = HybridRetriever.CHUNK_OVERFETCH

        t = time.perf_counter()
        qvec = self.dense.encode([query])[0]
        encode_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        bm25_docs = aggregate_to_documents(bm25.search(query, p.top_k * overfetch * 2), p.top_k)
        bm25_ms = (time.perf_counter() - t) * 1000

        dense_search = self._dense_search(p)
        t = time.perf_counter()
        dense_docs = aggregate_to_documents(
            dense_search(qvec, p.top_k * overfetch * 2, max(p.ef_search, p.top_k * overfetch * 2)), p.top_k
        )
        dense_ms = (time.perf_counter() - t) * 1000 + encode_ms

        hybrid = HybridRetriever(bm25, dense_search, self.dense.encode)
        hp = self._hybrid_params(p)
        pool = max(p.top_k, p.rerank_pool) if p.rerank else p.top_k
        t = time.perf_counter()
        fused = hybrid.search(query, pool, hp, query_vec=qvec)
        hybrid_ms = (time.perf_counter() - t) * 1000 + encode_ms

        dense_name = "Dense (HNSW)" if p.dense_backend == "hnsw" else "Dense (exact)"
        fused_label = {"rrf": "RRF score", "weighted_rrf": "weighted RRF score", "score_fusion": "fused score"}[
            p.fusion_method
        ]
        panels = [
            self._panel("BM25", "score", bm25_ms, bm25_docs, p.top_k),
            self._panel(dense_name, "similarity", dense_ms, dense_docs, p.top_k),
            self._panel("Hybrid", fused_label, hybrid_ms, fused, p.top_k),
        ]
        if p.rerank:
            t = time.perf_counter()
            reranked = self.reranker.rerank(query, fused[:pool], self.doc_texts, p.top_k)
            rerank_ms = (time.perf_counter() - t) * 1000
            panels.append(
                self._panel(f"Hybrid + reranker (top {pool})", "reranker score", hybrid_ms + rerank_ms, reranked, p.top_k)
            )
        return panels

    def warm_up(self) -> None:
        """Load both models now, so the first real query doesn't pay the one-time cost."""
        self.dense.encode(["warm up"])
        self.reranker.score("warm up", ["warm up"])

    def info(self) -> dict:
        return {"documents": len(self.docs), "chunks": self.n_chunks, "index": self.hnsw.index.stats()}
