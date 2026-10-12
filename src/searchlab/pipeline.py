import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from searchlab.chunking import chunk_documents
from searchlab.config import Config
from searchlab.index.inverted_index import InvertedIndex
from searchlab.models import Chunk, Document
from searchlab.retrieval.bm25 import BM25Retriever
from searchlab.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


@dataclass
class Systems:
    chunks: list[Chunk]
    bm25: BM25Retriever
    dense: DenseRetriever
    bm25_build_seconds: float
    dense_build_seconds: float


def build_systems(
    docs: list[Document], config: Config, cache_dir: Path = Path(".cache")
) -> Systems:
    """Chunk documents and build both the BM25 index and the dense index."""
    chunks = chunk_documents(docs, config.chunk_size, config.chunk_overlap)

    start = time.perf_counter()
    index = InvertedIndex()
    index.build(chunks)
    bm25 = BM25Retriever(index, k1=config.bm25_k1, b=config.bm25_b)
    bm25_seconds = time.perf_counter() - start

    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", config.embedding_model)
    cache_path = cache_dir / (
        f"{model_slug}_{config.chunk_size}_{config.chunk_overlap}_{len(chunks)}.npz"
    )
    dense = DenseRetriever(config.embedding_model)
    dense_seconds = dense.build(chunks, cache_path=cache_path)

    return Systems(chunks, bm25, dense, bm25_seconds, dense_seconds)
