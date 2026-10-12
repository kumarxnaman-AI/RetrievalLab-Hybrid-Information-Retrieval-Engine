from dataclasses import dataclass
from pathlib import Path

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class Config:
    # Paths
    data_dir: Path = Path("data")

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Retrieval
    top_k: int = 10

    # Dense retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Logging
    log_level: str = "INFO"

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")
        if self.bm25_k1 < 0:
            raise ValueError("bm25_k1 must be non-negative")
        if not 0 <= self.bm25_b <= 1:
            raise ValueError("bm25_b must be between 0 and 1")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.log_level.upper() not in VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {VALID_LOG_LEVELS}, got {self.log_level!r}"
            )

