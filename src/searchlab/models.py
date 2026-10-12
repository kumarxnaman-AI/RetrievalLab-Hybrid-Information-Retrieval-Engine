from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int  # position of this chunk within its document (0, 1, 2, ...)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    doc_id: str
    score: float
