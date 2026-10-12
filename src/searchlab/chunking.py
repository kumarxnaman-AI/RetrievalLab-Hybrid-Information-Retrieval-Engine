import logging

from searchlab.cleaning import clean_text
from searchlab.models import Chunk, Document

logger = logging.getLogger(__name__)


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split one document into fixed-size character windows that overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")

    text = clean_text(doc.text)
    step = chunk_size - chunk_overlap
    chunks = []
    for index, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{index}",
                    doc_id=doc.doc_id,
                    text=piece,
                    chunk_index=index,
                    metadata={**doc.metadata, "start_char": start},
                )
            )
        if start + chunk_size >= len(text):
            break
    return chunks


def chunk_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    logger.info("created %d chunks from %d documents", len(chunks), len(docs))
    return chunks
