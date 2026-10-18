"""Chunking strategies compared in the Phase 7 experiment.

fixed      non-overlapping character windows
sliding    overlapping character windows (stride = half the window)
recursive  structure-aware: paragraphs, then sentences, then words; never cuts mid-sentence
           unless a single sentence is longer than the chunk size
semantic   splits where the topic shifts, detected from embedding similarity between
           adjacent sentences
"""

import logging
import re
from typing import Callable

import numpy as np

from searchlab.chunking import chunk_documents
from searchlab.cleaning import clean_text
from searchlab.models import Chunk, Document

logger = logging.getLogger(__name__)

STRATEGIES = ("fixed", "sliding", "recursive", "semantic")

# A sentence ends at . ! ? followed by whitespace and something that looks like a new sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _make_chunks(doc: Document, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(f"{doc.doc_id}#{i}", doc.doc_id, text, i, dict(doc.metadata))
        for i, text in enumerate(texts)
        if text.strip()
    ]


# ---------------------------------------------------------------------- recursive


def _split_long(sentence: str, size: int) -> list[str]:
    """Last resort for a sentence longer than `size`: cut on word boundaries."""
    words, pieces, current = sentence.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_recursive(doc: Document, size: int = 500) -> list[Chunk]:
    """Pack whole sentences into chunks of at most `size` characters, respecting paragraphs.

    Paragraph breaks are preferred cut points: a chunk is closed at a paragraph boundary
    once it is at least half full.
    """
    text = clean_text(doc.text)
    units: list[tuple[str, bool]] = []  # (text, starts_new_paragraph)
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        first = True
        for sentence in split_sentences(paragraph):
            for piece in _split_long(sentence, size) if len(sentence) > size else [sentence]:
                units.append((piece, first))
                first = False

    chunks, current = [], ""
    for piece, starts_paragraph in units:
        joined_len = len(current) + 1 + len(piece)
        if current and (joined_len > size or (starts_paragraph and len(current) >= size // 2)):
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return _make_chunks(doc, chunks)


# ---------------------------------------------------------------------- semantic


def chunk_semantic(
    docs: list[Document],
    encode: Callable[[list[str]], np.ndarray],
    max_size: int = 800,
    min_size: int = 150,
    percentile: float = 25.0,
) -> list[Chunk]:
    """Start a new chunk where adjacent sentences are unusually dissimilar.

    All sentences are embedded once. The boundary threshold is the `percentile`-th
    percentile of adjacent-sentence cosine similarity over the whole corpus, so roughly
    that share of sentence gaps becomes a cut. Chunks are capped at `max_size` characters
    and chunks shorter than `min_size` are merged into the previous one.
    """
    per_doc = [split_sentences(clean_text(d.text)) for d in docs]
    flat = [s for sentences in per_doc for s in sentences]
    if not flat:
        return []
    vectors = np.asarray(encode(flat), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    # Cosine similarity between each sentence and the next one within the same document.
    sims_per_doc, offset, all_sims = [], 0, []
    for sentences in per_doc:
        n = len(sentences)
        sims = (vectors[offset : offset + n - 1] * vectors[offset + 1 : offset + n]).sum(axis=1) if n > 1 else np.array([])
        sims_per_doc.append(sims)
        all_sims.extend(sims.tolist())
        offset += n
    threshold = float(np.percentile(all_sims, percentile)) if all_sims else -1.0
    logger.info("semantic chunking: %d sentences, boundary threshold %.3f", len(flat), threshold)

    chunks: list[Chunk] = []
    for doc, sentences, sims in zip(docs, per_doc, sims_per_doc):
        groups: list[str] = []
        current = ""
        for i, sentence in enumerate(sentences):
            if not current:
                current = sentence
                continue
            topic_shift = sims[i - 1] < threshold
            if topic_shift or len(current) + 1 + len(sentence) > max_size:
                groups.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}"
        if current:
            groups.append(current)

        merged: list[str] = []
        for group in groups:
            if merged and len(group) < min_size and len(merged[-1]) + 1 + len(group) <= max_size:
                merged[-1] = f"{merged[-1]} {group}"
            else:
                merged.append(group)
        chunks.extend(_make_chunks(doc, merged))
    return chunks


# ---------------------------------------------------------------------- entry point


def build_chunks(
    strategy: str,
    docs: list[Document],
    size: int = 500,
    encode: Callable[[list[str]], np.ndarray] | None = None,
) -> list[Chunk]:
    """Chunk a whole corpus with the named strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if strategy == "fixed":
        return chunk_documents(docs, size, 0)
    if strategy == "sliding":
        return chunk_documents(docs, size, size // 2)
    if strategy == "recursive":
        return [c for doc in docs for c in chunk_recursive(doc, size)]
    if encode is None:
        raise ValueError("semantic chunking needs an `encode` function")
    return chunk_semantic(docs, encode, max_size=int(size * 1.6))
