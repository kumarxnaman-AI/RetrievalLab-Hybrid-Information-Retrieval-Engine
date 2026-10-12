import pytest

from searchlab.chunking import chunk_document
from searchlab.models import Document


def test_fixed_size_chunks_with_overlap():
    doc = Document("d", "abcdefghijklmnopqrstuvwxy")  # 25 chars, no whitespace

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)

    assert [c.text for c in chunks] == ["abcdefghij", "ijklmnopqr", "qrstuvwxy"]
    assert [c.chunk_id for c in chunks] == ["d#0", "d#1", "d#2"]
    assert all(c.doc_id == "d" for c in chunks)


def test_short_document_is_a_single_chunk():
    chunks = chunk_document(Document("d", "short text"), 100, 10)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_empty_document_has_no_chunks():
    assert chunk_document(Document("d", "   \n  "), 100, 10) == []


def test_chunk_metadata_keeps_document_metadata_and_start_char():
    doc = Document("d", "abcdefghijklmnop", {"source": "x.txt"})
    chunks = chunk_document(doc, 10, 0)
    assert chunks[1].metadata == {"source": "x.txt", "start_char": 10}


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_document(Document("d", "abc"), chunk_size=10, chunk_overlap=10)
