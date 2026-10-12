import math

import pytest

from searchlab.index.inverted_index import InvertedIndex
from searchlab.models import Chunk


def make_index():
    index = InvertedIndex()
    index.build(
        [
            Chunk("a#0", "a", "pension plan pension", 0),
            Chunk("b#0", "b", "leave policy", 0),
        ]
    )
    return index


def test_posting_lists_store_term_frequencies():
    index = make_index()
    assert index.postings["pension"] == {"a#0": 2}
    assert index.postings["policy"] == {"b#0": 1}


def test_chunk_lengths_and_average():
    index = make_index()
    assert index.chunk_lengths == {"a#0": 3, "b#0": 2}
    assert index.avg_chunk_length == 2.5


def test_document_frequency_and_idf():
    index = make_index()
    assert index.doc_freq("pension") == 1
    assert index.doc_freq("missing") == 0
    assert index.idf("pension") == pytest.approx(math.log(1 + (2 - 1 + 0.5) / (1 + 0.5)))


def test_rarer_terms_have_higher_idf():
    index = InvertedIndex()
    index.build(
        [
            Chunk("a#0", "a", "common rare", 0),
            Chunk("b#0", "b", "common", 0),
            Chunk("c#0", "c", "common", 0),
        ]
    )
    assert index.idf("rare") > index.idf("common")


def test_duplicate_chunk_id_raises():
    index = make_index()
    with pytest.raises(ValueError):
        index.add_chunk(Chunk("a#0", "a", "again", 0))
