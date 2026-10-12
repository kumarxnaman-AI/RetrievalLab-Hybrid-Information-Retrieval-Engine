import numpy as np
import pytest

from searchlab.retrieval.aggregate import aggregate_to_documents
from searchlab.models import SearchResult
from searchlab.retrieval.dense import cosine_similarity, top_k_indices


def test_cosine_similarity_known_values():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    sims = cosine_similarity(np.array([1.0, 0.0]), matrix)
    assert sims == pytest.approx([1.0, 0.0, 1 / np.sqrt(2), -1.0])


def test_cosine_is_scale_invariant():
    matrix = np.array([[1.0, 2.0, 3.0]])
    a = cosine_similarity(np.array([1.0, 1.0, 1.0]), matrix)
    b = cosine_similarity(np.array([10.0, 10.0, 10.0]), matrix)
    assert a == pytest.approx(b)


def test_top_k_indices_sorted_best_first():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert list(top_k_indices(scores, 2)) == [1, 3]


def test_top_k_larger_than_corpus():
    assert len(top_k_indices(np.array([0.2, 0.1]), 10)) == 2


def test_aggregate_keeps_best_chunk_per_document():
    results = [
        SearchResult("a#1", "a", 9.0),
        SearchResult("a#0", "a", 8.0),
        SearchResult("b#0", "b", 7.0),
    ]
    docs = aggregate_to_documents(results, top_k=5)
    assert [(r.doc_id, r.score) for r in docs] == [("a", 9.0), ("b", 7.0)]
