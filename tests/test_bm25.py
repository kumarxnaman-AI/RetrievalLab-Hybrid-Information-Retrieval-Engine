import math

import pytest

from searchlab.index.inverted_index import InvertedIndex
from searchlab.models import Chunk
from searchlab.retrieval.bm25 import BM25Retriever


def make_retriever(k1=1.5, b=0.75):
    index = InvertedIndex()
    index.build(
        [
            Chunk("a#0", "a", "pension plan pension", 0),
            Chunk("b#0", "b", "leave policy", 0),
        ]
    )
    return BM25Retriever(index, k1=k1, b=b)


def test_score_matches_hand_calculation():
    # N=2, df=1 -> idf = ln(2); tf=2; len=3; avg_len=2.5
    idf = math.log(2)
    length_norm = 1 - 0.75 + 0.75 * 3 / 2.5
    expected = idf * 2 * 2.5 / (2 + 1.5 * length_norm)

    results = make_retriever().search("pension")

    assert len(results) == 1
    assert results[0].chunk_id == "a#0"
    assert results[0].score == pytest.approx(expected)


def test_ranks_more_relevant_chunk_first():
    index = InvertedIndex()
    index.build(
        [
            Chunk("a#0", "a", "pension pension pension plan", 0),
            Chunk("b#0", "b", "pension leave policy rules", 0),
            Chunk("c#0", "c", "unrelated words only", 0),
        ]
    )
    results = BM25Retriever(index).search("pension")
    assert [r.chunk_id for r in results] == ["a#0", "b#0"]


def test_unknown_terms_return_nothing():
    assert make_retriever().search("zebra") == []


def test_top_k_limits_results():
    index = InvertedIndex()
    index.build([Chunk(f"d{i}#0", f"d{i}", "shared term", 0) for i in range(5)])
    assert len(BM25Retriever(index).search("shared", top_k=3)) == 3


def test_b_zero_disables_length_normalization():
    index = InvertedIndex()
    index.build(
        [
            Chunk("short#0", "short", "term", 0),
            Chunk("long#0", "long", "term filler filler filler filler", 0),
        ]
    )
    scores = {r.chunk_id: r.score for r in BM25Retriever(index, b=0).search("term")}
    assert scores["short#0"] == pytest.approx(scores["long#0"])


def test_invalid_parameters_raise():
    index = InvertedIndex()
    with pytest.raises(ValueError):
        BM25Retriever(index, k1=-1)
    with pytest.raises(ValueError):
        BM25Retriever(index, b=1.5)
