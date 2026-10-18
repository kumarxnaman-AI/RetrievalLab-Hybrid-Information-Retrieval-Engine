import pytest

from searchlab.models import SearchResult
from searchlab.retrieval.fusion import (
    fuse_results,
    min_max_normalize,
    reciprocal_rank_fusion,
    score_fusion,
)


def test_rrf_matches_hand_calculation():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement_between_rankers():
    ranked = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]])
    assert [doc for doc, _ in ranked][0] == "b"  # ranked 2nd and 1st beats 1st and absent


def test_rrf_weights_shift_the_winner():
    rankings = [["a"], ["b"]]  # each retriever has a different favourite
    assert reciprocal_rank_fusion(rankings, weights=[2.0, 1.0])[0][0] == "a"
    assert reciprocal_rank_fusion(rankings, weights=[1.0, 2.0])[0][0] == "b"


def test_rrf_validates_arguments():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_ties_are_broken_deterministically():
    ranked = reciprocal_rank_fusion([["b"], ["a"]])
    assert [doc for doc, _ in ranked] == ["a", "b"]


def test_min_max_normalize():
    assert min_max_normalize({"a": 10.0, "b": 20.0, "c": 30.0}) == {"a": 0.0, "b": 0.5, "c": 1.0}
    assert min_max_normalize({"a": 5.0, "b": 5.0}) == {"a": 1.0, "b": 1.0}
    assert min_max_normalize({}) == {}


def test_score_fusion_extremes_follow_one_retriever():
    dense = {"a": 0.9, "b": 0.5}
    bm25 = {"b": 30.0, "c": 10.0}
    assert score_fusion(dense, bm25, alpha=1.0)[0][0] == "a"
    assert score_fusion(dense, bm25, alpha=0.0)[0][0] == "b"


def test_score_fusion_hand_calculation():
    fused = dict(score_fusion({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 10.0}, alpha=0.7))
    assert fused["a"] == pytest.approx(0.7)
    assert fused["b"] == pytest.approx(0.3)


def test_score_fusion_validates_alpha():
    with pytest.raises(ValueError):
        score_fusion({"a": 1.0}, {"a": 1.0}, alpha=1.5)


def results(*pairs):
    return [SearchResult(f"{doc}#0", doc, score) for doc, score in pairs]


def test_fuse_results_returns_search_results_with_chunk_ids():
    bm25 = results(("a", 9.0), ("b", 5.0))
    dense = results(("b", 0.9), ("c", 0.8))
    fused = fuse_results(bm25, dense, method="rrf")
    assert fused[0].doc_id == "b"
    assert fused[0].chunk_id == "b#0"
    assert {r.doc_id for r in fused} == {"a", "b", "c"}


@pytest.mark.parametrize("method", ["rrf", "weighted_rrf", "score_fusion"])
def test_fuse_results_supports_every_method(method):
    fused = fuse_results(results(("a", 1.0)), results(("a", 1.0)), method=method)
    assert [r.doc_id for r in fused] == ["a"]


def test_fuse_results_rejects_unknown_method():
    with pytest.raises(ValueError):
        fuse_results([], [], method="magic")
