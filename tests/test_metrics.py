import math

import pytest

from searchlab.evaluation.metrics import (
    evaluate_run,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["d1", "d2", "d3"]
RELEVANCE = {"d2": 1, "d9": 1}  # d9 is relevant but never retrieved


def test_recall():
    assert recall_at_k(RANKED, RELEVANCE, 3) == pytest.approx(1 / 2)
    assert recall_at_k(RANKED, RELEVANCE, 1) == 0.0


def test_precision():
    assert precision_at_k(RANKED, RELEVANCE, 3) == pytest.approx(1 / 3)


def test_hit_rate():
    assert hit_rate_at_k(RANKED, RELEVANCE, 3) == 1.0
    assert hit_rate_at_k(RANKED, RELEVANCE, 1) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANCE, 3) == pytest.approx(1 / 2)
    assert reciprocal_rank(["x", "y"], RELEVANCE, 2) == 0.0


def test_ndcg():
    dcg = 1 / math.log2(3)  # relevant doc at rank 2
    ideal = 1 + 1 / math.log2(3)  # two relevant docs at ranks 1 and 2
    assert ndcg_at_k(RANKED, RELEVANCE, 3) == pytest.approx(dcg / ideal)


def test_perfect_ranking_has_ndcg_one():
    assert ndcg_at_k(["d2", "d9"], RELEVANCE, 2) == pytest.approx(1.0)


def test_evaluate_run_averages_over_queries():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": ["a"], "q2": ["x"]}  # q1 perfect, q2 total miss
    result = evaluate_run(run, qrels, k=1)
    assert result["Recall@1"] == pytest.approx(0.5)
    assert result["MRR@1"] == pytest.approx(0.5)
