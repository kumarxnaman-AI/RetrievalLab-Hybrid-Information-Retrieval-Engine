from searchlab.models import SearchResult
from searchlab.retrieval.reranker import CrossEncoderReranker

DOC_TEXTS = {"a": "unrelated text", "b": "pension plan details", "c": "pension"}


def fake_scorer(pairs):
    # Pretend relevance = number of query words that appear in the document.
    return [sum(w in text for w in query.split()) for query, text in pairs]


def candidates():
    return [SearchResult(f"{d}#0", d, 0.0) for d in ("a", "b", "c")]


def test_rerank_sorts_by_cross_encoder_score():
    reranker = CrossEncoderReranker(scorer=fake_scorer)
    ranked = reranker.rerank("pension plan", candidates(), DOC_TEXTS, top_k=3)
    assert [r.doc_id for r in ranked] == ["b", "c", "a"]
    assert ranked[0].score == 2.0


def test_rerank_keeps_only_top_k():
    reranker = CrossEncoderReranker(scorer=fake_scorer)
    assert len(reranker.rerank("pension plan", candidates(), DOC_TEXTS, top_k=2)) == 2


def test_rerank_with_no_candidates():
    reranker = CrossEncoderReranker(scorer=fake_scorer)
    assert reranker.rerank("anything", [], DOC_TEXTS) == []


def test_score_passes_query_and_text_pairs():
    seen = []
    reranker = CrossEncoderReranker(scorer=lambda pairs: seen.extend(pairs) or [0.0] * len(pairs))
    reranker.score("q", ["t1", "t2"])
    assert seen == [("q", "t1"), ("q", "t2")]
