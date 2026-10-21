from searchlab.chunking_benchmark import subset_corpus
from searchlab.models import Document


def corpus(n=50):
    return [Document(f"d{i}", f"text {i}") for i in range(n)]


def test_subset_always_keeps_every_relevant_document():
    qrels = {"q1": {"d3": 1, "d40": 2}, "q2": {"d7": 1, "d9": 0}}
    kept = {d.doc_id for d in subset_corpus(corpus(), qrels, max_docs=10)}
    assert {"d3", "d40", "d7"} <= kept
    assert "d9" not in kept or len(kept) == 10  # grade 0 is not relevant; it only appears as a random distractor


def test_subset_respects_the_size_cap_with_distractors():
    kept = subset_corpus(corpus(), {"q1": {"d1": 1}}, max_docs=12)
    assert len(kept) == 12


def test_subset_never_drops_relevant_docs_to_hit_the_cap():
    qrels = {"q": {f"d{i}": 1 for i in range(20)}}
    assert len(subset_corpus(corpus(), qrels, max_docs=5)) == 20


def test_subset_is_deterministic():
    qrels = {"q1": {"d1": 1}}
    a = [d.doc_id for d in subset_corpus(corpus(), qrels, 15)]
    b = [d.doc_id for d in subset_corpus(corpus(), qrels, 15)]
    assert a == b
