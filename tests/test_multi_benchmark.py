from searchlab.multi_benchmark import categorize, has_specific_terms


def test_bm25_wins_when_it_beats_dense_by_the_margin():
    labels = categorize(bm25=0.9, dense=0.5, hybrid=0.8)
    assert "BM25 wins" in labels
    assert "Dense wins" not in labels


def test_dense_wins():
    assert "Dense wins" in categorize(bm25=0.2, dense=0.7, hybrid=0.6)


def test_small_gaps_are_not_wins():
    assert categorize(0.50, 0.55, 0.52) == set()


def test_hybrid_wins_only_when_it_beats_both():
    assert "Hybrid wins" in categorize(0.5, 0.6, 0.8)
    assert "Hybrid wins" not in categorize(0.5, 0.9, 0.9)


def test_hybrid_can_lose_to_the_best_single_retriever():
    labels = categorize(bm25=0.9, dense=0.3, hybrid=0.6)
    assert "Hybrid loses to the best single retriever" in labels
    assert "BM25 wins" in labels  # labels overlap by design


def test_all_systems_fail():
    assert "All systems fail" in categorize(0.0, 0.05, 0.0)
    assert "All systems fail" not in categorize(0.0, 0.5, 0.0)


def test_has_specific_terms():
    assert has_specific_terms("effect of BRCA1 on risk")
    assert has_specific_terms("deaths in 2010")
    assert not has_specific_terms("does exercise help with sleep")
