from searchlab.rewrite_analysis import classify, content_tokens, entity_tokens, rewrite_flags
from searchlab.query_benchmark import sample_queries


def test_entity_tokens_finds_acronyms_digits_and_mid_sentence_capitals():
    found = entity_tokens("does BRCA1 raise risk in Chinese women with HbA1c above 7")
    assert {"brca1", "chinese", "hba1c"} <= found
    assert "does" not in found  # first word capitalization is not evidence of an entity


def test_content_tokens_drop_stopwords():
    assert content_tokens("the risk of the disease") == {"risk", "disease"}


def test_faithful_rewrite_raises_no_flags():
    flags = rewrite_flags("do statins reduce cardiovascular risk", "statins reduce cardiovascular risk", similarity=0.95)
    assert not any(flags.values())


def test_entity_removal_is_flagged():
    flags = rewrite_flags("BRCA1 mutations and breast cancer", "gene mutations and breast cancer")
    assert flags["entity_removed"]


def test_constraint_removal_is_flagged():
    flags = rewrite_flags("aspirin does not reduce stroke risk", "aspirin stroke risk")
    assert flags["constraint_removed"]


def test_number_change_is_flagged_for_removed_and_added_numbers():
    assert rewrite_flags("deaths in 2010 cohort", "deaths in cohort")["number_or_date_changed"]
    assert rewrite_flags("deaths in 2010 cohort", "deaths in 2012 cohort")["number_or_date_changed"]
    assert not rewrite_flags("deaths in 2010 cohort", "mortality in 2010 cohort")["number_or_date_changed"]


def test_generalization_is_flagged_when_most_content_words_vanish():
    flags = rewrite_flags("vitamin D supplementation reduces fracture incidence elderly", "vitamin health")
    assert flags["generalized"]


def test_intent_change_uses_embedding_similarity_when_given():
    assert rewrite_flags("a", "b", similarity=0.5)["intent_changed"]
    assert not rewrite_flags("a", "b", similarity=0.9)["intent_changed"]
    assert not rewrite_flags("a", "b")["intent_changed"]


def test_classify_uses_an_epsilon_band():
    assert classify(0.2) == "helped"
    assert classify(-0.2) == "harmed"
    assert classify(0.005) == "neutral"
    assert classify(-0.01) == "neutral"


def test_sample_queries_is_deterministic_and_bounded():
    queries = {str(i): f"q{i}" for i in range(50)}
    a, b = sample_queries(queries, 10), sample_queries(queries, 10)
    assert a == b and len(a) == 10
    assert sample_queries(queries, None) == queries
