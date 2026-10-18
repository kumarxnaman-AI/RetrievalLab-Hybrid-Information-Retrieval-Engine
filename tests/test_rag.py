from searchlab.models import SearchResult
from searchlab.rag import (
    REFUSAL,
    RAGPipeline,
    Source,
    build_context,
    check_grounding,
    estimate_tokens,
    extract_citations,
    truncate_to_tokens,
)

PASSAGES = {
    "d1": ("Statins", "Statins reduce LDL cholesterol and lower cardiovascular risk in adults."),
    "d2": ("Aspirin", "Aspirin is used for secondary prevention of stroke in some patients."),
    "d3": ("Vitamin D", "Vitamin D supplementation has mixed effects on fracture risk."),
}


def results(*doc_ids):
    return [SearchResult(f"{d}#0", d, 1.0 - i * 0.1) for i, d in enumerate(doc_ids)]


def passage_of(result):
    return PASSAGES[result.doc_id]


class StubLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, system, user, temperature=0.0, max_tokens=300):
        self.prompts.append((system, user))
        return self.reply


def test_estimate_tokens_is_about_four_characters_each():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 0


def test_truncate_prefers_sentence_boundary():
    text = "First sentence is here. Second sentence goes on for a good while longer than the limit."
    cut = truncate_to_tokens(text, 8)  # 32 chars
    assert cut == "First sentence is here."


def test_truncate_leaves_short_text_alone():
    assert truncate_to_tokens("short", 100) == "short"


def test_build_context_numbers_sources_from_one():
    sources, context, used = build_context(results("d1", "d2"), passage_of, token_budget=1000)
    assert [s.n for s in sources] == [1, 2]
    assert "[1] Statins" in context and "[2] Aspirin" in context
    assert used <= 1000


def test_build_context_respects_the_token_budget():
    sources, context, used = build_context(results("d1", "d2", "d3"), passage_of, token_budget=60)
    assert len(sources) < 3
    assert used <= 60


def test_build_context_with_no_room_returns_nothing():
    assert build_context(results("d1"), passage_of, token_budget=10)[0] == []


def test_extract_citations_handles_adjacent_and_comma_forms():
    assert extract_citations("A [1][2]. B [3, 4].") == [1, 2, 3, 4]
    assert extract_citations("no citations") == []


def sources():
    return [Source(1, "d1", *PASSAGES["d1"], 1.0), Source(2, "d2", *PASSAGES["d2"], 0.9)]


def test_supported_cited_sentence_passes():
    check = check_grounding("Statins reduce LDL cholesterol and lower cardiovascular risk [1].", sources())
    assert check["unsupported_sentences"] == []
    assert check["grounded_ratio"] == 1.0
    assert check["cited"] == [1]


def test_uncited_sentence_is_flagged():
    check = check_grounding("Statins reduce LDL cholesterol and lower cardiovascular risk.", sources())
    assert len(check["unsupported_sentences"]) == 1
    assert check["grounded_ratio"] == 0.0


def test_citation_that_does_not_support_the_claim_is_flagged():
    check = check_grounding("Statins reduce LDL cholesterol and lower cardiovascular risk [2].", sources())
    assert len(check["unsupported_sentences"]) == 1


def test_invented_citation_number_is_reported():
    check = check_grounding("Statins reduce LDL cholesterol and lower cardiovascular risk [1][9].", sources())
    assert check["invalid_citations"] == [9]
    assert check["unsupported_sentences"] == []  # the valid [1] still supports it


def test_mixed_answer_gives_partial_ratio():
    answer = (
        "Statins reduce LDL cholesterol and lower cardiovascular risk [1]. "
        "Moreover they definitely cure every known disease completely."
    )
    check = check_grounding(answer, sources())
    assert check["grounded_ratio"] == 0.5


def test_pipeline_returns_cited_grounded_answer():
    llm = StubLLM("Statins reduce LDL cholesterol and lower cardiovascular risk [1].")
    pipeline = RAGPipeline(lambda q: results("d1", "d2"), passage_of, llm, token_budget=500)
    answer = pipeline.answer("do statins help")
    assert answer.cited == [1]
    assert answer.grounded_ratio == 1.0
    assert not answer.abstained
    assert "Question: do statins help" in llm.prompts[0][1]
    assert "[1] Statins" in llm.prompts[0][1]


def test_pipeline_passes_through_a_refusal():
    pipeline = RAGPipeline(lambda q: results("d1"), passage_of, StubLLM(REFUSAL))
    answer = pipeline.answer("what is the capital of France")
    assert answer.abstained
    assert answer.grounded_ratio is None


def test_pipeline_with_no_retrieved_sources_never_calls_the_llm():
    llm = StubLLM("should not be used")
    answer = RAGPipeline(lambda q: [], passage_of, llm).answer("anything")
    assert answer.abstained and answer.answer == REFUSAL
    assert llm.prompts == []
