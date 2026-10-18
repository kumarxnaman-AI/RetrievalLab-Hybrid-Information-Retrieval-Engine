"""Phase 8: grounded question answering on top of the retrieval engine.

retrieve -> build a context within a token budget -> ask the LLM to answer from numbered
sources only -> verify the answer's citations and check each sentence is supported.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Callable

from searchlab.models import SearchResult
from searchlab.tokenizer import tokenize

REFUSAL = "I cannot answer from the provided sources."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources provided.
Rules:
- Every factual sentence must end with a citation like [1] or [1][2] naming the source(s) that support it.
- Use only information that appears in the sources. Do not add outside knowledge.
- If the sources do not contain the answer, reply with exactly: {REFUSAL}
- Be concise: at most 4 sentences."""

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")
MIN_EVALUABLE_TOKENS = 4  # shorter sentences ("Yes.", "In summary.") are not checked
SUPPORT_THRESHOLD = 0.5


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English). No tokenizer dependency."""
    return math.ceil(len(text) / 4)


@dataclass
class Source:
    n: int  # the citation number the LLM uses, starting at 1
    doc_id: str
    title: str
    passage: str
    score: float


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[Source]
    context_tokens: int
    token_budget: int
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    grounded_ratio: float | None = None  # supported / checkable sentences; None if nothing to check
    abstained: bool = False


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut text to roughly max_tokens, preferring a sentence boundary."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: boundary + 1] if boundary > max_chars // 2 else cut.rstrip() + "..."


def build_context(
    results: list[SearchResult],
    passage_of: Callable[[SearchResult], tuple[str, str]],
    token_budget: int = 1500,
    max_passage_tokens: int = 220,
) -> tuple[list[Source], str, int]:
    """Number the retrieved passages and fit as many as the token budget allows.

    `passage_of(result)` returns (title, passage). Sources are added in rank order; the last
    one is truncated to what is left, and a source with under 40 tokens of room is dropped.
    """
    sources, blocks, used = [], [], 0
    for result in results:
        title, passage = passage_of(result)
        header = f"[{len(sources) + 1}] {title}\n"
        room = token_budget - used - estimate_tokens(header)
        if room < 40:
            break
        passage = truncate_to_tokens(passage, min(max_passage_tokens, room))
        block = header + passage
        blocks.append(block)
        used += estimate_tokens(block) + 1
        sources.append(Source(len(sources) + 1, result.doc_id, title, passage, result.score))
    return sources, "\n\n".join(blocks), used


def extract_citations(text: str) -> list[int]:
    numbers = []
    for group in _CITATION.findall(text):
        numbers.extend(int(n) for n in re.split(r"\s*,\s*", group))
    return numbers


def check_grounding(answer: str, sources: list[Source]) -> dict:
    """Verify citations exist and that each checkable sentence is supported by what it cites.

    A sentence is supported when it cites at least one real source and at least half of its
    content words appear in the cited passages. This is a lexical check: it catches uncited or
    clearly unsupported claims, not subtle paraphrase errors.
    """
    valid = {s.n for s in sources}
    by_number = {s.n: s for s in sources}
    cited_all = extract_citations(answer)
    unsupported, checked, supported = [], 0, 0

    for sentence in _SENTENCE.split(answer):
        sentence = sentence.strip()
        words = set(tokenize(_CITATION.sub("", sentence)))
        if len(words) < MIN_EVALUABLE_TOKENS:
            continue
        checked += 1
        cites = [n for n in extract_citations(sentence) if n in valid]
        if not cites:
            unsupported.append(sentence)
            continue
        evidence = set()
        for n in cites:
            evidence |= set(tokenize(by_number[n].title + " " + by_number[n].passage))
        if len(words & evidence) / len(words) >= SUPPORT_THRESHOLD:
            supported += 1
        else:
            unsupported.append(sentence)

    return {
        "cited": sorted(set(n for n in cited_all if n in valid)),
        "invalid_citations": sorted(set(n for n in cited_all if n not in valid)),
        "unsupported_sentences": unsupported,
        "grounded_ratio": supported / checked if checked else None,
    }


class RAGPipeline:
    def __init__(
        self,
        retrieve: Callable[[str], list[SearchResult]],
        passage_of: Callable[[SearchResult], tuple[str, str]],
        llm,
        token_budget: int = 1500,
    ) -> None:
        self.retrieve = retrieve
        self.passage_of = passage_of
        self.llm = llm
        self.token_budget = token_budget

    def answer(self, question: str) -> RAGAnswer:
        results = self.retrieve(question)
        sources, context, used = build_context(results, self.passage_of, self.token_budget)
        if not sources:
            return RAGAnswer(question, REFUSAL, [], 0, self.token_budget, abstained=True)

        user = f"Sources:\n{context}\n\nQuestion: {question}"
        answer = self.llm.complete(SYSTEM_PROMPT, user, temperature=0.0, max_tokens=300)
        if REFUSAL.rstrip(".") in answer:
            return RAGAnswer(question, answer, sources, used, self.token_budget, abstained=True)

        return RAGAnswer(question, answer, sources, used, self.token_budget, **check_grounding(answer, sources))
