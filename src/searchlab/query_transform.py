"""Query transformations for Phase 6: rewriting, HyDE and expansion.

Each transformation turns the user's query into the text sent to the lexical retriever
(`bm25_text`) and the text embedded for the dense retriever (`dense_text`).
"""

from dataclasses import dataclass

REWRITE_SYSTEM = (
    "You rewrite a user's query into a concise, self-contained search query for a scientific "
    "literature search engine. Keep every named entity, number, date and constraint from the "
    "original. Output only the rewritten query on one line, with no explanation."
)
HYDE_SYSTEM = (
    "You write a short hypothetical passage, in the style of a scientific paper abstract, that "
    "would directly address the given claim or question. Write 2 to 3 sentences. Output only the passage."
)
EXPAND_SYSTEM = (
    "You list search terms related to a query: synonyms, alternative names, abbreviations and "
    "closely related concepts. Output exactly 8 terms separated by commas, with no explanation."
)

STRATEGY_NAMES = ("original", "rewrite", "hyde", "expansion")


@dataclass(frozen=True)
class TransformedQuery:
    strategy: str
    bm25_text: str
    dense_text: str


def _one_line(text: str) -> str:
    return " ".join(text.split()).strip("\"' ")


def rewrite_query(llm, query: str) -> str:
    return _one_line(llm.complete(REWRITE_SYSTEM, query, max_tokens=100)) or query


def hypothetical_document(llm, query: str) -> str:
    return _one_line(llm.complete(HYDE_SYSTEM, query, max_tokens=200)) or query


def expansion_terms(llm, query: str) -> str:
    return _one_line(llm.complete(EXPAND_SYSTEM, query, max_tokens=80))


def transform(llm, query: str, strategy: str) -> TransformedQuery:
    """Apply one strategy.

    original   the query unchanged (the baseline)
    rewrite    LLM-rewritten query for both retrievers
    hyde       dense retriever embeds an LLM-written hypothetical passage; BM25 keeps the
               original query (HyDE changes only the dense side)
    expansion  original query plus LLM-suggested related terms, for both retrievers
    """
    if strategy == "original":
        return TransformedQuery(strategy, query, query)
    if strategy == "rewrite":
        rewritten = rewrite_query(llm, query)
        return TransformedQuery(strategy, rewritten, rewritten)
    if strategy == "hyde":
        return TransformedQuery(strategy, query, hypothetical_document(llm, query))
    if strategy == "expansion":
        expanded = f"{query} {expansion_terms(llm, query)}".strip()
        return TransformedQuery(strategy, expanded, expanded)
    raise ValueError(f"strategy must be one of {STRATEGY_NAMES}, got {strategy!r}")
