"""Phase 10: when does LLM query rewriting harm retrieval?

Reads the per-query results written by `query_benchmark`, classifies every query as
helped / neutral / harmed by the rewrite, and checks whether specific rewrite behaviours are
associated with harm. Flags are cheap, transparent heuristics (not LLM judgments), so they can be
audited and reproduced, but they are approximate.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from searchlab.hybrid_benchmark import table
from searchlab.tokenizer import STOPWORDS

EPSILON = 0.01

CONSTRAINT_WORDS = frozenset(
    "not no never without only except least most more less fewer greater higher lower than "
    "before after during between within over under".split()
)
FLAGS = (
    "entity_removed",
    "constraint_removed",
    "number_or_date_changed",
    "generalized",
    "intent_changed",
)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def entity_tokens(query: str) -> set[str]:
    """Likely named entities: acronyms, tokens with digits, and capitalized words after the first."""
    found = set()
    for position, word in enumerate(words(query)):
        has_digit = any(c.isdigit() for c in word)
        is_acronym = len(word) > 1 and word.isupper()
        is_capitalized = position > 0 and word[0].isupper()
        if has_digit or is_acronym or is_capitalized:
            found.add(word.lower())
    return found


def content_tokens(text: str) -> set[str]:
    return {w.lower() for w in words(text) if w.lower() not in STOPWORDS}


def rewrite_flags(original: str, rewrite: str, similarity: float | None = None) -> dict[str, bool]:
    """Which kinds of meaning drift does this rewrite show relative to the original query?"""
    orig_words, new_words = {w.lower() for w in words(original)}, {w.lower() for w in words(rewrite)}
    orig_content, new_content = content_tokens(original), content_tokens(rewrite)
    kept = len(orig_content & new_content) / len(orig_content) if orig_content else 1.0
    return {
        "entity_removed": bool(entity_tokens(original) - new_words),
        "constraint_removed": bool((CONSTRAINT_WORDS & orig_words) - new_words),
        "number_or_date_changed": set(_NUMBER.findall(original)) != set(_NUMBER.findall(rewrite)),
        "generalized": kept < 0.7,
        "intent_changed": similarity is not None and similarity < 0.75,
    }


def classify(delta: float) -> str:
    if delta > EPSILON:
        return "helped"
    if delta < -EPSILON:
        return "harmed"
    return "neutral"


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="When does query rewriting harm retrieval?")
    parser.add_argument("--details", default="results/query_transform_details.json")
    parser.add_argument("--system", default="Hybrid", choices=["BM25", "Dense", "Hybrid"])
    parser.add_argument("--out", default="results/rewrite_analysis.md")
    args = parser.parse_args()

    data = json.loads(Path(args.details).read_text(encoding="utf-8"))
    queries, transformed, metrics = data["queries"], data["transformed"], data["metrics"]

    # Embedding similarity between original and rewrite, to detect intent drift.
    from searchlab.config import Config
    from searchlab.retrieval.dense import DenseRetriever

    encoder = DenseRetriever(Config().embedding_model)
    qids = list(queries)
    originals = encoder.encode([queries[q] for q in qids])
    rewrites = encoder.encode([transformed["rewrite"][q]["dense_text"] for q in qids])
    cosine = (originals * rewrites).sum(axis=1) / (
        np.linalg.norm(originals, axis=1) * np.linalg.norm(rewrites, axis=1)
    )
    similarity = dict(zip(qids, cosine.tolist()))

    rows = []
    for qid in qids:
        base = metrics["original"][args.system][qid]["ndcg"]
        new = metrics["rewrite"][args.system][qid]["ndcg"]
        rewrite_text = transformed["rewrite"][qid]["dense_text"]
        rows.append(
            {
                "qid": qid,
                "original": queries[qid],
                "rewrite": rewrite_text,
                "delta": new - base,
                "outcome": classify(new - base),
                "flags": rewrite_flags(queries[qid], rewrite_text, similarity[qid]),
                "similarity": similarity[qid],
            }
        )

    n = len(rows)
    counts = {o: sum(r["outcome"] == o for r in rows) for o in ("helped", "neutral", "harmed")}
    outcome_table = table(
        ["Outcome", "Queries", "Share", "Mean Δ nDCG@10"],
        [
            [
                o,
                counts[o],
                pct(counts[o], n),
                f"{np.mean([r['delta'] for r in rows if r['outcome'] == o]):+.3f}" if counts[o] else "-",
            ]
            for o in ("helped", "neutral", "harmed")
        ],
    )

    flag_rows = []
    for flag in FLAGS:
        with_flag = [r for r in rows if r["flags"][flag]]
        without = [r for r in rows if not r["flags"][flag]]
        harmed_with = sum(r["outcome"] == "harmed" for r in with_flag)
        harmed_without = sum(r["outcome"] == "harmed" for r in without)
        flag_rows.append(
            [
                flag,
                len(with_flag),
                f"{harmed_with} ({pct(harmed_with, len(with_flag))})",
                f"{harmed_without} ({pct(harmed_without, len(without))})",
                f"{np.mean([r['delta'] for r in with_flag]):+.3f}" if with_flag else "-",
            ]
        )
    flag_table = table(
        ["Rewrite behaviour", "Queries flagged", "Harmed when flagged", "Harmed when not flagged", "Mean Δ nDCG when flagged"],
        flag_rows,
    )

    def example_block(outcome, limit=8):
        chosen = sorted((r for r in rows if r["outcome"] == outcome), key=lambda r: r["delta"], reverse=(outcome == "helped"))[:limit]
        lines = []
        for r in chosen:
            active = ", ".join(f for f, on in r["flags"].items() if on) or "none"
            lines.append(
                f"- **Δ {r['delta']:+.2f}** | original: \"{r['original']}\" | rewrite: \"{r['rewrite']}\" | flags: {active}"
            )
        return "\n".join(lines) or "_none_"

    report = f"""# When does query rewriting harm retrieval?

System analyzed: **{args.system}** retrieval, {n} queries, rewriting model `{data['model']}`.
Each query is classified by the change in nDCG@10 from using the LLM rewrite instead of the original query
(changes within ±{EPSILON} count as neutral).

## Outcomes

{outcome_table}

## Is a rewrite behaviour associated with harm?

Flags are simple heuristics: *entity_removed* (an acronym, digit-bearing token or capitalized term from the original is
absent from the rewrite), *constraint_removed* (a negation or comparison word such as "not", "only", "more than", "before"
disappears), *number_or_date_changed*, *generalized* (the rewrite keeps under 70% of the original's content words), and
*intent_changed* (embedding cosine between original and rewrite is below 0.75). A query can carry several flags.

{flag_table}

Small groups: treat differences of a few queries as anecdotal rather than evidence.

## Most harmed

{example_block('harmed')}

## Most helped

{example_block('helped')}
"""
    print("\n" + report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
