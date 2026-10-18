import numpy as np
import pytest

from searchlab.chunkers import (
    STRATEGIES,
    build_chunks,
    chunk_recursive,
    chunk_semantic,
    split_sentences,
)
from searchlab.models import Document

TEXT = (
    "Statins lower cholesterol. They reduce cardiovascular risk. Doses vary by patient. "
    "Separately, the study enrolled forty volunteers. Volunteers were randomized in two arms."
)


def test_split_sentences():
    assert split_sentences("One thing. Two things! Is it three? Yes.") == [
        "One thing.",
        "Two things!",
        "Is it three?",
        "Yes.",
    ]


def test_recursive_never_cuts_mid_sentence():
    chunks = chunk_recursive(Document("d", TEXT), size=90)
    assert len(chunks) > 1
    sentences = set(split_sentences(TEXT))
    for chunk in chunks:
        assert all(s in sentences for s in split_sentences(chunk.text))
        assert len(chunk.text) <= 90


def test_recursive_keeps_all_text():
    chunks = chunk_recursive(Document("d", TEXT), size=90)
    assert " ".join(c.text for c in chunks) == TEXT


def test_recursive_prefers_paragraph_boundaries():
    text = "First paragraph sentence one here. Sentence two goes here.\n\nSecond paragraph begins now."
    chunks = chunk_recursive(Document("d", text), size=100)
    # Everything would fit in 100 chars, but the first paragraph is over half full,
    # so the paragraph break closes the chunk.
    assert len(chunks) == 2
    assert chunks[1].text == "Second paragraph begins now."


def test_recursive_merges_a_tiny_paragraph_with_the_next():
    chunks = chunk_recursive(Document("d", "Tiny.\n\nSecond paragraph begins now."), size=100)
    assert len(chunks) == 1


def test_recursive_splits_an_overlong_sentence_on_words():
    long_sentence = " ".join(["word"] * 100) + "."
    chunks = chunk_recursive(Document("d", long_sentence), size=60)
    assert all(len(c.text) <= 60 for c in chunks)
    assert len(chunks) > 1


def topic_encoder(texts):
    """Two fake topics: sentences mentioning 'statin' vs everything else."""
    return np.array([[1.0, 0.0] if "statin" in t.lower() else [0.0, 1.0] for t in texts], dtype=np.float32)


def test_semantic_splits_where_the_topic_changes():
    doc = Document(
        "d",
        "Statins help. Statin doses vary. Statin side effects exist. "
        "Volunteers were recruited. Volunteers were randomized. Volunteers were followed.",
    )
    chunks = chunk_semantic([doc], topic_encoder, max_size=500, min_size=10, percentile=20)
    assert [c.text for c in chunks] == [
        "Statins help. Statin doses vary. Statin side effects exist.",
        "Volunteers were recruited. Volunteers were randomized. Volunteers were followed.",
    ]


def test_semantic_respects_max_size():
    doc = Document("d", " ".join(f"Statin sentence number {i} here." for i in range(20)))
    chunks = chunk_semantic([doc], topic_encoder, max_size=100, min_size=10)
    assert all(len(c.text) <= 100 for c in chunks)


def test_semantic_merges_tiny_chunks():
    doc = Document("d", "Statins help patients a lot. Okay. Volunteers were recruited from the clinic.")
    chunks = chunk_semantic([doc], topic_encoder, max_size=500, min_size=50, percentile=90)
    assert all(len(c.text) >= 5 for c in chunks)
    assert len(chunks) <= 2


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_produces_valid_unique_chunks(strategy):
    docs = [Document("a", TEXT * 3), Document("b", "Short abstract about statins.")]
    chunks = build_chunks(strategy, docs, size=150, encode=topic_encoder)
    assert {c.doc_id for c in chunks} == {"a", "b"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.text.strip() for c in chunks)


def test_sliding_window_overlaps_more_than_fixed():
    docs = [Document("a", "abcdefghij" * 30)]
    fixed = build_chunks("fixed", docs, size=100)
    sliding = build_chunks("sliding", docs, size=100)
    assert len(sliding) > len(fixed)


def test_unknown_strategy_and_missing_encoder_raise():
    with pytest.raises(ValueError):
        build_chunks("magic", [])
    with pytest.raises(ValueError):
        build_chunks("semantic", [Document("a", "x.")])
