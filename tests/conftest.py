import hashlib

import numpy as np
import pytest

from searchlab.config import Config
from searchlab.engine import SearchEngine
from searchlab.models import Document
from searchlab.retrieval.reranker import CrossEncoderReranker
from searchlab.tokenizer import tokenize

DIM = 64


def hashing_encoder(texts):
    """A tiny deterministic bag-of-words 'embedding', so tests never load a real model."""
    vectors = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in tokenize(text):
            vectors[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
    return vectors


def overlap_scorer(pairs):
    return [sum(t in text.lower() for t in query.lower().split()) for query, text in pairs]


DOCS = [
    Document("hr_pension", "Employees become eligible for the company pension plan after one year of service.", {"title": "Pension plan"}),
    Document("hr_leave", "Full-time employees receive twenty days of paid annual leave.", {"title": "Leave policy"}),
    Document("it_security", "Passwords must be rotated every ninety days and stored in the vault.", {"title": "Password policy"}),
    Document("fin_expense", "Travel expenses are reimbursed within thirty days of submitting receipts.", {"title": "Expense claims"}),
    Document("hr_retire", "Retirement benefits include a pension paid monthly after service ends.", {"title": "Retirement benefits"}),
    Document("it_vpn", "Remote staff connect to the corporate network through the approved VPN client.", {"title": "VPN access"}),
]


@pytest.fixture(scope="session")
def engine():
    return SearchEngine.from_documents(
        DOCS,
        Config(chunk_size=200, chunk_overlap=20),
        cache_dir=None,
        encoder=hashing_encoder,
        reranker=CrossEncoderReranker(scorer=overlap_scorer),
        M=4,
        ef_construction=16,
    )
