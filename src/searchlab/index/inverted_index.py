import logging
import math
import pickle
from collections import Counter

from searchlab.models import Chunk
from searchlab.tokenizer import tokenize

logger = logging.getLogger(__name__)


class InvertedIndex:
    """Maps each term to the chunks containing it (a posting list).

    postings[term] = {chunk_id: term_frequency}
    """

    def __init__(self) -> None:
        self.postings: dict[str, dict[str, int]] = {}
        self.chunk_lengths: dict[str, int] = {}
        self.chunk_to_doc: dict[str, str] = {}

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_lengths)

    @property
    def avg_chunk_length(self) -> float:
        if not self.chunk_lengths:
            return 0.0
        return sum(self.chunk_lengths.values()) / len(self.chunk_lengths)

    def add_chunk(self, chunk: Chunk) -> None:
        if chunk.chunk_id in self.chunk_lengths:
            raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
        tokens = tokenize(chunk.text)
        self.chunk_lengths[chunk.chunk_id] = len(tokens)
        self.chunk_to_doc[chunk.chunk_id] = chunk.doc_id
        for term, tf in Counter(tokens).items():
            self.postings.setdefault(term, {})[chunk.chunk_id] = tf

    def build(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.add_chunk(chunk)
        logger.info(
            "indexed %d chunks, %d unique terms", self.num_chunks, len(self.postings)
        )

    def doc_freq(self, term: str) -> int:
        """Number of chunks containing the term."""
        return len(self.postings.get(term, {}))

    def idf(self, term: str) -> float:
        """BM25 IDF with +1 inside the log so it is never negative."""
        n = self.num_chunks
        df = self.doc_freq(term)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def size_bytes(self) -> int:
        """Serialized size of the index, used for the benchmark."""
        return len(
            pickle.dumps((self.postings, self.chunk_lengths, self.chunk_to_doc))
        )
