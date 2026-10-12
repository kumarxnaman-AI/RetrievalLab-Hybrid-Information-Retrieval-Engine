import csv
import json
import logging
from pathlib import Path

from searchlab.models import Document

logger = logging.getLogger(__name__)


def load_beir_dataset(
    path: Path, split: str = "test"
) -> tuple[list[Document], dict[str, str], dict[str, dict[str, int]]]:
    """Load a BEIR-format dataset: corpus, queries, and relevance labels.

    Returns (documents, {query_id: query_text}, {query_id: {doc_id: grade}}).
    Only queries that have labels in the chosen split are returned.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {path}")

    documents = []
    with open(path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = f"{row.get('title', '')}. {row['text']}".strip(". ")
            documents.append(
                Document(doc_id=str(row["_id"]), text=text, metadata={"title": row.get("title", "")})
            )

    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / f"{split}.tsv", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for query_id, doc_id, score in reader:
            qrels.setdefault(query_id, {})[doc_id] = int(score)

    queries: dict[str, str] = {}
    with open(path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row["_id"]) in qrels:
                queries[str(row["_id"])] = row["text"]

    qrels = {qid: rels for qid, rels in qrels.items() if qid in queries}
    logger.info(
        "loaded %d documents, %d queries (%s split)", len(documents), len(queries), split
    )
    return documents, queries, qrels
