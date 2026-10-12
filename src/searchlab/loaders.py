import logging
from pathlib import Path

from searchlab.models import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(data_dir: Path) -> list[Document]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    documents = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8")
        if not text.strip():
            logger.warning("skipping empty file: %s", path)
            continue

        doc_id = path.relative_to(data_dir).with_suffix("").as_posix()
        documents.append(
            Document(
                doc_id=doc_id,
                text=text,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    logger.info("loaded %d documents from %s", len(documents), data_dir)
    return documents
