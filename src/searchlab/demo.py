import argparse
from pathlib import Path

from searchlab.config import Config
from searchlab.evaluation.dataset import load_beir_dataset
from searchlab.logging_setup import setup_logging
from searchlab.pipeline import build_systems
from searchlab.retrieval.aggregate import aggregate_to_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BM25 and dense results")
    parser.add_argument("query")
    parser.add_argument("--dataset", default="datasets/scifact")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    config = Config(log_level="WARNING")
    setup_logging(config.log_level)
    docs, _, _ = load_beir_dataset(Path(args.dataset))
    systems = build_systems(docs, config)
    titles = {d.doc_id: d.metadata.get("title", "") for d in docs}

    print(f'\nQuery: "{args.query}"')
    for name, retriever, label in [
        ("BM25", systems.bm25, "score"),
        ("Dense", systems.dense, "similarity"),
    ]:
        print(f"\n{name}")
        results = aggregate_to_documents(retriever.search(args.query, args.k * 5), args.k)
        for rank, r in enumerate(results, start=1):
            print(f"{rank}. doc_{r.doc_id:<10} {label}: {r.score:.2f}  {titles[r.doc_id][:60]}")


if __name__ == "__main__":
    main()
