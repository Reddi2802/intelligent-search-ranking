"""
MS MARCO data loader.
Loads passages, queries, and relevance judgments from HuggingFace.
Handles subset selection and caching to avoid re-downloading.
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

PASSAGES_FILE = RAW_DIR / "passages.json"
QUERIES_FILE = RAW_DIR / "queries.json"
QRELS_FILE = RAW_DIR / "qrels.json"


def load_msmarco(num_passages: int = 10_000) -> tuple[list[dict], list[dict], dict]:
    """
    Load MS MARCO passages, queries, and relevance judgments.

    Args:
        num_passages: Number of passages to load. Default 10,000.

    Returns:
        passages: list of {"id": str, "text": str}
        queries:  list of {"id": str, "text": str}
        qrels:    dict of {query_id: [passage_id, ...]}
    """
    if PASSAGES_FILE.exists() and QUERIES_FILE.exists() and QRELS_FILE.exists():
        logger.info("Loading cached MS MARCO data from data/raw/")
        passages = json.loads(PASSAGES_FILE.read_text())
        queries = json.loads(QUERIES_FILE.read_text())
        qrels = json.loads(QRELS_FILE.read_text())
        logger.info(
            f"Loaded {len(passages)} passages, {len(queries)} queries from cache."
        )
        return passages, queries, qrels

    logger.info("Downloading MS MARCO from HuggingFace...")
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train")

    passages = []
    queries = []
    qrels = {}
    seen_passage_ids = set()

    for row in dataset:
        query_id = str(row["query_id"])
        query_text = row["query"]
        passage_texts = row["passages"]["passage_text"]
        passage_ids = [
            f"{query_id}_{i}" for i in range(len(passage_texts))
        ]
        is_selected = row["passages"]["is_selected"]

        queries.append({"id": query_id, "text": query_text})
        relevant_ids = []

        for pid, ptext, selected in zip(passage_ids, passage_texts, is_selected):
            if pid not in seen_passage_ids:
                passages.append({"id": pid, "text": ptext})
                seen_passage_ids.add(pid)
                if len(passages) >= num_passages:
                    break
            if selected == 1:
                relevant_ids.append(pid)

        if relevant_ids:
            qrels[query_id] = relevant_ids

        if len(passages) >= num_passages:
            break

    logger.info(
        f"Loaded {len(passages)} passages, {len(queries)} queries, "
        f"{len(qrels)} queries with relevance labels."
    )

    logger.info("Caching to data/raw/...")
    PASSAGES_FILE.write_text(json.dumps(passages, indent=2))
    QUERIES_FILE.write_text(json.dumps(queries, indent=2))
    QRELS_FILE.write_text(json.dumps(qrels, indent=2))
    logger.info("Cached.")

    return passages, queries, qrels


if __name__ == "__main__":
    passages, queries, qrels = load_msmarco(num_passages=10_000)
    print(f"\nPassages : {len(passages)}")
    print(f"Queries  : {len(queries)}")
    print(f"Qrels    : {len(qrels)}")
    print(f"\nSample passage : {passages[0]}")
    print(f"Sample query   : {queries[0]}")