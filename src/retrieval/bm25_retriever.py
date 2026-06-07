"""
BM25 retriever using rank_bm25.
Builds an index over the passage corpus and retrieves top-k candidates for a query.
Index is cached to disk to avoid rebuilding on every run.
"""

import json
import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.retrieval.data_loader import load_msmarco

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_DIR = Path("data/indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
BM25_INDEX_FILE = INDEX_DIR / "bm25.pkl"


def tokenize(text: str) -> list[str]:
    """Lowercase and whitespace tokenize."""
    return text.lower().split()


def build_bm25_index(passages: list[dict]) -> BM25Okapi:
    """
    Build and cache a BM25 index over the passage corpus.

    Args:
        passages: list of {"id": str, "text": str}

    Returns:
        BM25Okapi index
    """
    if BM25_INDEX_FILE.exists():
        logger.info("Loading cached BM25 index...")
        with open(BM25_INDEX_FILE, "rb") as f:
            index = pickle.load(f)
        logger.info("BM25 index loaded.")
        return index

    logger.info(f"Building BM25 index over {len(passages)} passages...")
    tokenized = [tokenize(p["text"]) for p in passages]
    index = BM25Okapi(tokenized)

    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(index, f)
    logger.info(f"BM25 index built and cached to {BM25_INDEX_FILE}")

    return index


def retrieve_bm25(
    query: str,
    passages: list[dict],
    index: BM25Okapi,
    top_k: int = 100,
) -> list[dict]:
    """
    Retrieve top-k passages for a query using BM25.

    Args:
        query:    raw query string
        passages: list of {"id": str, "text": str}
        index:    BM25Okapi index
        top_k:    number of candidates to return

    Returns:
        list of {"id": str, "text": str, "bm25_score": float, "rank": int}
        sorted by score descending
    """
    tokens = tokenize(query)
    scores = index.get_scores(tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = [
        {
            "id": passages[i]["id"],
            "text": passages[i]["text"],
            "bm25_score": float(scores[i]),
            "rank": rank + 1,
        }
        for rank, i in enumerate(top_indices)
    ]

    return results


if __name__ == "__main__":
    passages, queries, qrels = load_msmarco(num_passages=10_000)
    index = build_bm25_index(passages)

    # Test with first 3 queries
    for q in queries[:3]:
        results = retrieve_bm25(q["text"], passages, index, top_k=5)
        print(f"\nQuery: {q['text']}")
        for r in results:
            print(f"  [{r['rank']}] (score={r['bm25_score']:.4f}) {r['text'][:80]}...")