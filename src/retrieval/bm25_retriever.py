"""
BM25 retriever using bm25s — vectorized NumPy implementation.
50-100x faster than rank_bm25 for large corpora.
"""

import logging
import pickle
from pathlib import Path

import bm25s
import numpy as np

from src.retrieval.data_loader import load_msmarco
from src.config import BM25_INDEX_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BM25S_DIR = BM25_INDEX_FILE.parent / "bm25s_index"


def build_bm25_index(passages: list[dict]) -> tuple[bm25s.BM25, list[dict]]:
    """
    Build and cache a BM25 index over the passage corpus.
    Returns (index, passages) — passages order matches index positions.
    """
    if BM25S_DIR.exists():
        logger.info("Loading cached BM25 index...")
        retriever = bm25s.BM25.load(str(BM25S_DIR), load_corpus=False)
        logger.info("BM25 index loaded.")
        return retriever, passages

    logger.info(f"Building BM25 index over {len(passages)} passages...")
    corpus = [p["text"] for p in passages]
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    BM25S_DIR.mkdir(parents=True, exist_ok=True)
    retriever.save(str(BM25S_DIR))
    logger.info("BM25 index built and cached.")

    return retriever, passages


def retrieve_bm25(
    query: str,
    passages: list[dict],
    index: bm25s.BM25,
    top_k: int = 100,
) -> list[dict]:
    """Retrieve top-k passages using BM25."""
    query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
    result_indices, scores = index.retrieve(
        query_tokens, k=min(top_k, len(passages)), return_as="tuple"
    )

    output = []
    for rank, (idx, score) in enumerate(zip(result_indices[0], scores[0])):
        output.append({
            "id": passages[idx]["id"],
            "text": passages[idx]["text"],
            "bm25_score": float(score),
            "rank": rank + 1,
        })

    return output