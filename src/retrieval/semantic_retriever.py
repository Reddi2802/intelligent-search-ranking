"""
Semantic retriever using sentence-transformers and FAISS.
Encodes passages and queries into dense vectors.
Retrieves top-k candidates via approximate nearest neighbor search.
Uses GPU if available (RTX 4050), falls back to CPU.
"""

import logging
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.data_loader import load_msmarco

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    PASSAGE_MAP_FILE,
)


def load_embedding_model() -> SentenceTransformer:
    """Load sentence-transformer model. Uses GPU if available."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")
    logger.info(f"Device: {model.device}")
    return model


def build_faiss_index(
    passages: list[dict],
    model: SentenceTransformer,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Encode all passages and build a FAISS index.
    Uses inner product (cosine similarity on normalized vectors).

    Args:
        passages: list of {"id": str, "text": str}
        model:    SentenceTransformer model

    Returns:
        index:       FAISS index
        passage_map: ordered list of passages matching index positions
    """
    if FAISS_INDEX_FILE.exists() and PASSAGE_MAP_FILE.exists():
        logger.info("Loading cached FAISS index...")
        index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(PASSAGE_MAP_FILE, "rb") as f:
            passage_map = pickle.load(f)
        logger.info(f"FAISS index loaded: {index.ntotal} vectors.")
        return index, passage_map

    logger.info(f"Encoding {len(passages)} passages in batches of {BATCH_SIZE}...")
    texts = [p["text"] for p in passages]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # normalize for cosine similarity
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(np.float32)

    # IndexFlatIP = exact inner product search (cosine sim on normalized vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    logger.info(f"FAISS index built: {index.ntotal} vectors.")

    # Save
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    with open(PASSAGE_MAP_FILE, "wb") as f:
        pickle.dump(passages, f)
    logger.info("FAISS index cached.")

    return index, passages


def retrieve_semantic(
    query: str,
    passage_map: list[dict],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    top_k: int = 100,
) -> list[dict]:
    """
    Retrieve top-k passages for a query using semantic similarity.

    Args:
        query:       raw query string
        passage_map: ordered list of passages matching index positions
        index:       FAISS index
        model:       SentenceTransformer model
        top_k:       number of candidates to return

    Returns:
        list of {"id": str, "text": str, "semantic_score": float, "rank": int}
        sorted by score descending
    """
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = [
        {
            "id": passage_map[idx]["id"],
            "text": passage_map[idx]["text"],
            "semantic_score": float(scores[0][rank]),
            "rank": rank + 1,
        }
        for rank, idx in enumerate(indices[0])
        if idx != -1  # FAISS returns -1 for empty slots
    ]

    return results


if __name__ == "__main__":
    passages, queries, qrels = load_msmarco(num_passages=10_000)
    model = load_embedding_model()
    index, passage_map = build_faiss_index(passages, model)

    # Test with first 3 queries
    for q in queries[:3]:
        results = retrieve_semantic(q["text"], passage_map, index, model, top_k=5)
        print(f"\nQuery: {q['text']}")
        for r in results:
            print(f"  [{r['rank']}] (score={r['semantic_score']:.4f}) {r['text'][:80]}...")