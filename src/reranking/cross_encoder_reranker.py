"""
Neural reranker using a cross-encoder model.
Applied to top-K candidates from ML ranking.
Cross-encoders jointly encode query and document for fine-grained relevance scoring.
Much more accurate than bi-encoders but too slow for full corpus retrieval.
"""

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

from src.ranking.ranker import load_ranker, rerank
from src.retrieval.bm25_retriever import build_bm25_index
from src.retrieval.data_loader import load_msmarco
from src.retrieval.semantic_retriever import (
    build_faiss_index,
    load_embedding_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import (  # only rerank top 20 from ML ranking
    CROSS_ENCODER_MODEL,
    TOP_K_RERANK,
)


def load_cross_encoder() -> CrossEncoder:
    """Load cross-encoder model. Downloads on first use, cached after."""
    logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
    model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
    logger.info("Cross-encoder loaded.")
    return model


def rerank_with_cross_encoder(
    query: str,
    candidates: list[dict],
    cross_encoder: CrossEncoder,
    top_k: int = TOP_K_RERANK,
) -> list[dict]:
    """
    Rerank top-K candidates using cross-encoder.

    Args:
        query:         raw query string
        candidates:    ML-ranked candidates (we take top_k of these)
        cross_encoder: CrossEncoder model
        top_k:         how many candidates to rerank

    Returns:
        reranked list with cross_encoder_score added, sorted by score descending
    """
    # only rerank the top candidates — cross-encoder is too slow for all 100
    top_candidates = candidates[:top_k]
    remaining = candidates[top_k:]

    # cross-encoder expects list of [query, document] pairs
    pairs = [[query, doc["text"]] for doc in top_candidates]
    scores = cross_encoder.predict(pairs, batch_size=20, show_progress_bar=False)

    # attach scores and sort
    for doc, score in zip(top_candidates, scores):
        doc["cross_encoder_score"] = float(score)

    reranked = sorted(top_candidates, key=lambda x: x["cross_encoder_score"], reverse=True)

    # append remaining candidates below the reranked ones
    for rank, doc in enumerate(reranked + remaining):
        doc["rank"] = rank + 1

    return reranked + remaining


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    """Compute NDCG@K."""
    relevant_set = set(relevant_ids)
    relevances = [1 if pid in relevant_set else 0 for pid in retrieved_ids[:k]]
    if not relevances:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    discounts = np.log2(positions + 1)
    dcg = float(np.sum(np.array(relevances) / discounts))
    ideal = [1] * min(len(relevant_ids), k)
    idcg = float(np.sum(np.array(ideal) / np.log2(np.arange(1, len(ideal) + 1) + 1)))
    return dcg / idcg if idcg > 0 else 0.0


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Load everything
    passages, queries, qrels = load_msmarco(num_passages=50_000)
    bm25_index, _ = build_bm25_index(passages)
    model = load_embedding_model()
    faiss_index, passage_map = build_faiss_index(passages, model)
    ranker = load_ranker()
    cross_encoder = load_cross_encoder()

    # Load cached training data for validation split
    TRAINING_CACHE = Path("data/raw/training_data.json")
    all_data = json.loads(TRAINING_CACHE.read_text())

    import random
    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    val_data = all_data[split_idx:]

    logger.info(f"Evaluating cross-encoder on {len(val_data)} validation queries...")

    ml_scores = []
    ce_scores = []

    for i, item in enumerate(val_data):
        if i % 20 == 0:
            logger.info(f"  Query {i}/{len(val_data)}...")

        relevant_ids = item["relevant_ids"]

        # ML ranked results
        ml_results = rerank(
            query=item["query"],
            candidates=item["candidates"],
            ranker=ranker,
            bm25_results=item["bm25_results"],
            semantic_results=item["semantic_results"],
        )
        ml_ids = [r["id"] for r in ml_results]
        ml_scores.append(ndcg_at_k(ml_ids, relevant_ids))

        # Cross-encoder reranked
        ce_results = rerank_with_cross_encoder(
            query=item["query"],
            candidates=ml_results,
            cross_encoder=cross_encoder,
            top_k=TOP_K_RERANK,
        )
        ce_ids = [r["id"] for r in ce_results]
        ce_scores.append(ndcg_at_k(ce_ids, relevant_ids))

    print("\n--- Neural Reranking Results ---\n")
    print(f"{'Method':<40} {'NDCG@10':>10}")
    print("-" * 52)
    print(f"{'Hybrid + LightGBM':<40} {np.mean(ml_scores):>10.4f}")
    print(f"{'Hybrid + LightGBM + Cross-Encoder':<40} {np.mean(ce_scores):>10.4f}")
    print(f"\nImprovement: +{np.mean(ce_scores) - np.mean(ml_scores):.4f}")