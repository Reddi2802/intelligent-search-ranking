"""
Training pipeline for the LightGBM ranker.
Builds training data from MS MARCO and trains the ranker.
Uses a proper train/validation split to avoid overfitting.
Evaluates NDCG@10 on held-out validation queries only.
"""

import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ranking.ranker import load_ranker, rerank, train_ranker
from src.retrieval.bm25_retriever import build_bm25_index, retrieve_bm25
from src.retrieval.data_loader import load_msmarco
from src.retrieval.hybrid_retriever import retrieve_hybrid
from src.retrieval.semantic_retriever import (
    build_faiss_index,
    load_embedding_model,
    retrieve_semantic,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
TRAIN_RATIO = 0.8


def build_training_data(
    queries: list[dict],
    qrels: dict,
    passages: list[dict],
    bm25_index,
    faiss_index,
    passage_map: list[dict],
    model,
    top_k: int = 100,
) -> list[dict]:
    """
    Build training data by running retrieval on all labeled queries.
    Each item contains the query, hybrid candidates, and relevance labels.
    """
    training_data = []
    labeled_queries = [q for q in queries if q["id"] in qrels]
    logger.info(f"Building training data for {len(labeled_queries)} queries...")

    for i, q in enumerate(labeled_queries):
        if i % 100 == 0:
            logger.info(f"  Processing query {i}/{len(labeled_queries)}...")

        bm25_results = retrieve_bm25(q["text"], passages, bm25_index, top_k=top_k)
        semantic_results = retrieve_semantic(
            q["text"], passage_map, faiss_index, model, top_k=top_k
        )
        hybrid_results = retrieve_hybrid(
            q["text"], passages, bm25_index, faiss_index, passage_map, model, top_k=top_k
        )

        training_data.append({
            "query": q["text"],
            "query_id": q["id"],
            "candidates": hybrid_results,
            "bm25_results": bm25_results,
            "semantic_results": semantic_results,
            "relevant_ids": qrels[q["id"]],
        })

    return training_data


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
    positions_i = np.arange(1, len(ideal) + 1)
    idcg = float(np.sum(np.array(ideal) / np.log2(positions_i + 1)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(data: list[dict], ranker, label: str) -> float:
    """Evaluate ranker on a dataset split. Returns mean NDCG@10."""
    hybrid_scores = []
    ml_scores = []

    for item in data:
        relevant_ids = item["relevant_ids"]

        hybrid_ids = [c["id"] for c in item["candidates"]]
        hybrid_scores.append(ndcg_at_k(hybrid_ids, relevant_ids))

        ml_results = rerank(
            query=item["query"],
            candidates=item["candidates"],
            ranker=ranker,
            bm25_results=item["bm25_results"],
            semantic_results=item["semantic_results"],
        )
        ml_ids = [r["id"] for r in ml_results]
        ml_scores.append(ndcg_at_k(ml_ids, relevant_ids))

    hybrid_mean = float(np.mean(hybrid_scores))
    ml_mean = float(np.mean(ml_scores))

    print(f"\n--- {label} ---")
    print(f"{'Method':<35} {'NDCG@10':>10}")
    print("-" * 47)
    print(f"{'Hybrid (BM25 + Semantic)':<35} {hybrid_mean:>10.4f}")
    print(f"{'Hybrid + LightGBM Ranking':<35} {ml_mean:>10.4f}")
    print(f"Improvement: +{ml_mean - hybrid_mean:.4f}")

    return ml_mean


if __name__ == "__main__":
    import json

    TRAINING_CACHE = Path("data/raw/training_data.json")

    # Load data and indexes
    passages, queries, qrels = load_msmarco(num_passages=50_000)
    bm25_index, _ = build_bm25_index(passages)
    model = load_embedding_model()
    faiss_index, passage_map = build_faiss_index(passages, model)

    # Build or load training data
    if TRAINING_CACHE.exists():
        logger.info("Loading cached training data...")
        all_data = json.loads(TRAINING_CACHE.read_text())
        logger.info(f"Loaded {len(all_data)} training items from cache.")
    else:
        all_data = build_training_data(
            queries, qrels, passages, bm25_index, faiss_index, passage_map, model
        )
        TRAINING_CACHE.write_text(json.dumps(all_data))
        logger.info(f"Training data cached to {TRAINING_CACHE}")

    # Train/validation split
    random.seed(RANDOM_SEED)
    random.shuffle(all_data)

    split_idx = int(len(all_data) * TRAIN_RATIO)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    logger.info(f"Train: {len(train_data)} queries | Validation: {len(val_data)} queries")

    # Train ranker on training split only
    logger.info("Training LightGBM ranker...")
    ranker = train_ranker(train_data)

    # Evaluate on both splits
    train_ndcg = evaluate(train_data, ranker, "Training Set (seen during training)")
    val_ndcg = evaluate(val_data, ranker, "Validation Set (never seen during training)")

    print(f"\n{'='*47}")
    print(f"Final validation NDCG@10: {val_ndcg:.4f}")
    print(f"Train/val gap: {train_ndcg - val_ndcg:.4f} (lower is better)")