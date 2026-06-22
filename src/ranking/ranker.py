"""
LightGBM learning-to-rank model.
Uses LambdaMART to optimize directly for NDCG.
"""

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.ranking.feature_extractor import RankingFeatures, extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import RANKER_FILE


def train_ranker(
    training_data: list[dict],
) -> lgb.LGBMRanker:
    """
    Train a LightGBM LambdaMART ranker.

    Args:
        training_data: list of dicts, each with:
            - query: str
            - candidates: list of dicts (hybrid retrieval results)
            - relevant_ids: list of relevant passage IDs

    Returns:
        trained LGBMRanker
    """
    X = []       # feature matrix
    y = []       # relevance labels
    groups = []  # number of candidates per query (required by LightGBM ranker)

    for item in training_data:
        query = item["query"]
        candidates = item["candidates"]
        relevant_ids = set(item["relevant_ids"])

        # build rank lookup for BM25 and semantic
        bm25_rank_lookup = {
            c["id"]: c.get("rank", 101)
            for c in item.get("bm25_results", candidates)
        }
        semantic_rank_lookup = {
            c["id"]: c.get("rank", 101)
            for c in item.get("semantic_results", candidates)
        }

        group_size = 0
        for doc in candidates:
            features = extract_features(
                query=query,
                doc=doc,
                bm25_rank=bm25_rank_lookup.get(doc["id"], 101),
                semantic_rank=semantic_rank_lookup.get(doc["id"], 101),
            )
            X.append(features.to_list())
            y.append(1 if doc["id"] in relevant_ids else 0)
            group_size += 1

        groups.append(group_size)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    logger.info(
        f"Training on {len(groups)} queries, "
        f"{len(X)} (query, doc) pairs, "
        f"{sum(y)} positive labels."
    )

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        n_estimators=100,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )

    ranker.fit(
        X, y,
        group=groups,
        eval_set=[(X, y)],
        eval_group=[groups],
    )

    with open(RANKER_FILE, "wb") as f:
        pickle.dump(ranker, f)
    logger.info(f"Ranker saved to {RANKER_FILE}")

    return ranker


def load_ranker() -> lgb.LGBMRanker:
    """Load trained ranker from disk."""
    if not RANKER_FILE.exists():
        raise FileNotFoundError(
            f"No trained ranker found at {RANKER_FILE}. Run train.py first."
        )
    with open(RANKER_FILE, "rb") as f:
        return pickle.load(f)


def rerank(
    query: str,
    candidates: list[dict],
    ranker: lgb.LGBMRanker,
    bm25_results: list[dict] = None,
    semantic_results: list[dict] = None,
) -> list[dict]:
    """
    Rerank candidates using the trained LightGBM ranker.

    Args:
        query:            raw query string
        candidates:       hybrid retrieval results
        ranker:           trained LGBMRanker
        bm25_results:     original BM25 results for rank features
        semantic_results: original semantic results for rank features

    Returns:
        candidates reordered by ML ranking score
    """
    bm25_rank_lookup = {
        c["id"]: c.get("rank", 101)
        for c in (bm25_results or candidates)
    }
    semantic_rank_lookup = {
        c["id"]: c.get("rank", 101)
        for c in (semantic_results or candidates)
    }

    X = []
    for doc in candidates:
        features = extract_features(
            query=query,
            doc=doc,
            bm25_rank=bm25_rank_lookup.get(doc["id"], 101),
            semantic_rank=semantic_rank_lookup.get(doc["id"], 101),
        )
        X.append(features.to_list())

    X = np.array(X, dtype=np.float32)
    scores = ranker.predict(X)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for rank, (doc, score) in enumerate(ranked):
        doc = doc.copy()
        doc["ml_score"] = float(score)
        doc["rank"] = rank + 1
        results.append(doc)

    return results