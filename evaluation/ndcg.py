"""
NDCG@10 evaluation script.
Benchmarks BM25, semantic, and hybrid retrieval against MS MARCO relevance labels.
Results are written to docs/Evaluation.md.
"""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.bm25_retriever import build_bm25_index, retrieve_bm25
from src.retrieval.data_loader import load_msmarco
from src.retrieval.hybrid_retriever import retrieve_hybrid
from src.retrieval.semantic_retriever import (
    build_faiss_index,
    load_embedding_model,
    retrieve_semantic,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

K = 10


def dcg_at_k(relevances: list[int], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at K.

    DCG@K = sum(relevance_i / log2(i + 2)) for i in 0..K-1

    The log2(i+2) term discounts lower-ranked results.
    Position 1 gets no discount (log2(2)=1).
    Position 2 gets log2(3)~1.58 discount. And so on.
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    discounts = np.log2(positions + 1)
    return float(np.sum(np.array(relevances) / discounts))


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Compute Normalized DCG at K.

    NDCG@K = DCG@K / IDCG@K

    IDCG = ideal DCG — what you'd get if all relevant docs were ranked first.
    Normalizing by IDCG puts the score between 0 and 1.

    Args:
        retrieved_ids: ordered list of retrieved passage IDs (ranked 1..N)
        relevant_ids:  list of passage IDs that are actually relevant
        k:             cutoff

    Returns:
        NDCG@K score between 0.0 and 1.0
    """
    relevant_set = set(relevant_ids)
    relevances = [1 if pid in relevant_set else 0 for pid in retrieved_ids[:k]]

    dcg = dcg_at_k(relevances, k)

    # Ideal: all relevant docs ranked first
    ideal_relevances = [1] * min(len(relevant_ids), k)
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_retriever(
    retriever_name: str,
    retrieve_fn,
    queries: list[dict],
    qrels: dict,
    k: int = K,
) -> float:
    """
    Evaluate a retriever function over all queries that have relevance labels.

    Args:
        retriever_name: display name for logging
        retrieve_fn:    callable(query_text) -> list of {"id": str, ...}
        queries:        list of {"id": str, "text": str}
        qrels:          dict of {query_id: [relevant_passage_ids]}
        k:              NDCG cutoff

    Returns:
        mean NDCG@K across all evaluated queries
    """
    scores = []
    evaluated = 0

    for q in queries:
        qid = q["id"]
        if qid not in qrels:
            continue

        results = retrieve_fn(q["text"])
        retrieved_ids = [r["id"] for r in results]
        relevant_ids = qrels[qid]

        score = ndcg_at_k(retrieved_ids, relevant_ids, k)
        scores.append(score)
        evaluated += 1

    mean_ndcg = float(np.mean(scores)) if scores else 0.0
    print(f"{retriever_name:30s} NDCG@{k} = {mean_ndcg:.4f}  ({evaluated} queries)")
    return mean_ndcg


if __name__ == "__main__":
    print("Loading data...")
    passages, queries, qrels = load_msmarco(num_passages=10_000)
    print(f"Evaluating on {len(qrels)} queries with relevance labels.\n")

    # BM25
    print("Building BM25 index...")
    bm25_index = build_bm25_index(passages)

    # Semantic
    print("Loading embedding model and FAISS index...")
    model = load_embedding_model()
    faiss_index, passage_map = build_faiss_index(passages, model)

    print("\n--- Evaluation Results ---\n")

    bm25_ndcg = evaluate_retriever(
        retriever_name="BM25",
        retrieve_fn=lambda q: retrieve_bm25(q, passages, bm25_index, top_k=K),
        queries=queries,
        qrels=qrels,
        k=K,
    )

    semantic_ndcg = evaluate_retriever(
        retriever_name="Semantic (FAISS)",
        retrieve_fn=lambda q: retrieve_semantic(
            q, passage_map, faiss_index, model, top_k=K
        ),
        queries=queries,
        qrels=qrels,
        k=K,
    )

    hybrid_ndcg = evaluate_retriever(
        retriever_name="Hybrid (BM25 + Semantic)",
        retrieve_fn=lambda q: retrieve_hybrid(
            q, passages, bm25_index, faiss_index, passage_map, model, top_k=K
        ),
        queries=queries,
        qrels=qrels,
        k=K,
    )

    print("\n--- Summary ---\n")
    print(f"{'Method':<30} {'NDCG@10':>10}")
    print(f"{'-'*42}")
    print(f"{'BM25':<30} {bm25_ndcg:>10.4f}")
    print(f"{'Semantic (FAISS)':<30} {semantic_ndcg:>10.4f}")
    print(f"{'Hybrid (BM25 + Semantic)':<30} {hybrid_ndcg:>10.4f}")