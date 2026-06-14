"""
Hybrid retriever combining BM25 and semantic retrieval.
Uses Reciprocal Rank Fusion (RRF) to merge ranked lists.
No score normalization needed — RRF works on ranks, not raw scores.
"""

import logging

from src.retrieval.bm25_retriever import build_bm25_index, retrieve_bm25
from src.retrieval.data_loader import load_msmarco
from src.retrieval.semantic_retriever import (
    build_faiss_index,
    load_embedding_model,
    retrieve_semantic,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RRF_K = 60  # RRF constant — controls influence of lower-ranked results


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    score_keys: list[str],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for a document = sum(1 / (k + rank)) across all lists.
    Documents appearing in multiple lists get boosted.
    Documents appearing in only one list are not penalized — just not boosted.

    Args:
        ranked_lists: list of ranked result lists, each item has "id", "text", score
        score_keys:   name of the score field in each list (for storing original scores)
        k:            RRF constant (default 60, from original RRF paper)

    Returns:
        list of merged results sorted by RRF score descending
    """
    rrf_scores: dict[str, float] = {}
    doc_store: dict[str, dict] = {}

    for ranked_list, score_key in zip(ranked_lists, score_keys):
        for rank, doc in enumerate(ranked_list):
            doc_id = doc["id"]
            rrf_score = 1.0 / (k + rank + 1)

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

            if doc_id not in doc_store:
                doc_store[doc_id] = {"id": doc_id, "text": doc["text"]}

            doc_store[doc_id][score_key] = doc.get(score_key, 0.0)

    merged = []
    for doc_id, rrf_score in sorted(
        rrf_scores.items(), key=lambda x: x[1], reverse=True
    ):
        doc = doc_store[doc_id].copy()
        doc["rrf_score"] = rrf_score
        merged.append(doc)

    for rank, doc in enumerate(merged):
        doc["rank"] = rank + 1

    return merged


def retrieve_hybrid(
    query: str,
    passages: list[dict],
    bm25_index,
    faiss_index,
    passage_map: list[dict],
    model,
    top_k: int = 100,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline for a single query.

    Args:
        query:       raw query string
        passages:    list of {"id": str, "text": str}
        bm25_index:  BM25Okapi index
        faiss_index: FAISS index
        passage_map: ordered passages matching FAISS index positions
        model:       SentenceTransformer model
        top_k:       final number of candidates to return

    Returns:
        list of merged results with rrf_score, bm25_score, semantic_score, rank
    """
    bm25_results = retrieve_bm25(query, passages, bm25_index, top_k=top_k)
    semantic_results = retrieve_semantic(
        query, passage_map, faiss_index, model, top_k=top_k
    )

    merged = reciprocal_rank_fusion(
        ranked_lists=[bm25_results, semantic_results],
        score_keys=["bm25_score", "semantic_score"],
        k=RRF_K,
    )

    return merged[:top_k]


if __name__ == "__main__":
    passages, queries, qrels = load_msmarco(num_passages=10_000)

    bm25_index = build_bm25_index(passages)
    model = load_embedding_model()
    faiss_index, passage_map = build_faiss_index(passages, model)

    for q in queries[:3]:
        results = retrieve_hybrid(
            q["text"], passages, bm25_index, faiss_index, passage_map, model, top_k=5
        )
        print(f"\nQuery: {q['text']}")
        for r in results:
            print(
                f"  [{r['rank']}] (rrf={r['rrf_score']:.4f} "
                f"bm25={r.get('bm25_score', 0):.4f} "
                f"sem={r.get('semantic_score', 0):.4f}) "
                f"{r['text'][:80]}..."
            )