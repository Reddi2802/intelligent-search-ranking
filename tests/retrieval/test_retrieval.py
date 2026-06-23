"""
Unit tests for retrieval components.
No server required — tests core logic in isolation.
"""

import pytest
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion


# ─── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def test_rrf_higher_score_for_doc_in_both_lists():
    list1 = [{"id": "a", "text": "doc a", "bm25_score": 10.0}]
    list2 = [{"id": "a", "text": "doc a", "semantic_score": 0.9},
             {"id": "b", "text": "doc b", "semantic_score": 0.8}]

    results = reciprocal_rank_fusion(
        ranked_lists=[list1, list2],
        score_keys=["bm25_score", "semantic_score"],
    )

    ids = [r["id"] for r in results]
    assert ids[0] == "a"  # doc in both lists should rank first


def test_rrf_returns_all_unique_docs():
    list1 = [{"id": "a", "text": "a", "bm25_score": 1.0},
             {"id": "b", "text": "b", "bm25_score": 0.5}]
    list2 = [{"id": "b", "text": "b", "semantic_score": 0.9},
             {"id": "c", "text": "c", "semantic_score": 0.8}]

    results = reciprocal_rank_fusion(
        ranked_lists=[list1, list2],
        score_keys=["bm25_score", "semantic_score"],
    )

    result_ids = {r["id"] for r in results}
    assert result_ids == {"a", "b", "c"}


def test_rrf_scores_are_positive():
    list1 = [{"id": "a", "text": "a", "bm25_score": 5.0}]
    list2 = [{"id": "a", "text": "a", "semantic_score": 0.7}]

    results = reciprocal_rank_fusion(
        ranked_lists=[list1, list2],
        score_keys=["bm25_score", "semantic_score"],
    )

    assert all(r["rrf_score"] > 0 for r in results)


def test_rrf_ranks_are_sequential():
    list1 = [{"id": str(i), "text": f"doc {i}", "bm25_score": float(i)}
             for i in range(5)]
    list2 = [{"id": str(i), "text": f"doc {i}", "semantic_score": float(i)}
             for i in range(5)]

    results = reciprocal_rank_fusion(
        ranked_lists=[list1, list2],
        score_keys=["bm25_score", "semantic_score"],
    )

    ranks = [r["rank"] for r in results]
    assert ranks == list(range(1, len(results) + 1))


def test_rrf_empty_list_handled():
    list1 = [{"id": "a", "text": "a", "bm25_score": 1.0}]
    list2 = []

    results = reciprocal_rank_fusion(
        ranked_lists=[list1, list2],
        score_keys=["bm25_score", "semantic_score"],
    )

    assert len(results) == 1
    assert results[0]["id"] == "a"