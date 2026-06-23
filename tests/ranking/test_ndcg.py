"""
Unit tests for NDCG evaluation metric.
Tests against known values so we can trust our evaluation numbers.
"""

import pytest
import numpy as np
from evaluation.ndcg import ndcg_at_k, dcg_at_k


# ─── DCG ──────────────────────────────────────────────────────────────────────

def test_dcg_perfect_single_relevant():
    """Relevant doc at rank 1 — maximum DCG."""
    relevances = [1, 0, 0]
    dcg = dcg_at_k(relevances, k=3)
    assert dcg == pytest.approx(1.0, abs=1e-6)


def test_dcg_relevant_at_rank_2():
    """Relevant doc at rank 2 — discounted."""
    relevances = [0, 1, 0]
    dcg = dcg_at_k(relevances, k=3)
    expected = 1.0 / np.log2(3)
    assert dcg == pytest.approx(expected, abs=1e-6)


def test_dcg_empty():
    assert dcg_at_k([], k=10) == 0.0


def test_dcg_all_zeros():
    assert dcg_at_k([0, 0, 0], k=3) == 0.0


# ─── NDCG ─────────────────────────────────────────────────────────────────────

def test_ndcg_perfect_ranking():
    """Relevant doc ranked first — NDCG should be 1.0."""
    retrieved = ["doc_1", "doc_2", "doc_3"]
    relevant = ["doc_1"]
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_worst_ranking():
    """Relevant doc ranked last — NDCG should be less than 1.0."""
    retrieved = ["doc_2", "doc_3", "doc_1"]
    relevant = ["doc_1"]
    score = ndcg_at_k(retrieved, relevant, k=3)
    assert score < 1.0
    assert score > 0.0


def test_ndcg_relevant_not_in_results():
    """No relevant docs in top-k — NDCG should be 0.0."""
    retrieved = ["doc_2", "doc_3"]
    relevant = ["doc_1"]
    assert ndcg_at_k(retrieved, relevant, k=2) == 0.0


def test_ndcg_empty_retrieved():
    assert ndcg_at_k([], ["doc_1"], k=10) == 0.0


def test_ndcg_between_zero_and_one():
    retrieved = ["doc_3", "doc_1", "doc_2"]
    relevant = ["doc_1"]
    score = ndcg_at_k(retrieved, relevant, k=3)
    assert 0.0 <= score <= 1.0


def test_ndcg_higher_rank_beats_lower_rank():
    """Same relevant doc, different rank — higher rank should score higher."""
    retrieved_good = ["doc_1", "doc_2", "doc_3"]
    retrieved_bad = ["doc_2", "doc_3", "doc_1"]
    relevant = ["doc_1"]

    score_good = ndcg_at_k(retrieved_good, relevant, k=3)
    score_bad = ndcg_at_k(retrieved_bad, relevant, k=3)
    assert score_good > score_bad