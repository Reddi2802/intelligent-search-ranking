"""
Unit tests for feature extraction.
"""

import pytest
from src.ranking.feature_extractor import extract_features, RankingFeatures, tokenize


# ─── Tokenizer ────────────────────────────────────────────────────────────────

def test_tokenize_lowercases():
    result = tokenize("Hello World")
    assert "hello" in result
    assert "world" in result


def test_tokenize_removes_punctuation():
    result = tokenize("hello, world!")
    assert "hello" in result
    assert "world" in result


def test_tokenize_empty_string():
    result = tokenize("")
    assert result == set()


def test_tokenize_returns_set():
    result = tokenize("the the the")
    assert isinstance(result, set)
    assert len(result) == 1


# ─── Feature extraction ───────────────────────────────────────────────────────

def make_doc(**kwargs):
    defaults = {
        "id": "test_1",
        "text": "Stalin wanted to control Eastern Europe for security reasons",
        "bm25_score": 10.0,
        "semantic_score": 0.8,
        "rrf_score": 0.03,
    }
    defaults.update(kwargs)
    return defaults


def test_extract_features_returns_correct_count():
    features = extract_features(
        query="why did stalin want eastern europe",
        doc=make_doc(),
    )
    assert len(features.to_list()) == len(RankingFeatures.feature_names())


def test_query_term_coverage_perfect_overlap():
    features = extract_features(
        query="stalin europe",
        doc=make_doc(text="stalin europe control"),
    )
    assert features.query_term_coverage == 1.0


def test_query_term_coverage_no_overlap():
    features = extract_features(
        query="quantum physics",
        doc=make_doc(text="stalin europe control"),
    )
    assert features.query_term_coverage == 0.0


def test_query_term_coverage_partial_overlap():
    features = extract_features(
        query="stalin quantum",
        doc=make_doc(text="stalin europe control"),
    )
    assert features.query_term_coverage == 0.5


def test_doc_length_correct():
    features = extract_features(
        query="test query",
        doc=make_doc(text="one two three four five"),
    )
    assert features.doc_length == 5


def test_query_length_correct():
    features = extract_features(
        query="one two three",
        doc=make_doc(),
    )
    assert features.query_length == 3


def test_default_ranks_are_101():
    features = extract_features(
        query="test",
        doc=make_doc(),
    )
    assert features.bm25_rank == 101
    assert features.semantic_rank == 101


def test_title_overlap_first_10_words():
    features = extract_features(
        query="stalin europe",
        doc=make_doc(text="stalin europe control security " + "filler " * 20),
    )
    assert features.title_overlap > 0.0


def test_feature_names_match_vector_length():
    assert len(RankingFeatures.feature_names()) == 9