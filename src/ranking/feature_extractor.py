"""
Feature extractor for learning-to-rank.
Extracts a feature vector for every (query, document) pair.
These features are the inputs to the LightGBM ranker.
"""

import re
from dataclasses import dataclass


@dataclass
class RankingFeatures:
    """Feature vector for a single (query, document) pair."""
    bm25_score: float
    semantic_score: float
    rrf_score: float
    bm25_rank: int
    semantic_rank: int
    doc_length: int
    query_length: int
    query_term_coverage: float
    title_overlap: float

    def to_list(self) -> list[float]:
        return [
            self.bm25_score,
            self.semantic_score,
            self.rrf_score,
            float(self.bm25_rank),
            float(self.semantic_rank),
            float(self.doc_length),
            float(self.query_length),
            self.query_term_coverage,
            self.title_overlap,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "bm25_score",
            "semantic_score",
            "rrf_score",
            "bm25_rank",
            "semantic_rank",
            "doc_length",
            "query_length",
            "query_term_coverage",
            "title_overlap",
        ]


def tokenize(text: str) -> set[str]:
    """Lowercase and split into word tokens."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def extract_features(
    query: str,
    doc: dict,
    bm25_rank: int = 101,
    semantic_rank: int = 101,
) -> RankingFeatures:
    """
    Extract features for a single (query, document) pair.

    Args:
        query:         raw query string
        doc:           dict with keys: text, bm25_score, semantic_score, rrf_score
        bm25_rank:     rank of this doc in BM25 results (default 101 = not retrieved)
        semantic_rank: rank of this doc in semantic results (default 101 = not retrieved)

    Returns:
        RankingFeatures dataclass
    """
    query_tokens = tokenize(query)
    doc_tokens = tokenize(doc["text"])
    doc_words = doc["text"].lower().split()

    # fraction of query words that appear in the document
    if query_tokens:
        query_term_coverage = len(query_tokens & doc_tokens) / len(query_tokens)
    else:
        query_term_coverage = 0.0

    # fraction of query words that appear in first 10 words of document
    title_words = set(doc_words[:10])
    if query_tokens:
        title_overlap = len(query_tokens & title_words) / len(query_tokens)
    else:
        title_overlap = 0.0

    return RankingFeatures(
        bm25_score=float(doc.get("bm25_score", 0.0)),
        semantic_score=float(doc.get("semantic_score", 0.0)),
        rrf_score=float(doc.get("rrf_score", 0.0)),
        bm25_rank=bm25_rank,
        semantic_rank=semantic_rank,
        doc_length=len(doc_words),
        query_length=len(query.split()),
        query_term_coverage=query_term_coverage,
        title_overlap=title_overlap,
    )