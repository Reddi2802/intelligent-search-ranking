"""
Pydantic models for API request/response validation.
These define the contracts between the client and the API.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SearchMode(str, Enum):
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    FULL = "full"  # hybrid + LightGBM + cross-encoder


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    mode: SearchMode = Field(default=SearchMode.FULL)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class ScoreExplanation(BaseModel):
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    rrf_score: float = 0.0
    ml_score: float = 0.0
    cross_encoder_score: float = 0.0


class SearchResult(BaseModel):
    passage_id: str
    text: str
    rank: int
    scores: ScoreExplanation


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    results: list[SearchResult]
    total_results: int
    latency_ms: float


class RankRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    candidates: list[dict] = Field(..., min_length=1, max_length=1000)
    method: str = Field(default="full")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class AnalyticsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    p95_latency_ms: float
    top_queries: list[str]
    mode_distribution: dict[str, int]


class ModelStatus(str, Enum):
    LOADED = "loaded"
    NOT_LOADED = "not_loaded"


class HealthResponse(BaseModel):
    status: str
    bm25_index: ModelStatus
    faiss_index: ModelStatus
    embedding_model: ModelStatus
    ml_ranker: ModelStatus
    cross_encoder: ModelStatus