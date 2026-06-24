"""
API route handlers.
All models are accessed from app.state — loaded once at startup.
Thread safety via locks for non-thread-safe models.
"""

import logging
import threading
import time
from collections import deque

from fastapi import APIRouter, HTTPException, Request

from src.api.models import (
    AnalyticsResponse,
    HealthResponse,
    ModelStatus,
    RankRequest,
    ScoreExplanation,
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.ranking.ranker import rerank
from src.reranking.cross_encoder_reranker import rerank_with_cross_encoder
from src.retrieval.bm25_retriever import retrieve_bm25
from src.retrieval.hybrid_retriever import fuse_results, retrieve_hybrid
from src.retrieval.semantic_retriever import retrieve_semantic

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory analytics store
_query_log: deque = deque(maxlen=1000)
_latency_log: deque = deque(maxlen=1000)
_mode_counts: dict[str, int] = {"bm25": 0, "semantic": 0, "hybrid": 0, "full": 0}

# Thread lock for cross-encoder (not thread-safe)
_ce_lock = threading.Lock()
_embed_lock = threading.Lock()


def _build_result(doc: dict, rank: int) -> SearchResult:
    return SearchResult(
        passage_id=doc["id"],
        text=doc["text"],
        rank=rank,
        scores=ScoreExplanation(
            bm25_score=doc.get("bm25_score", 0.0),
            semantic_score=doc.get("semantic_score", 0.0),
            rrf_score=doc.get("rrf_score", 0.0),
            ml_score=doc.get("ml_score", 0.0),
            cross_encoder_score=doc.get("cross_encoder_score", 0.0),
        ),
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, body: SearchRequest):
    state = request.app.state
    start = time.time()

    query = body.query
    top_k = body.top_k
    mode = body.mode

    try:
        if mode == SearchMode.BM25:
            results = retrieve_bm25(
                query, state.passages, state.bm25_index, top_k=top_k
            )

        elif mode == SearchMode.SEMANTIC:
            with _embed_lock:
                results = retrieve_semantic(
                    query, state.passage_map, state.faiss_index,
                    state.embedding_model, top_k=top_k
                )

        elif mode == SearchMode.HYBRID:
            with _embed_lock:
                results = retrieve_hybrid(
                    query, state.passages, state.bm25_index,
                    state.faiss_index, state.passage_map,
                    state.embedding_model, top_k=top_k
                )

        elif mode == SearchMode.FULL:
            bm25_results = retrieve_bm25(
                query, state.passages, state.bm25_index, top_k=100
            )

            with _embed_lock:
                semantic_results = retrieve_semantic(
                    query, state.passage_map, state.faiss_index,
                    state.embedding_model, top_k=100
                )

            hybrid_results = fuse_results(bm25_results, semantic_results, top_k=100)

            ml_results = rerank(
                query=query,
                candidates=hybrid_results,
                ranker=state.ranker,
                bm25_results=bm25_results,
                semantic_results=semantic_results,
            )

            with _ce_lock:
                results = rerank_with_cross_encoder(
                    query=query,
                    candidates=ml_results,
                    cross_encoder=state.cross_encoder,
                    top_k=20,
                )

        latency_ms = (time.time() - start) * 1000

        # log analytics
        _query_log.append(query)
        _latency_log.append(latency_ms)
        _mode_counts[mode.value] += 1

        return SearchResponse(
            query=query,
            mode=mode,
            results=[_build_result(r, i + 1) for i, r in enumerate(results[:top_k])],
            total_results=len(results),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rank", response_model=SearchResponse)
async def rank(request: Request, body: RankRequest):
    state = request.app.state
    start = time.time()

    try:
        ml_results = rerank(
            query=body.query,
            candidates=body.candidates,
            ranker=state.ranker,
        )

        with _ce_lock:
            results = rerank_with_cross_encoder(
                query=body.query,
                candidates=ml_results,
                cross_encoder=state.cross_encoder,
            )

        latency_ms = (time.time() - start) * 1000

        return SearchResponse(
            query=body.query,
            mode=SearchMode.FULL,
            results=[_build_result(r, i + 1) for i, r in enumerate(results)],
            total_results=len(results),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Rank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics", response_model=AnalyticsResponse)
async def analytics():
    if not _latency_log:
        return AnalyticsResponse(
            total_queries=0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            top_queries=[],
            mode_distribution=_mode_counts,
        )

    latencies = list(_latency_log)
    avg = sum(latencies) / len(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]

    from collections import Counter
    top_queries = [q for q, _ in Counter(_query_log).most_common(10)]

    return AnalyticsResponse(
        total_queries=len(_latency_log),
        avg_latency_ms=round(avg, 2),
        p95_latency_ms=round(p95, 2),
        top_queries=top_queries,
        mode_distribution=dict(_mode_counts),
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    state = request.app.state

    def status(obj) -> ModelStatus:
        return ModelStatus.LOADED if obj is not None else ModelStatus.NOT_LOADED

    all_loaded = all([
        state.bm25_index, state.faiss_index,
        state.embedding_model, state.ranker, state.cross_encoder
    ])

    return HealthResponse(
        status="ok" if all_loaded else "degraded",
        bm25_index=status(state.bm25_index),
        faiss_index=status(state.faiss_index),
        embedding_model=status(state.embedding_model),
        ml_ranker=status(state.ranker),
        cross_encoder=status(state.cross_encoder),
    )