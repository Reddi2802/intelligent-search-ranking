"""
FastAPI application entry point.
Models are loaded once at startup via lifespan and stored in app.state.
All requests reuse loaded models — no per-request loading.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.retrieval.data_loader import load_msmarco
from src.retrieval.bm25_retriever import build_bm25_index
from src.retrieval.semantic_retriever import build_faiss_index, load_embedding_model
from src.ranking.ranker import load_ranker
from src.reranking.cross_encoder_reranker import load_cross_encoder
from src.config import TOP_K_RETRIEVAL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup. Clean up at shutdown."""
    logger.info("Loading pipeline components...")

    passages, queries, qrels = load_msmarco()
    app.state.passages = passages
    app.state.queries = queries
    app.state.qrels = qrels
    logger.info(f"Loaded {len(passages)} passages")

    app.state.bm25_index, _ = build_bm25_index(passages)
    logger.info("BM25 index loaded")

    app.state.embedding_model = load_embedding_model()
    app.state.faiss_index, app.state.passage_map = build_faiss_index(
        passages, app.state.embedding_model
    )
    logger.info("FAISS index loaded")

    app.state.ranker = load_ranker()
    logger.info("LightGBM ranker loaded")

    app.state.cross_encoder = load_cross_encoder()
    logger.info("Cross-encoder loaded")

    logger.info("All components loaded. API ready.")
    yield

    logger.info("Shutting down...")
    app.state.passages = None
    app.state.bm25_index = None
    app.state.faiss_index = None
    app.state.passage_map = None
    app.state.embedding_model = None
    app.state.ranker = None
    app.state.cross_encoder = None


app = FastAPI(
    title="Intelligent Search & Ranking System",
    description="Multi-stage hybrid search with BM25, FAISS, LightGBM, and cross-encoder reranking",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "Intelligent Search & Ranking System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }