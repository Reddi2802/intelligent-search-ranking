"""
Central configuration. All paths are absolute, anchored to project root.
Import this everywhere instead of using Path() directly.
"""

from pathlib import Path

# Project root — works regardless of where Python is invoked from
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

# Data files
PASSAGES_FILE = RAW_DIR / "passages.json"
QUERIES_FILE = RAW_DIR / "queries.json"
QRELS_FILE = RAW_DIR / "qrels.json"
TRAINING_CACHE = RAW_DIR / "training_data.json"

# Index files
BM25_INDEX_FILE = INDEX_DIR / "bm25.pkl"
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
PASSAGE_MAP_FILE = INDEX_DIR / "passage_map.pkl"

# Model files
RANKER_FILE = MODEL_DIR / "lightgbm_ranker.pkl"

# Model names
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 256

# Pipeline settings
TOP_K_RETRIEVAL = 100
TOP_K_RERANK = 20
RRF_K = 60

# Ensure directories exist
for directory in [RAW_DIR, INDEX_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)