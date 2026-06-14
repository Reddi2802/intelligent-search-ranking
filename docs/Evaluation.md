# Evaluation

Benchmark results will be filled in as each pipeline stage is built and evaluated.

## Metric — NDCG@10

All stages are evaluated on MS MARCO passage ranking using NDCG@10.

| Stage | Method | NDCG@10 |
|-------|--------|---------|
| Baseline | BM25 only | 0.4771 |
| Stage 3 | Semantic only (FAISS) | 0.6879 |
| Stage 4 | Hybrid (BM25 + Semantic, RRF) | 0.6146 |
| Stage 5 | + ML Ranking (LightGBM) | *TBD* |
| Stage 6 | + Neural Reranking | *TBD* |

## Retrieval Stage Analysis

**BM25 (0.4771):** Strong keyword matching baseline. Fails on synonyms and semantic intent — MS MARCO queries are natural language questions, which disadvantages pure keyword retrieval.

**Semantic FAISS (0.6879):** 44% improvement over BM25. The `all-MiniLM-L6-v2` model captures query intent well. Dense retrieval is clearly better suited for this dataset.

**Hybrid RRF (0.6146):** Naive equal-weight RRF underperforms semantic alone. BM25's weakness on this dataset means including it dilutes the stronger semantic signal. Expected to recover at the ML ranking stage where the model learns optimal feature weighting from data.
