# Evaluation

Benchmark results will be filled in as each pipeline stage is built and evaluated.

## Metric — NDCG@10

All stages are evaluated on MS MARCO passage ranking using NDCG@10.

| Stage | Method | NDCG@10 |
|-------|--------|---------|
| Baseline | BM25 only | *TBD* |
| Stage 3 | Semantic only | *TBD* |
| Stage 4 | Hybrid (BM25 + Semantic) | *TBD* |
| Stage 5 | + ML Ranking (LightGBM) | *TBD* |
| Stage 6 | + Neural Reranking | *TBD* |
