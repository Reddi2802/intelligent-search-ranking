# Intelligent Hybrid Search & Ranking Platform

This started as "let me build a search engine" and turned into something much bigger. What I'm actually building is a modular retrieval and ranking system — the kind of architecture that underlies Google Search, LinkedIn, and enterprise knowledge bases. Not the UI, not the brand, but the actual pipeline underneath.

The reason I chose this as my main project is that it secretly teaches about ten different things at once: information retrieval, embeddings, vector databases, ML ranking, backend APIs, evaluation methodology, and deployment. Most projects teach one or two. This one ties all of them together into something that actually makes sense as a system.

---

## System Architecture

```
                        User Query
                            │
                            ▼
              ┌─────────────────────────┐
              │     Query Processing    │
              │  cleaning · expansion   │
              │  spell correction       │
              └─────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
   ┌─────────────────────┐   ┌─────────────────────┐
   │    BM25 Retrieval   │   │  Semantic Retrieval  │
   │   (sparse / fast)   │   │   FAISS + embeddings │
   └─────────────────────┘   └─────────────────────┘
               │                         │
               └────────────┬────────────┘
                            ▼
              ┌─────────────────────────┐
              │   Hybrid Candidate Set  │
              │     top ~100 docs       │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │     ML Ranking Layer    │
              │  XGBoost / LightGBM     │
              │  feature engineering    │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │    Neural Reranking     │
              │    cross-encoder        │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   Personalization +     │
              │   Click Feedback        │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   Ranked Results + UI   │
              │   explanations · stats  │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Analytics Dashboard    │
              │  latency · click data   │
              └─────────────────────────┘
```

---

## Where You Start

- comfortable with Python
- basic understanding of what an API is

## Where You Stop (Defined Scope)

- [ ] BM25 retrieval over a real document corpus
- [ ] Sentence embeddings + FAISS for semantic retrieval
- [ ] Hybrid retrieval combining both signals
- [ ] Feature extraction pipeline (BM25 score, cosine sim, doc length, recency)
- [ ] XGBoost/LightGBM ranker trained on labeled pairs
- [ ] Cross-encoder neural reranker on top of the ML ranking layer
- [ ] NDCG@10 evaluation script benchmarking all stages against each other
- [ ] FastAPI backend with `/search`, `/rank`, and `/analytics` endpoints
- [ ] Streamlit frontend with mode toggles (BM25 / semantic / hybrid) and result explanations
- [ ] Basic click tracking and a dashboard showing query trends and latency
- [ ] Docker deployment (optional stretch goal at the very end)

**Not building:**
- A web crawler (use an existing dataset, stretch goal if you want to build this as a separate service in docker)
- A distributed system (single machine is fine)
- Real personalization from user history (lightweight interest-aware reranking is enough) and Click feedback

**Dataset:** MS MARCO passage ranking on HuggingFace — already has queries, passages, and relevance labels. Start with 10k passages. Scale up later if you want.

---

## Build Phases

This is too big to build all at once. The phases exist so there's always something working at every stage.

**Phase 1 — Engineering Foundations**
Learn venv, pip, and project structure using **uv package and env manager**, Docker for containerization, and APIs with FastAPI. Familiarize yourself with Python. Set up the repo cleanly from the start so adding features later doesn't become chaos.

**Phase 2 — Core Retrieval**
Get BM25 working first. Then add embeddings. Then FAISS. Then combine them into hybrid retrieval. Evaluate each step so you actually know if hybrid is better than BM25 alone (it will be).

**Phase 3 — ML Ranking**
Extract features per (query, document) pair. Train XGBoost on those features. Add the cross-encoder reranker. Benchmark everything with NDCG@10.

**Phase 4 — UI Layer**
Build the Streamlit frontend. Add results, if possible add explanations (why did this rank first?). Add search mode toggles. (stretch goal - add the analytics dashboard as part of UX).

**Phase 5 — Advanced Features**
Caching for repeated queries. Latency optimization. Query expansion for typos and synonyms using basic embeddings (do not delve into query history yet).

**Phase 6 — Polish**
Docker. README with architecture diagrams. Benchmark writeup comparing stages. See how to deploy this project as different services like query pare-BM25-FAISS, ML ranking, neural reranking, and final UI.

---

## Tools & Libraries

| Tool | What It's For |
|------|--------------|
| `rank_bm25` | BM25 retrieval (sparse keyword matching) |
| `sentence-transformers` | Generating semantic embeddings |
| `faiss-cpu` | Fast approximate nearest neighbor search over vectors |
| `lightgbm` / `xgboost` | ML ranking model (LambdaMART variant) |
| `scikit-learn` | Feature scaling, evaluation helpers |
| `fastapi` + `uvicorn` | REST API backend |
| `streamlit` | Frontend UI |
| `datasets` (HuggingFace) | Loading MS MARCO without scraping |
| `pandas` / `numpy` | Data wrangling |

---

## What I'll Learn

**Information Retrieval**
- Why BM25 works, and exactly where it breaks down (synonyms, intent, context)
- The retrieval/ranking split — why you can't rank 10 million documents per query and what to do about it
- Sparse vs dense retrieval and why hybrid almost always wins over either alone
- FAISS and approximate nearest neighbor search — foundational for any modern search or RAG system

**Machine Learning Engineering**
- Feature engineering for ranking: turning raw signals (BM25 score, cosine similarity, document length) into model inputs
- Learning-to-rank as a problem type — different from classification, different loss functions, different evaluation
- Why NDCG@10 exists and why accuracy is completely useless for evaluating ranked lists
- Neural reranking: retrieve broadly, then use a heavier model for precision

**Backend and Systems**
- Building and documenting a real REST API with FastAPI
- Async patterns for retrieval pipelines
- Caching, latency measurement, and thinking about throughput
- Modular architecture — separating retrieval, ranking, and serving into distinct components

**Product Thinking**
- Click tracking as implicit relevance feedback
- Search analytics: query trends, zero-result queries, latency percentiles
- Result explainability — showing users *why* something ranked where it did

**Software Engineering Fundamentals**
- Virtual environments and dependency management
- Project structure that doesn't fall apart as the codebase grows
- Writing a README that actually communicates what the project does

---

## Why This Project and Not Something Simpler

The honest reason is that I wanted one project that I could keep building on for months without it feeling done. Every phase here adds something genuinely new to learn — it doesn't just increase the number of rows in a database or add another API endpoint that does the same thing differently.

The other reason is that search is everywhere. Whether I end up working in data engineering, ML, or backend, understanding how retrieval and ranking work together is directly applicable. RAG systems, recommendation engines, enterprise search, semantic similarity — they're all variations of this architecture.

The project is intentionally scoped to stay on one machine. The goal is depth, not scale.

---

## What This Looks Like on a Resume

> *"Built a multi-stage hybrid search and ranking platform with BM25 + FAISS retrieval, LightGBM ranking, and cross-encoder reranking. Evaluated across stages using NDCG@10 on MS MARCO. Served via FastAPI with a Streamlit analytics frontend."*

That sentence covers information retrieval, ML ranking, vector search, evaluation methodology, API development, and frontend — six distinct skill areas in one project.
