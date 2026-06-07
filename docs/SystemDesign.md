# System Design

## Technology Choices

### Retrieval layer

| Decision | Chosen | Alternatives | Reason |
|----------|--------|--------------|--------|
| Sparse retrieval | BM25 (`rank_bm25`) | TF-IDF, Elasticsearch | No infra overhead, interpretable, strong baseline |
| Dense retrieval | FAISS + sentence-transformers | Pinecone, Weaviate, ChromaDB | Free, local, GPU-accelerated |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) | Learned fusion, linear interpolation | No training data needed, robust to score scale differences |
| Embedding model | `all-MiniLM-L6-v2` | `all-mpnet-base-v2`, OpenAI | Best speed/quality tradeoff, 384-dim, runs locally |

### Ranking layer

| Decision | Chosen | Alternatives | Reason |
|----------|--------|--------------|--------|
| ML ranker | LightGBM (LambdaMART) | XGBoost, RankNet | Native LambdaMART, faster training, lower memory |
| Neural reranker | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Bi-encoder, GPT reranking | More accurate than bi-encoder; applied to top 20 only |
| Evaluation metric | NDCG@10 | MRR, MAP, Precision@K | Industry standard; penalizes bad ordering |

### Backend and serving

| Decision | Chosen | Alternatives | Reason |
|----------|--------|--------------|--------|
| API framework | FastAPI | Flask, Django | Async-native, auto OpenAPI docs, Pydantic validation |
| Frontend | Streamlit | Gradio, React | Built for ML apps, no JavaScript required |
| Containerization | Docker + Compose | Bare metal, venv only | Reproducible, production-ready |

### Cloud and infrastructure

| Decision | Chosen | Alternatives | Reason |
|----------|--------|--------------|--------|
| Cloud provider | AWS | GCP, Azure | Largest market share, free tier sufficient |
| Compute | EC2 t2.micro | Lambda, ECS, Fargate | Zero cost, sufficient for pre-loaded index serving |
| Model storage | S3 | EFS, local disk | Durable, cheap, decoupled from compute |
| CI/CD | GitHub Actions | CircleCI, Jenkins | Native GitHub integration, free for public repos |

## Architecture Decisions

### Why the retrieval/ranking split exists

Running a neural reranker over 10 million documents per query is computationally impossible in real time. The pipeline is designed around this constraint: retrieve broadly and cheaply (BM25 + FAISS over the full corpus), then rank precisely and expensively (LightGBM + cross-encoder over ~100 candidates). This is the same design used in production search at Google, LinkedIn, and Airbnb.

### Why hybrid retrieval almost always beats either alone

BM25 excels at exact keyword matches but fails on synonyms and semantic intent. Dense retrieval captures meaning but can miss exact terminology. RRF combines both ranked lists without requiring score normalization — a document that ranks well in both lists gets boosted, a document that only appears in one gets penalized.

### Why cross-encoders only at the final stage

Bi-encoders (used in FAISS retrieval) encode query and document independently — fast but less accurate. Cross-encoders encode them jointly, capturing fine-grained interaction — accurate but slow. Applying a cross-encoder to the full corpus would be unusable. Applied to the top 20 candidates after ML ranking, the latency is acceptable and the precision gain is significant.

### Why NDCG@10 and not accuracy

Accuracy measures whether the right answer is in the results. NDCG@10 measures whether the right answer is ranked first. For a search system, ranking matters more than recall at the top level. NDCG also accounts for graded relevance — a highly relevant document ranked 2nd is better than a marginally relevant document ranked 1st.

### Cost-aware deployment

All heavy computation (embedding generation, model training, index building) runs locally on an RTX 4050. AWS is used only for serving — a pre-built FAISS index and trained model loaded into EC2 on startup, with artifacts stored in S3. This keeps cloud costs at zero on the free tier.

---

## What is deliberately out of scope

- **Web crawler** — MS MARCO provides a pre-built corpus with relevance labels. Building a crawler is a separate project.
- **Distributed search** — Single-machine FAISS is sufficient for MS MARCO scale. Distributed ANN (e.g. using Milvus or Pinecone) is a future enhancement.
- **Real user personalization** — Requires real user history. Out of scope; lightweight interest-aware reranking is sufficient for this project.
- **Query history** — Not stored. Each query is stateless.

---

## Constraints

- Single machine deployment (RTX 4050 for local compute, EC2 t2.micro for serving)
- Zero cloud cost (AWS free tier only)
- MS MARCO passage ranking dataset — starting with 10k passages, scalable to full corpus
