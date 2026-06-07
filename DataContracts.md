# Data Contracts

Defines the input/output contracts between every component in the pipeline. Any change to these schemas must be reflected across all dependent components.

---

## API Endpoints

### `POST /search`

Runs the full pipeline: query processing → hybrid retrieval → ML ranking → neural reranking.

#### POST /search — request

```json
{
  "query": "string",
  "top_k": "integer (default: 10)",
  "mode": "string (enum: hybrid | bm25 | semantic, default: hybrid)"
}
```

#### POST /search — response

```json
{
  "query": "string",
  "mode": "string",
  "results": [
    {
      "passage_id": "string",
      "text": "string",
      "score": "float",
      "rank": "integer",
      "explanation": {
        "bm25_score": "float",
        "semantic_score": "float",
        "ml_score": "float",
        "rerank_score": "float"
      }
    }
  ],
  "latency_ms": "float",
  "stages_run": ["string"]
}
```

---

### `POST /rank`

Re-ranks a provided candidate set without running retrieval. Used for benchmarking ranking in isolation.

#### POST /rank — request

```json
{
  "query": "string",
  "candidates": [
    {
      "passage_id": "string",
      "text": "string"
    }
  ],
  "method": "string (enum: lightgbm | crossencoder | both, default: both)"
}
```

#### POST /rank — response

```json
{
  "query": "string",
  "ranked_results": [
    {
      "passage_id": "string",
      "text": "string",
      "score": "float",
      "rank": "integer"
    }
  ],
  "latency_ms": "float"
}
```

---

### `GET /analytics`

Returns aggregated query analytics.

#### GET /analytics — response

```json
{
  "total_queries": "integer",
  "avg_latency_ms": "float",
  "p95_latency_ms": "float",
  "top_queries": ["string"],
  "zero_result_queries": ["string"],
  "mode_distribution": {
    "hybrid": "integer",
    "bm25": "integer",
    "semantic": "integer"
  }
}
```

---

### `GET /health`

#### GET /health — response

```json
{
  "status": "string (ok | degraded)",
  "bm25_index": "string (loaded | not_loaded)",
  "faiss_index": "string (loaded | not_loaded)",
  "ml_model": "string (loaded | not_loaded)",
  "reranker": "string (loaded | not_loaded)"
}
```

---

## Internal Component Contracts

### Query Processor → Retrieval

```python
# Input
query: str  # raw user query

# Output
ProcessedQuery(
    original: str,
    normalized: str,       # lowercased, punctuation removed
    tokens: list[str],     # tokenized
    expanded: list[str],   # synonyms added if applicable
)
```

### Retrieval → Ranking

```python
# Output of hybrid retrieval consumed by ML ranker
RetrievalResult(
    passage_id: str,
    text: str,
    bm25_score: float,
    semantic_score: float,
    rrf_score: float,
    retrieval_rank: int,
)
```

### Ranking → Reranking

```python
# Top N candidates from ML ranker passed to cross-encoder
RankedCandidate(
    passage_id: str,
    text: str,
    ml_score: float,
    ml_rank: int,
)
```

### Reranking → API

```python
# Final output of pipeline
FinalResult(
    passage_id: str,
    text: str,
    final_score: float,
    final_rank: int,
    explanation: dict,     # all intermediate scores
)
```
