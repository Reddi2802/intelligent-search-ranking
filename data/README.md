# Data

## Dataset — MS MARCO Passage Ranking

This project uses the [MS MARCO Passage Ranking](https://huggingface.co/datasets/ms_marco) dataset, available on HuggingFace.

## What we use

| Split | Queries | Passages | Relevance Labels |
|-------|---------|----------|-----------------|
| Train | 808,731 | ~8.8M | Sparse (1 positive per query) |
| Validation | 101,093 | ~8.8M | Sparse |
| Dev small | 6,980 | ~8.8M | Sparse |

We start with **10,000 passages** from the train split for development and benchmarking. The pipeline is designed to scale to the full corpus without code changes.

## How to obtain

The dataset is loaded programmatically via HuggingFace `datasets` — no manual download needed:

```python
from datasets import load_dataset
dataset = load_dataset("ms_marco", "v2.1")
```

## What is not committed to git

Raw data files, processed parquet files, and FAISS indexes are excluded via `.gitignore`. Only this README is committed. All data is fetched at runtime via the setup script.
