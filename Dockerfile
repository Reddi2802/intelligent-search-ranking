FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev
RUN uv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Only download embedding model — cross-encoder optional at runtime
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV TRANSFORMERS_OFFLINE=1

COPY src/ ./src/
COPY data/ ./data/
COPY evaluation/ ./evaluation/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]