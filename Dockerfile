FROM python:3.11-slim

WORKDIR /app

# The default PyPI torch wheel for Linux bundles CUDA and adds gigabytes the API never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
# Editable, so config.ROOT resolves to /app and data/ is found next to src/.
RUN pip install --no-cache-dir -e ".[ml]"

# Bake the models into the image so a cold start doesn't download them.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

# BM25 is built from the chunk files at startup; the vectors live in Qdrant.
COPY data/chunks ./data/chunks

EXPOSE 8000
CMD ["uvicorn", "secrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
