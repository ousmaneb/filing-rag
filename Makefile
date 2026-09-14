PY ?= .venv/bin/python

-include .env
export

.PHONY: setup qdrant ingest index eval eval-fast serve test lint clean

setup:
	python3 -m venv .venv
	$(PY) -m pip install -e ".[ml,dev]"

qdrant:
	docker compose up -d qdrant

ingest:
	$(PY) -m secrag.ingest

index:
	$(PY) -m secrag.index

eval:
	$(PY) -m secrag.eval.run_ablation

eval-fast:
	$(PY) -m secrag.eval.run_ablation --retrieval-only

serve:
	$(PY) -m uvicorn secrag.api:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
