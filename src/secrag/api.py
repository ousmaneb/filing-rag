import threading
import time
from collections import deque
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from secrag.config import VARIANTS, PipelineConfig
from secrag.generate import answer
from secrag.retrieval import Hit, Retriever

# Crude on purpose: a global hourly cap on /ask so an exposed demo URL can't drain the
# API budget overnight. It resets on restart and isn't shared across worker processes.
ASKS_PER_HOUR = 60

app = FastAPI()
retriever = Retriever()
client = anthropic.Anthropic()
recent_asks: deque[float] = deque()
ask_lock = threading.Lock()
# FastAPI runs sync endpoints in a thread pool; the embedder and reranker on MPS are not
# safe to call from several threads at once.
model_lock = threading.Lock()


class Query(BaseModel):
    question: str
    variant: str = "v3_rerank"


def config(name: str) -> PipelineConfig:
    if name not in VARIANTS:
        raise HTTPException(404, f"unknown variant {name!r}, expected one of {list(VARIANTS)}")
    return VARIANTS[name]


def trace(hits: list[Hit]) -> list[dict]:
    return [
        {
            "index": i,
            "chunk_id": h.chunk["chunk_id"],
            "ticker": h.chunk["ticker"],
            "fiscal_year": h.chunk["fiscal_year"],
            "item": h.chunk["item"],
            "score": h.score,
            "ranks": h.ranks,
            "text": h.chunk["text"],
        }
        for i, h in enumerate(hits, 1)
    ]


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search(query: Query) -> dict:
    cfg = config(query.variant)
    with model_lock:
        hits = retriever.retrieve(query.question, cfg)
    return {"variant": cfg.name, "results": trace(hits)}


@app.post("/ask")
def ask(query: Query) -> dict:
    cfg = config(query.variant)
    with ask_lock:
        now = time.monotonic()
        while recent_asks and now - recent_asks[0] > 3600:
            recent_asks.popleft()
        if len(recent_asks) >= ASKS_PER_HOUR:
            raise HTTPException(429, "hourly /ask limit reached")
        recent_asks.append(now)

    with model_lock:
        a = answer(query.question, cfg, retriever, client)
    return {
        "answer": a.text,
        "refused": a.refused,
        "citations": [
            {"index": n, "chunk_id": a.hits[n - 1].chunk["chunk_id"]}
            for n in a.cited
            if 1 <= n <= len(a.hits)
        ],
        "trace": trace(a.hits),
        "variant": cfg.name,
        "model": a.model,
        "retrieval_ms": a.retrieval_ms,
        "generation_ms": a.generation_ms,
        "input_tokens": a.input_tokens,
        "output_tokens": a.output_tokens,
    }
