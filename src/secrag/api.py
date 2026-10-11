import os
import threading
from pathlib import Path

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from secrag.config import VARIANTS, PipelineConfig
from secrag.generate import answer
from secrag.retrieval import Hit, Retriever
from secrag.security import RateLimiter, api_key_ok, bearer_or_header, client_ip, env_int

# Optional. When set, /ask and /search require it in X-API-Key or "Authorization: Bearer".
# When unset the API stays an open public demo, protected only by the rate limits below.
API_KEY = os.environ.get("SECRAG_API_KEY", "").strip() or None
# Number of reverse proxies in front of the app that append to X-Forwarded-For. 0 (the
# default) ignores the header and uses the socket peer, which can't be spoofed.
TRUST_PROXY_HOPS = env_int("SECRAG_TRUST_PROXY_HOPS", 0)

# In memory on purpose: they reset on restart and aren't shared across worker processes.
# The global /ask cap keeps an exposed demo URL from draining the API budget overnight;
# the per-IP limits stop one caller from using that budget, or the CPU, up for everyone.
ASKS_PER_HOUR = env_int("SECRAG_ASKS_PER_HOUR", 60)
ASKS_PER_IP_PER_HOUR = env_int("SECRAG_ASKS_PER_IP_PER_HOUR", 10)
SEARCHES_PER_IP_PER_MINUTE = env_int("SECRAG_SEARCHES_PER_IP_PER_MINUTE", 20)

app = FastAPI()
retriever = Retriever()
client = anthropic.Anthropic()
ask_limit = RateLimiter(ASKS_PER_HOUR, 3600)
ask_ip_limit = RateLimiter(ASKS_PER_IP_PER_HOUR, 3600)
search_ip_limit = RateLimiter(SEARCHES_PER_IP_PER_MINUTE, 60)
# FastAPI runs sync endpoints in a thread pool; the embedder and reranker on MPS are not
# safe to call from several threads at once.
model_lock = threading.Lock()


class Query(BaseModel):
    # Bounded so one request can't send an arbitrarily large prompt to the model.
    question: str = Field(min_length=1, max_length=2000)
    variant: str = "v3_rerank"


def config(name: str) -> PipelineConfig:
    if name not in VARIANTS:
        raise HTTPException(404, f"unknown variant {name!r}, expected one of {list(VARIANTS)}")
    return VARIANTS[name]


def require_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    if not api_key_ok(API_KEY, bearer_or_header(authorization, x_api_key)):
        raise HTTPException(401, "missing or invalid API key", {"WWW-Authenticate": "Bearer"})


def caller(request: Request) -> str:
    peer = request.client.host if request.client else None
    return client_ip(peer, request.headers.get("x-forwarded-for"), TRUST_PROXY_HOPS)


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


@app.post("/search", dependencies=[Depends(require_key)])
def search(query: Query, ip: str = Depends(caller)) -> dict:
    cfg = config(query.variant)
    if not search_ip_limit.allow(ip):
        raise HTTPException(429, "/search rate limit reached, try again in a minute")
    with model_lock:
        hits = retriever.retrieve(query.question, cfg)
    return {"variant": cfg.name, "results": trace(hits)}


@app.post("/ask", dependencies=[Depends(require_key)])
def ask(query: Query, ip: str = Depends(caller)) -> dict:
    cfg = config(query.variant)
    if not ask_ip_limit.allow(ip):
        raise HTTPException(429, "hourly /ask limit for your address reached")
    if not ask_limit.allow():
        raise HTTPException(429, "hourly /ask limit reached")

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
