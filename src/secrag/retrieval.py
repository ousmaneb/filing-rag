import json
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from secrag.config import DATA_DIR, EMBED_MODEL, QDRANT_URL, RERANK_MODEL, PipelineConfig
from secrag.query import filters, tokenize
from secrag.rrf import rrf

# bge-base-en-v1.5 ships with empty prompts, so its query instruction is added here.
# Documents are embedded without one.
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


@dataclass
class Hit:
    chunk: dict
    score: float
    ranks: dict[str, int]


@dataclass
class Corpus:
    chunks: list[dict]
    by_id: dict[str, dict]
    bm25: BM25Okapi
    tickers: np.ndarray
    years: np.ndarray


class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.corpora: dict[str, Corpus] = {}

    def corpus(self, strategy: str) -> Corpus:
        if strategy not in self.corpora:
            with open(DATA_DIR / "chunks" / f"{strategy}.jsonl") as f:
                chunks = [json.loads(line) for line in f]
            self.corpora[strategy] = Corpus(
                chunks=chunks,
                by_id={c["chunk_id"]: c for c in chunks},
                bm25=BM25Okapi([tokenize(c["text"]) for c in chunks]),
                tickers=np.array([c["ticker"] for c in chunks]),
                years=np.array([c["fiscal_year"] for c in chunks]),
            )
        return self.corpora[strategy]

    def retrieve(self, query: str, cfg: PipelineConfig) -> list[Hit]:
        corpus = self.corpus(cfg.chunk_strategy)
        tickers, years = filters(query)
        ranked = {}
        if cfg.use_dense:
            ranked["dense"] = self.dense(query, cfg, tickers, years)
        if cfg.use_sparse:
            ranked["sparse"] = self.sparse(query, corpus, cfg.candidate_k, tickers, years)

        if len(ranked) == 2:
            fused = rrf([[cid for cid, _ in r] for r in ranked.values()], cfg.rrf_k)
            scored = fused[: cfg.candidate_k]
        else:
            scored = next(iter(ranked.values()))

        positions = {
            name: {cid: i for i, (cid, _) in enumerate(r, 1)} for name, r in ranked.items()
        }
        hits = [
            Hit(
                chunk=corpus.by_id[cid],
                score=score,
                ranks={name: pos[cid] for name, pos in positions.items() if cid in pos},
            )
            for cid, score in scored
        ]

        if cfg.use_rerank and hits:
            scores = self.reranker.predict([(query, h.chunk["text"]) for h in hits])
            for hit, score in zip(hits, scores, strict=True):
                hit.score = float(score)
            hits.sort(key=lambda h: -h.score)
        return hits[: cfg.final_k]

    def dense(
        self, query: str, cfg: PipelineConfig, tickers: list[str], years: list[int]
    ) -> list[tuple[str, float]]:
        vector = self.embedder.encode_query(query, prompt=QUERY_PROMPT, normalize_embeddings=True)
        conditions = []
        if tickers:
            conditions.append(
                models.FieldCondition(key="ticker", match=models.MatchAny(any=tickers))
            )
        if years:
            conditions.append(
                models.FieldCondition(key="fiscal_year", match=models.MatchAny(any=years))
            )
        result = self.qdrant.query_points(
            cfg.collection,
            query=vector.tolist(),
            query_filter=models.Filter(must=conditions) if conditions else None,
            limit=cfg.candidate_k,
            with_payload=["chunk_id"],
        )
        return [(p.payload["chunk_id"], p.score) for p in result.points]

    def sparse(
        self, query: str, corpus: Corpus, limit: int, tickers: list[str], years: list[int]
    ) -> list[tuple[str, float]]:
        scores = corpus.bm25.get_scores(tokenize(query))
        # Apply the same filters as the dense side, or hybrid search leaks the wrong
        # company back in through BM25.
        if tickers:
            scores[~np.isin(corpus.tickers, tickers)] = 0
        if years:
            scores[~np.isin(corpus.years, years)] = 0
        top = np.argsort(-scores)[:limit]
        return [(corpus.chunks[i]["chunk_id"], float(scores[i])) for i in top if scores[i] > 0]
