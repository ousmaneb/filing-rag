import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

UNIVERSE = {
    "semiconductor": ["AMD", "NVDA", "INTC", "AVGO", "TXN"],
    "retail": ["WMT", "TGT", "COST", "HD", "LOW"],
    "banking": ["JPM", "BAC", "WFC", "GS", "MS"],
}
TICKERS = [t for tickers in UNIVERSE.values() for t in tickers]
FISCAL_YEARS_PER_TICKER = 2

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

# Model IDs live in env: provider catalogues change faster than this repo will.
GENERATION_MODEL = os.environ.get("SECRAG_GENERATION_MODEL", "claude-opus-5")
JUDGE_MODEL = os.environ.get("SECRAG_JUDGE_MODEL", "claude-opus-5")


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    chunk_strategy: Literal["fixed", "structural"]
    use_dense: bool = True
    use_sparse: bool = False
    use_rerank: bool = False
    rrf_k: int = 60
    candidate_k: int = 50
    final_k: int = 8

    @property
    def collection(self) -> str:
        return f"chunks_{self.chunk_strategy}"


# Each row changes exactly one thing from the row above, so the deltas in the
# ablation table are attributable.
VARIANTS = {
    v.name: v
    for v in [
        PipelineConfig("v0_baseline", chunk_strategy="fixed"),
        PipelineConfig("v1_structural_chunks", chunk_strategy="structural"),
        PipelineConfig("v2_hybrid_rrf", chunk_strategy="structural", use_sparse=True),
        PipelineConfig("v3_rerank", chunk_strategy="structural", use_sparse=True, use_rerank=True),
    ]
}
