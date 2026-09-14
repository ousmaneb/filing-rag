import re
import time
from dataclasses import dataclass

import anthropic

from secrag.config import GENERATION_MODEL, PipelineConfig
from secrag.retrieval import Hit, Retriever

SYSTEM = """You answer questions about SEC 10-K filings using only the numbered excerpts in the user message.

- Use only the excerpts. Do not use outside knowledge, even when you know the answer.
- Cite every factual claim with the number of the excerpt that supports it, like [2]. A claim drawn from two excerpts cites both, like [1][3].
- Report figures with their units and the period they cover, for example "$25.8 billion in fiscal 2025".
- If the excerpts do not contain the answer, reply with exactly INSUFFICIENT_CONTEXT on the first line, followed by one sentence saying what information is missing.
- If excerpts conflict with each other, say so and cite both instead of picking one."""

CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: list[int]
    refused: bool
    retrieval_ms: float
    generation_ms: float
    model: str
    input_tokens: int
    output_tokens: int


def format_excerpts(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {h.chunk['ticker']} FY{h.chunk['fiscal_year']} 10-K\n{h.chunk['text']}"
        for i, h in enumerate(hits, 1)
    )


def answer(
    question: str, cfg: PipelineConfig, retriever: Retriever, client: anthropic.Anthropic
) -> Answer:
    start = time.perf_counter()
    hits = retriever.retrieve(question, cfg)
    retrieved = time.perf_counter()

    excerpts = format_excerpts(hits)
    response = client.beta.messages.create(
        model=GENERATION_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"<excerpts>\n{excerpts}\n</excerpts>\n\nQuestion: {question}",
            }
        ],
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()

    return Answer(
        question=question,
        text=text,
        hits=hits,
        # Out-of-range numbers are kept so the citation check can catch them.
        cited=sorted({int(n) for n in CITATION.findall(text)}),
        refused=text.startswith("INSUFFICIENT_CONTEXT"),
        retrieval_ms=(retrieved - start) * 1000,
        generation_ms=(time.perf_counter() - retrieved) * 1000,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
