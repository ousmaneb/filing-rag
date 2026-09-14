import re
from collections.abc import Callable
from dataclasses import dataclass

from secrag.config import EMBED_MODEL

# Maps text to the (start, end) character span of each token. Passed in so tests can
# use a whitespace tokenizer instead of downloading BGE's.
Spans = Callable[[str], list[tuple[int, int]]]

WINDOW = 512
OVERLAP = 64
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ticker: str
    fiscal_year: int
    strategy: str
    item: str | None
    text: str


def bge_spans() -> Spans:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(EMBED_MODEL)
    return lambda text: tokenizer.encode(text, add_special_tokens=False).offsets


def fixed_chunks(
    sections: list[dict], spans: Spans, window: int = WINDOW, overlap: int = OVERLAP
) -> list[Chunk]:
    doc = sections[0]
    text = "\n\n".join(s["text"] for s in sections)
    offsets = spans(text)
    chunks = []
    for start in range(0, len(offsets), window - overlap):
        end = min(start + window, len(offsets))
        chunks.append(
            Chunk(
                chunk_id=f"{doc['doc_id']}:fixed:{len(chunks)}",
                doc_id=doc["doc_id"],
                ticker=doc["ticker"],
                fiscal_year=doc["fiscal_year"],
                strategy="fixed",
                item=None,
                text=text[offsets[start][0] : offsets[end - 1][1]],
            )
        )
        if end == len(offsets):
            break
    return chunks


def structural_chunks(sections: list[dict], spans: Spans, budget: int = WINDOW) -> list[Chunk]:
    chunks = []
    for s in sections:
        # Without the header, a bare table of numbers embeds to a near-meaningless vector.
        header = f"{s['ticker']} FY{s['fiscal_year']} 10-K"
        if s["item"]:
            header += f" — Item {s['item']}: {s['title']}"
        body_budget = budget - len(spans(header))
        units = []
        for para in s["text"].split("\n\n"):
            lines = para.split("\n")
            if len(spans(para)) <= body_budget:
                units.append(para)
            elif para.startswith("|") and len(lines) > 2:
                # The embedder reads 512 tokens, so a whole oversized table is partly invisible
                # to dense search. Split by rows and repeat the header row in every piece so
                # no row is cut off from its column labels.
                columns = "\n".join(lines[:2])
                row_budget = body_budget - len(spans(columns))
                units.extend(
                    f"{columns}\n{rows}" for rows in pack(lines[2:], spans, row_budget, "\n")
                )
            else:
                units.extend(pack(SENTENCE_END.split(para), spans, body_budget, " "))
        for body in pack(units, spans, body_budget, "\n\n"):
            chunks.append(
                Chunk(
                    chunk_id=f"{s['doc_id']}:structural:{len(chunks)}",
                    doc_id=s["doc_id"],
                    ticker=s["ticker"],
                    fiscal_year=s["fiscal_year"],
                    strategy="structural",
                    item=s["item"],
                    text=f"{header}\n\n{body}",
                )
            )
    return chunks


def pack(pieces: list[str], spans: Spans, budget: int, sep: str) -> list[str]:
    out, current, used = [], [], 0
    for piece in pieces:
        n = len(spans(piece))
        if current and used + n > budget:
            out.append(sep.join(current))
            current, used = [], 0
        current.append(piece)
        used += n
    if current:
        out.append(sep.join(current))
    return out
