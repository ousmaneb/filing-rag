import math
import re

# Relevance is labelled as (doc_id, quote) rather than chunk IDs, because chunk IDs differ
# between strategies. A fixed window that cuts the quote in half does not count as a hit,
# which is exactly the failure the baseline is meant to show.
NON_WORD = re.compile(r"[^\w.]+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
CITATION = re.compile(r"\[\d+\]")


def normalize(text: str) -> str:
    # Drops table pipes, "$" and thousands separators so a quote copied from the rendered
    # filing matches the markdown table it was chunked into.
    return NON_WORD.sub(" ", text.lower()).strip()


def relevance(chunks: list[dict], evidence: list[dict]) -> list[set[int]]:
    quotes = [(e["doc_id"], normalize(e["quote"])) for e in evidence]
    return [
        {
            i
            for i, (doc_id, quote) in enumerate(quotes)
            if c["doc_id"] == doc_id and quote in normalize(c["text"])
        }
        for c in chunks
    ]


def recall_at_k(matches: list[set[int]], n_evidence: int, k: int) -> float:
    if not n_evidence:
        return math.nan
    return len(set().union(*matches[:k])) / n_evidence


def hit_at_k(matches: list[set[int]], n_evidence: int, k: int) -> float:
    if not n_evidence:
        return math.nan
    return float(any(matches[:k]))


def mrr(matches: list[set[int]], n_evidence: int) -> float:
    if not n_evidence:
        return math.nan
    return next((1 / rank for rank, m in enumerate(matches, 1) if m), 0.0)


def ndcg_at_k(matches: list[set[int]], n_evidence: int, k: int) -> float:
    if not n_evidence:
        return math.nan
    dcg = sum(1 / math.log2(rank + 1) for rank, m in enumerate(matches[:k], 1) if m)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(n_evidence, k) + 1))
    return dcg / ideal


def refusal_matches(refused: bool, answerable: bool) -> bool:
    return refused != answerable


def uncited_claim_rate(text: str) -> float:
    claims = [s for s in SENTENCE_END.split(text) if re.search(r"\d", CITATION.sub("", s))]
    if not claims:
        return math.nan
    return sum(not CITATION.search(s) for s in claims) / len(claims)
