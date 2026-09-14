from collections import defaultdict

# Fuses on rank, not score. Cosine similarity is bounded and BM25 is unbounded and
# corpus-dependent, so any weighted sum of the two needs a normalisation step that is
# itself a hyperparameter. RRF discards the magnitudes entirely.


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (k + rank)
    # sorted() is stable and dicts keep insertion order, so ties go to the first-seen doc.
    return sorted(scores.items(), key=lambda item: -item[1])
