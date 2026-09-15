import math

from secrag.eval.scoring import (
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    refusal_matches,
    relevance,
    uncited_claim_rate,
)


def test_retrieval_metrics_are_nan_without_labels():
    matches = [set(), {0}]
    assert math.isnan(recall_at_k(matches, 0, 8))
    assert math.isnan(hit_at_k(matches, 0, 8))
    assert math.isnan(mrr(matches, 0))
    assert math.isnan(ndcg_at_k(matches, 0, 8))


def test_recall_hit_and_mrr():
    matches = [set(), {0}, set(), {1}]
    assert recall_at_k(matches, 2, 2) == 0.5
    assert recall_at_k(matches, 2, 4) == 1.0
    assert hit_at_k(matches, 2, 1) == 0.0
    assert hit_at_k(matches, 2, 2) == 1.0
    assert mrr(matches, 2) == 0.5
    assert mrr([set(), set()], 2) == 0.0


def test_ndcg_rewards_earlier_hits():
    early = ndcg_at_k([{0}, set(), set()], 1, 3)
    late = ndcg_at_k([set(), set(), {0}], 1, 3)
    assert early == 1.0
    assert 0 < late < early


def test_evidence_quote_matches_through_table_formatting():
    chunks = [
        {"doc_id": "NVDA_FY2025", "text": "| Data Center | $ | 115,186 | $ | 47,525 |"},
        {"doc_id": "AMD_FY2024", "text": "| Data Center | $ | 115,186 |"},
    ]
    evidence = [{"doc_id": "NVDA_FY2025", "quote": "Data Center $115,186"}]
    assert relevance(chunks, evidence) == [{0}, set()]


def test_evidence_quote_ending_a_sentence_matches_mid_sentence_text():
    chunks = [
        {"doc_id": "WMT", "text": "U.S. sales contributed 3.3% and 2.3%, respectively, which"}
    ]
    sentence_end = relevance(chunks, [{"doc_id": "WMT", "quote": "and 2.3%, respectively."}])
    decimal_dropped = relevance(chunks, [{"doc_id": "WMT", "quote": "contributed 33%"}])
    assert sentence_end == [{0}]
    assert decimal_dropped == [set()]


def test_uncited_claim_rate_counts_numeric_sentences_without_citations():
    text = "Revenue was $25.8 billion in fiscal 2025 [1]. Margins rose to 61%. Demand was strong."
    assert uncited_claim_rate(text) == 0.5
    assert math.isnan(uncited_claim_rate("INSUFFICIENT_CONTEXT\nNo segment data is given."))


def test_refusal_matches():
    assert refusal_matches(refused=True, answerable=False)
    assert refusal_matches(refused=False, answerable=True)
    assert not refusal_matches(refused=True, answerable=True)
