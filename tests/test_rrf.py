from secrag.rrf import rrf


def test_consistent_second_place_beats_a_single_first_place():
    fused = [doc for doc, _ in rrf([["a", "b"], ["c", "b"]])]
    assert fused[0] == "b"


def test_single_ranking_keeps_its_order():
    assert [doc for doc, _ in rrf([["x", "y", "z"]])] == ["x", "y", "z"]


def test_docs_from_either_list_are_kept():
    assert {doc for doc, _ in rrf([["a"], ["b"]])} == {"a", "b"}


def test_larger_k_flattens_the_gap_between_ranks():
    def gap(k):
        scores = dict(rrf([["first", "second"]], k=k))
        return scores["first"] - scores["second"]

    assert gap(1) > gap(60)
