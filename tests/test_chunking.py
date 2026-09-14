import re

from secrag.chunking import fixed_chunks, structural_chunks

TABLE = "\n".join(
    ["| Segment | 2024 | 2023 |", "| --- | --- | --- |"]
    + [f"| Segment {i} | ${i},000 | ${i},500 |" for i in range(30)]
)


def words(text):
    return [m.span() for m in re.finditer(r"\S+", text)]


def section(item, text, title="Title", doc_id="AMD_FY2024"):
    return {
        "doc_id": doc_id,
        "ticker": doc_id.split("_")[0],
        "fiscal_year": 2024,
        "item": item,
        "title": title,
        "text": text,
    }


def test_table_survives_structural_chunking_intact():
    text = "\n\n".join(["Intro words here. " * 40, TABLE, "Closing words here. " * 40])
    chunks = structural_chunks([section("7", text)], words, budget=300)
    assert len(chunks) == 3
    with_table = [c for c in chunks if "| Segment" in c.text]
    assert len(with_table) == 1
    assert TABLE in with_table[0].text


def test_oversized_table_splits_by_rows_with_header_repeated():
    chunks = structural_chunks([section("8", TABLE)], words, budget=100)
    assert len(chunks) > 1
    header = "| Segment | 2024 | 2023 |\n| --- | --- | --- |"
    for c in chunks:
        assert header in c.text
        assert len(words(c.text)) <= 100
    rows = [line for c in chunks for line in c.text.split("\n") if re.match(r"\| Segment \d", line)]
    assert rows == TABLE.split("\n")[2:]


def test_structural_chunks_stay_within_items_and_carry_header():
    sections = [
        section("1A", "Risk is high. " * 60, "Risk Factors"),
        section("7", "Revenue grew fast. " * 60, "Management's Discussion and Analysis"),
    ]
    chunks = structural_chunks(sections, words, budget=100)
    assert len(chunks) > 2
    for c in chunks:
        assert len(words(c.text)) <= 100
        if c.item == "1A":
            assert c.text.startswith("AMD FY2024 10-K — Item 1A: Risk Factors\n\n")
            assert "Revenue" not in c.text
        else:
            assert "Risk" not in c.text


def test_fixed_windows_overlap_and_cover_every_token():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = [c.text.split() for c in fixed_chunks([section("1", text)], words, 100, 10)]
    assert chunks[0][0] == "w0"
    assert chunks[-1][-1] == "w999"
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert len(prev) <= 100
        assert prev[-10:] == cur[:10]


def test_fixed_windows_straddle_item_boundaries():
    sections = [section("1A", "risk " * 30), section("7", "revenue " * 30)]
    chunks = fixed_chunks(sections, words, window=100, overlap=10)
    assert len(chunks) == 1
    assert "risk" in chunks[0].text and "revenue" in chunks[0].text


def test_chunk_ids_are_unique_across_docs_and_strategies():
    ids = []
    for doc_id in ["AMD_FY2024", "AMD_FY2023"]:
        sections = [
            section("1A", "Risk is high. " * 60, doc_id=doc_id),
            section("7", TABLE, doc_id=doc_id),
        ]
        ids += [c.chunk_id for c in fixed_chunks(sections, words, 50, 5)]
        ids += [c.chunk_id for c in structural_chunks(sections, words, 50)]
    assert len(ids) == len(set(ids))
