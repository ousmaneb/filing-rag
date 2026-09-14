from secrag.parse import fiscal_year_focus, html_to_text, split_items

TOC = "\n".join(
    [
        "| Item 1. | Business | 4 |",
        "| --- | --- | --- |",
        "| Item 1A. | Risk Factors | 12 |",
        "| Item 1B. | Unresolved Staff Comments | 25 |",
        "| Item 7. | Management's Discussion and Analysis | 30 |",
        "| Item 8. | Financial Statements | 50 |",
    ]
)

DOC = "\n\n".join(
    [
        "PART I",
        TOC,
        "Item 1. Business",
        "We design semiconductors for data centers. " * 10,
        "Item 1A. Risk Factors",
        "Demand for our products is cyclical. " * 10,
        "Item 1B. Unresolved Staff Comments",
        "None.",
        "Item 7. Management's Discussion and Analysis",
        "Revenue grew 14% to $25.8 billion. " * 10,
        "Item 8. Financial Statements",
        "See the consolidated balance sheets. " * 10,
    ]
)


def test_split_items_skips_table_of_contents():
    sections = {s.item: s.text for s in split_items(DOC)}
    assert sections["7"].startswith("Revenue grew 14%")
    assert "Financial Statements" not in sections["7"]
    assert sections["1"].startswith("We design")
    assert sections["1A"].startswith("Demand")


def test_short_item_falls_back_to_last_occurrence():
    sections = {s.item: s.text for s in split_items(DOC)}
    assert sections["1B"] == "None."


def test_table_becomes_one_markdown_block():
    html = """<html><body>
      <div>Revenue summary</div>
      <table>
        <tr><td></td><td>2024</td><td></td></tr>
        <tr><td>Net revenue</td><td>$</td><td>5,872</td><td></td></tr>
        <tr><td>Operating loss</td><td>(</td><td>12</td><td>)</td></tr>
      </table>
      <div>After the <span>table</span>.</div>
    </body></html>"""
    paragraphs = html_to_text(html).split("\n\n")
    assert paragraphs[0] == "Revenue summary"
    assert paragraphs[2] == "After the table."
    assert "| Net revenue | $5,872 |" in paragraphs[1]
    assert "| Operating loss | (12) |" in paragraphs[1]


def test_hidden_xbrl_header_is_dropped_but_fiscal_year_is_read():
    html = """<html><body>
      <div style="display:none"><ix:header>
        <ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c1">2024</ix:nonNumeric>
      </ix:header></div>
      <p>Body text</p>
    </body></html>"""
    assert html_to_text(html) == "Body text"
    assert fiscal_year_focus(html) == 2024
    assert fiscal_year_focus("<p>no tags</p>") is None


def test_text_outside_kept_items_is_not_dropped():
    index = "\n".join(
        [
            "| Item 1. | Business | Pages 3 - 24 |",
            "| --- | --- | --- |",
            "| Item 1A. | Risk Factors | Pages 37 - 51 |",
            "| Item 7. | Management's Discussion and Analysis | Pages 25 - 36 |",
        ]
    )
    text = "\n\n".join(
        [
            "Our foundry business grew. " * 100,
            "Item 15. Exhibits and Financial Statement Schedules",
            "Consolidated statements of income follow. " * 20,
            index,
        ]
    )
    sections = {s.item: s.text for s in split_items(text)}
    assert list(sections) == [""]
    assert "Our foundry business grew." in sections[""]
    assert "Consolidated statements of income" in sections[""]
