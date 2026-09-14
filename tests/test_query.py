from secrag.query import filters, tokenize


def test_ticker_extraction_ignores_out_of_universe_uppercase_words():
    tickers, _ = filters("How did AI demand affect NVDA and AAPL GAAP margins per the CEO?")
    assert tickers == ["NVDA"]


def test_company_names_map_to_tickers():
    tickers, _ = filters("Compare Home Depot and Lowe's inventory with Goldman Sachs")
    assert tickers == ["HD", "LOW", "GS"]


def test_target_only_matches_the_capitalised_company():
    assert filters("Did the bank hit its inflation target?")[0] == []
    assert filters("What was Target's comparable sales growth?")[0] == ["TGT"]


def test_fiscal_year_forms():
    assert filters("AMD FY2024 vs FY23 and fiscal 2025")[1] == [2023, 2024, 2025]
    assert filters("What is Walmart's strategy?")[1] == []


def test_tokenizer_keeps_figures_and_identifiers_intact():
    tokens = tokenize("Revenue was $5,872,000 under ASC 606, per the 10-K; 2023, 2024.")
    assert tokens == [
        "revenue",
        "was",
        "5872000",
        "under",
        "asc",
        "606",
        "per",
        "the",
        "10-k",
        "2023",
        "2024",
    ]
