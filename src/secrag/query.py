import re

from secrag.config import TICKERS

# Regex rather than an LLM call: a round trip on every query to solve what regex
# solves for most queries is pure latency and cost.
UPPERCASE_WORD = re.compile(r"\b[A-Z]{2,5}\b")
NAMES = [
    (re.compile(rf"\b(?:{pattern})\b", re.I), ticker)
    for pattern, ticker in [
        ("advanced micro devices", "AMD"),
        ("nvidia", "NVDA"),
        ("intel", "INTC"),
        ("broadcom", "AVGO"),
        ("texas instruments", "TXN"),
        ("wal-?mart", "WMT"),
        ("costco", "COST"),
        ("home depot", "HD"),
        ("lowe['’]?s", "LOW"),
        ("jp ?morgan", "JPM"),
        ("bank of america", "BAC"),
        ("wells fargo", "WFC"),
        ("goldman sachs|goldman", "GS"),
        ("morgan stanley", "MS"),
    ]
]
# "target" is an ordinary word ("inflation target"), so only the capitalised form counts.
NAMES.append((re.compile(r"\bTarget\b"), "TGT"))

FY = re.compile(r"\bFY\s?'?(\d{4}|\d{2})\b", re.I)
YEAR = re.compile(r"\b(20\d{2})\b")

THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}\b)")
# Keeps internal punctuation so "10-K", "25.8" and "lowe's" stay single tokens.
TOKEN = re.compile(r"\w+(?:[-.'/&]\w+)*")


def filters(query: str) -> tuple[list[str], list[int]]:
    found = {w for w in UPPERCASE_WORD.findall(query) if w in TICKERS}
    found |= {ticker for pattern, ticker in NAMES if pattern.search(query)}
    years = {int(y) if len(y) == 4 else 2000 + int(y) for y in FY.findall(query)}
    years |= {int(y) for y in YEAR.findall(query)}
    return [t for t in TICKERS if t in found], sorted(years)


def tokenize(text: str) -> list[str]:
    # Exact figures ("5,872") and identifiers ("ASC 606", "10-K") are the reason sparse
    # retrieval is in the pipeline, so they must survive as tokens that match the query.
    return TOKEN.findall(THOUSANDS_SEP.sub("", text.lower()))
