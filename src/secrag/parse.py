import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

# iXBRL filings open with an XML declaration but are HTML documents.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
}

# Every item is a boundary, including ones we don't keep, or Item 4 text would bleed
# into Item 3. The lookahead stops "Item 1" from matching "Item 1A" or "Item 10".
# The optional pipe catches headings laid out in table cells.
HEADING = re.compile(
    r"^(?:\|\s*)?items?\s+(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16)(?![0-9a-z])",
    re.I | re.M,
)
MIN_WORDS = 50
BLOCKS = ["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"]
PAGE_JUNK = re.compile(r"\d{1,3}|page \d{1,3}|table of contents", re.I)
FY_FOCUS = re.compile(r'name="dei:DocumentFiscalYearFocus"[^>]*>(?:\s*<[^>]+>)*\s*(\d{4})')


@dataclass
class Section:
    item: str
    title: str
    text: str


def fiscal_year_focus(html: str) -> int | None:
    m = FY_FOCUS.search(html)
    return int(m.group(1)) if m else None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for el in soup.find_all(["script", "style"]):
        el.decompose()
    # The inline XBRL header sits in a hidden div and is full of tagged facts.
    for el in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        el.decompose()
    # Convert tables before flattening. As prose, each number loses its row label.
    # Reversed so nested tables are converted before the table that holds them.
    for table in reversed(soup.find_all("table")):
        table.replace_with(f"\n\n{table_to_markdown(table)}\n\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for el in soup.find_all(BLOCKS):
        el.append("\n\n")

    paragraphs = []
    for block in re.split(r"\n\s*\n", soup.get_text().replace("​", "")):
        block = block.strip()
        if block.startswith("|"):
            paragraphs.append(block)
            continue
        para = " ".join(block.split())
        if para and not PAGE_JUNK.fullmatch(para):
            paragraphs.append(para)
    return "\n\n".join(paragraphs)


def table_to_markdown(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        # EDGAR puts "$", "(" and ")" in their own cells; glue them back onto the number.
        for td in tr.find_all(["td", "th"]):
            cell = " ".join(td.get_text(" ").split())
            if not cell:
                continue
            if cells and cell in {")", "%", ")%"}:
                cells[-1] += cell
            elif cells and cells[-1] in {"$", "(", "$("}:
                cells[-1] += cell
            else:
                cells.append(cell)
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    # Layout tables used to position a heading or a paragraph are not data.
    if all(len(r) == 1 for r in rows):
        return "\n\n".join(r[0] for r in rows)
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "|" + " --- |" * max(len(r) for r in rows))
    return "\n".join(lines)


def split_items(text: str) -> list[Section]:
    matches = list(HEADING.finditer(text))
    sections, covered = [], []
    for item, title in ITEM_TITLES.items():
        candidates = []
        for i, m in enumerate(matches):
            if m.group(1).upper() != item:
                continue
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body_start = text.find("\n", m.end())
            body = text[body_start:end].strip() if body_start != -1 else ""
            candidates.append((m.start(), end, body))
        if not candidates:
            continue
        # Every heading also appears in the table of contents, where the text after it
        # is a page number. Take the last occurrence with a real body; fall back to the
        # last occurrence for items that are legitimately short ("Item 1B. None.").
        meaningful = [c for c in candidates if len(re.findall(r"\w+", c[2])) >= MIN_WORDS]
        start, end, body = (meaningful or candidates)[-1]
        # A cross-reference index row can leave just a "| --- |" separator as the body.
        if re.search(r"[a-z]", body, re.I):
            sections.append(Section(item, title, body))
            covered.append((start, end))

    # Text outside the kept Items is often the substance of the filing: Intel and Morgan
    # Stanley list Items only in a cross-reference index, and NVDA and JPM point Item 8
    # at pages elsewhere in the document. Keep all of it as one unlabelled section.
    rest, pos = [], 0
    for start, end in sorted(covered):
        rest.append(text[pos:start])
        pos = end
    rest.append(text[pos:])
    other = "\n\n".join(p.strip() for p in rest if p.strip())
    if other:
        sections.append(Section("", "Other", other))
    return sections
