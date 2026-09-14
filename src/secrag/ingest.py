import json
import logging
import os
import sys
from dataclasses import asdict

from secrag.config import DATA_DIR, FISCAL_YEARS_PER_TICKER, TICKERS
from secrag.edgar import Edgar
from secrag.parse import ITEM_TITLES, fiscal_year_focus, html_to_text, split_items

SECTIONS_DIR = DATA_DIR / "sections"
log = logging.getLogger("secrag.ingest")


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        sys.exit("SEC_USER_AGENT is not set, e.g. 'Jane Doe jane@example.com'. EDGAR requires it")

    edgar = Edgar(user_agent)
    ciks = edgar.ciks()
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ticker in TICKERS:
        for filing in edgar.ten_ks(ticker, ciks[ticker], FISCAL_YEARS_PER_TICKER):
            html = decode(edgar.get(filing.url))
            # The period-of-report year can't name the fiscal year on its own: WMT and LOW
            # both closed a year on 2025-01-31, and WMT calls it FY2025 while LOW calls it
            # FY2024. The filing's iXBRL tag carries the company's own label.
            fiscal_year = fiscal_year_focus(html) or int(filing.period_of_report[:4])
            text = html_to_text(html)
            exhibit_url = edgar.annual_report_url(filing)
            if exhibit_url:
                text += "\n\n" + html_to_text(decode(edgar.get(exhibit_url)))
            sections = split_items(text)
            doc_id = f"{ticker}_FY{fiscal_year}"

            records = [
                {"doc_id": doc_id, "ticker": ticker, "fiscal_year": fiscal_year, **asdict(s)}
                for s in sections
            ]
            (SECTIONS_DIR / f"{doc_id}.json").write_text(json.dumps(records, indent=1))

            found = [s.item for s in sections]
            missing = [i for i in ITEM_TITLES if i not in found]
            log.info(
                "%s period=%s chars=%d missing=%s",
                doc_id,
                filing.period_of_report,
                sum(len(s.text) for s in sections),
                missing or "-",
            )
            manifest.append(
                {
                    "doc_id": doc_id,
                    "fiscal_year": fiscal_year,
                    **asdict(filing),
                    "url": filing.url,
                    "exhibit_13_url": exhibit_url,
                    "items": found,
                    "missing_items": missing,
                }
            )

    doc_ids = [m["doc_id"] for m in manifest]
    if len(set(doc_ids)) != len(doc_ids):
        sys.exit(f"duplicate doc ids, fiscal year labelling is wrong: {sorted(doc_ids)}")
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %d filings to %s", len(manifest), DATA_DIR / "manifest.json")


if __name__ == "__main__":
    main()
