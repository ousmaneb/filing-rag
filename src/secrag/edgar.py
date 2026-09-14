import json
import time
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from secrag.config import DATA_DIR

CACHE_DIR = DATA_DIR / "edgar"
SUBMISSIONS = "https://data.sec.gov/submissions/"
# EDGAR allows 10 req/s and blocks the IP for a while when that is exceeded.
MIN_INTERVAL = 1 / 6


@dataclass
class Filing:
    ticker: str
    cik: int
    accession: str
    filing_date: str
    period_of_report: str
    primary_document: str

    @property
    def folder(self) -> str:
        return (
            f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accession.replace('-', '')}"
        )

    @property
    def url(self) -> str:
        return f"{self.folder}/{self.primary_document}"


class Edgar:
    def __init__(self, user_agent: str):
        self.client = httpx.Client(
            headers={"User-Agent": user_agent}, timeout=60, follow_redirects=True
        )
        self.last_request = 0.0

    def get(self, url: str) -> bytes:
        path = CACHE_DIR / url.split("://", 1)[1]
        if path.exists():
            return path.read_bytes()
        wait = self.last_request + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        resp = self.client.get(url)
        self.last_request = time.monotonic()
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename, so an interrupted run never leaves a truncated file in the cache.
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
        return resp.content

    def ciks(self) -> dict[str, int]:
        rows = json.loads(self.get("https://www.sec.gov/files/company_tickers.json")).values()
        return {r["ticker"]: r["cik_str"] for r in rows}

    def annual_report_url(self, filing: Filing) -> str | None:
        # Wells Fargo files a thin 10-K and puts MD&A and the financial statements in
        # Exhibit 13. The exhibit type is only listed on the filing's index page.
        index = self.get(f"{filing.folder}/{filing.accession}-index.htm")
        for tr in BeautifulSoup(index, "lxml").select("table.tableFile tr"):
            cells = tr.find_all("td")
            if len(cells) == 5 and cells[3].get_text(strip=True).startswith("EX-13"):
                # The link text is "name.htm iXBRL"; the href points at the iXBRL viewer.
                return f"{filing.folder}/{cells[2].get_text(' ', strip=True).split()[0]}"
        return None

    def ten_ks(self, ticker: str, cik: int, n: int) -> list[Filing]:
        subs = json.loads(self.get(f"{SUBMISSIONS}CIK{cik:010d}.json"))
        page = subs["filings"]["recent"]
        # "recent" holds only the last 1000 filings. GS and MS file thousands of
        # prospectus supplements a year, so their 10-Ks can be in the older pages.
        older = iter(f["name"] for f in subs["filings"]["files"])
        found = []
        while True:
            columns = ("accessionNumber", "filingDate", "reportDate", "primaryDocument")
            for form, *row in zip(page["form"], *(page[c] for c in columns), strict=True):
                # Exact match: 10-K/A amendments are usually just Part III.
                if form == "10-K":
                    found.append(Filing(ticker, cik, *row))
            name = next(older, None)
            if len(found) >= n or name is None:
                break
            page = json.loads(self.get(SUBMISSIONS + name))
        found.sort(key=lambda f: f.period_of_report, reverse=True)
        return found[:n]
