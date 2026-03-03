"""
pull_financials.py — Download 10-K filings from SEC EDGAR and extract financials.

Uses only `requests` — no third-party EDGAR libraries.

Strategy: Convert 10-K HTML to plain text, then regex for financial values
near their labels. This is more robust than table parsing for pre-2009
filings which have wildly inconsistent HTML table structures.

Usage:
    python pull_financials.py

Output:
    data/raw/html/{ticker}_{year}.html     (cached filing HTML)
    data/raw/submissions/{cik}.json        (cached submission history)
    data/processed/financials.csv          (extracted values)
"""

import re
import json
import time
import logging
import requests
import pandas as pd
import numpy as np

from config import (
    FIRMS, USER_AGENT, RAW_DIR, PROCESSED_DIR,
    START_YEAR, END_YEAR, RATE_LIMIT_SEC,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HTML_CACHE = RAW_DIR / "html"
SUB_CACHE = RAW_DIR / "submissions"
HTML_CACHE.mkdir(parents=True, exist_ok=True)
SUB_CACHE.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


# ──────────────────────────────────────────────
# 1. LOCATE 10-K FILINGS VIA SUBMISSIONS API
# ──────────────────────────────────────────────

def get_submissions(cik: str) -> dict:
    """Fetch a company's full filing history from data.sec.gov."""
    cache_path = SUB_CACHE / f"{cik}.json"
    if cache_path.exists():
        logger.info(f"  Submissions cache hit: {cik}")
        return json.loads(cache_path.read_text())

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    logger.info(f"  Fetching submissions: {url}")
    time.sleep(RATE_LIMIT_SEC)
    resp = SESSION.get(url)
    resp.raise_for_status()

    data = resp.json()
    cache_path.write_text(json.dumps(data, indent=2))
    return data


def get_all_filing_records(submissions: dict, cik: str) -> list[dict]:
    """Combine 'recent' filings with paginated older filing files."""
    recent = submissions.get("filings", {}).get("recent", {})
    all_records = _records_from_block(recent)

    extra_files = submissions.get("filings", {}).get("files", [])
    for file_info in extra_files:
        filename = file_info.get("name", "")
        if not filename:
            continue

        cache_path = SUB_CACHE / f"{cik}_{filename}"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
        else:
            url = f"https://data.sec.gov/submissions/{filename}"
            logger.info(f"    Fetching older filings: {url}")
            time.sleep(RATE_LIMIT_SEC)
            resp = SESSION.get(url)
            if resp.status_code != 200:
                logger.warning(f"    Failed to fetch {filename} ({resp.status_code})")
                continue
            data = resp.json()
            cache_path.write_text(json.dumps(data, indent=2))

        all_records.extend(_records_from_block(data))

    return all_records


def _records_from_block(block: dict) -> list[dict]:
    forms = block.get("form", [])
    dates = block.get("filingDate", [])
    accessions = block.get("accessionNumber", [])
    primary_docs = block.get("primaryDocument", [])
    report_dates = block.get("reportDate", [])

    records = []
    for i in range(len(forms)):
        records.append({
            "form": forms[i] if i < len(forms) else "",
            "filingDate": dates[i] if i < len(dates) else "",
            "accessionNumber": accessions[i] if i < len(accessions) else "",
            "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
            "reportDate": report_dates[i] if i < len(report_dates) else "",
        })
    return records


def find_10k_filings(submissions: dict, cik: str, start_year: int, end_year: int) -> list[dict]:
    all_records = get_all_filing_records(submissions, cik)
    logger.info(f"  Total filing records: {len(all_records)}")

    results = []
    for rec in all_records:
        if rec["form"] != "10-K":
            continue

        filing_date = rec["filingDate"]
        if not filing_date:
            continue

        report_date = rec["reportDate"]
        if report_date and len(report_date) >= 4:
            fiscal_year = int(report_date[:4])
        else:
            filing_year = int(filing_date[:4])
            filing_month = int(filing_date[5:7])
            fiscal_year = filing_year - 1 if filing_month <= 4 else filing_year

        if fiscal_year < start_year or fiscal_year > end_year:
            continue

        results.append({
            "accession": rec["accessionNumber"],
            "filing_date": filing_date,
            "primary_doc": rec["primaryDocument"],
            "fiscal_year": fiscal_year,
        })

    return results


# ──────────────────────────────────────────────
# 2. DOWNLOAD 10-K HTML
# ──────────────────────────────────────────────

def download_filing_html(cik: str, ticker: str, filing: dict) -> str | None:
    fiscal_year = filing["fiscal_year"]
    cache_path = HTML_CACHE / f"{ticker}_{fiscal_year}.html"

    if cache_path.exists():
        logger.info(f"    HTML cache hit: {cache_path.name}")
        return cache_path.read_text(encoding="utf-8", errors="replace")

    accession_no_dashes = filing["accession"].replace("-", "")
    cik_nodash = cik.lstrip("0") or "0"
    primary_doc = filing["primary_doc"]

    urls = [
        f"https://www.sec.gov/Archives/edgar/data/{cik_nodash}/{accession_no_dashes}/{primary_doc}",
        f"https://data.sec.gov/Archives/edgar/data/{cik_nodash}/{accession_no_dashes}/{primary_doc}",
    ]

    for url in urls:
        logger.info(f"    Trying: {url}")
        time.sleep(RATE_LIMIT_SEC)

        try:
            resp = SESSION.get(url, timeout=60)
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            logger.warning(f"    Connection error: {type(e).__name__}, trying next URL...")
            continue

        if resp.status_code == 200:
            html = resp.text
            cache_path.write_text(html, encoding="utf-8")
            logger.info(f"    Saved: {cache_path.name} ({len(html):,} chars)")
            return html
        elif resp.status_code in (503, 500, 403):
            logger.warning(f"    {resp.status_code} from {url.split('/')[2]}, trying next URL...")
            continue
        else:
            logger.warning(f"    Failed ({resp.status_code}): {url}")
            continue

    logger.error(f"    All URLs failed for {ticker} FY{fiscal_year}")
    return None


# ──────────────────────────────────────────────
# 3. HTML → PLAIN TEXT CONVERSION
# ──────────────────────────────────────────────

def html_to_text(raw: str) -> str:
    """Convert raw HTML to plain text, preserving line structure."""
    text = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|tr|li|h\d|td)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"&amp;", "&", text, flags=re.IGNORECASE)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


# ──────────────────────────────────────────────
# 4. TEXT-BASED FINANCIAL VALUE EXTRACTION
# ──────────────────────────────────────────────

# Each field: list of (label_regex, value_position) tuples
# value_position: "after" means value follows the label on the same or next line
TARGET_FIELDS = {
    "total_assets": [
        r"total\s+assets",
    ],
    "total_liabilities": [
        r"total\s+liabilities\b(?!\s+and)",  # avoid "total liabilities and equity"
    ],
    "stockholders_equity": [
        r"total\s+(?:stockholders|shareholders)['\u2019]?\s*equity",
        r"total\s+equity",
    ],
    "cash_and_equivalents": [
        r"cash\s+and\s+(?:cash\s+)?equivalents",
        r"cash\s+and\s+due\s+from\s+banks",
    ],
    "short_term_borrowings": [
        r"short[\s-]*term\s+borrowings",
        r"commercial\s+paper\s+and\s+other\s+short[\s-]*term\s+borrowings",
    ],
    "net_income": [
        r"net\s+income\s*(?:\(loss\))?\s*$",
        r"net\s+income\s*(?:\(loss\))?\s+attributable",
        r"net\s+(?:income|loss)\s*$",
    ],
    "provision_for_loan_losses": [
        r"provision\s+for\s+(?:credit|loan)\s+loss",
        r"provision\s+for\s+loan[\s,]+lease",
    ],
}


def extract_number_near_label(text: str, label_pattern: str) -> list[dict]:
    """
    Find all occurrences of a label pattern in the text, then extract
    the nearest dollar-amount numbers.

    Returns a list of {value, context, position} dicts, one per match.
    """
    results = []
    number_pattern = re.compile(
        r"[\($]?\s*\$?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*\)?",
    )

    for match in re.finditer(label_pattern, text, re.IGNORECASE | re.MULTILINE):
        # Look at the text within ~500 chars after the label match
        search_window = text[match.end():match.end() + 500]

        # Find all numbers in the window
        numbers = []
        for num_match in number_pattern.finditer(search_window):
            raw = num_match.group(0)
            num_str = num_match.group(1).replace(",", "")

            try:
                value = float(num_str)
            except ValueError:
                continue

            # Skip tiny numbers (percentages, footnote refs)
            if value < 10:
                continue

            # Detect negative (parentheses)
            is_negative = "(" in raw and ")" in raw
            if is_negative:
                value = -value

            numbers.append({
                "value": value,
                "raw": raw.strip(),
                "distance": num_match.start(),
            })

        if numbers:
            # Take the closest number to the label
            best = min(numbers, key=lambda x: x["distance"])
            context = text[match.start():match.end() + 80].replace("\n", " ").strip()
            results.append({
                "value": best["value"],
                "raw": best["raw"],
                "context": context[:100],
                "position": match.start(),
            })

    return results


def detect_unit_multiplier(text: str) -> int:
    """Detect if filing reports in millions, thousands, or raw dollars."""
    # Search in the first ~5000 chars where headers usually are
    header_text = text[:10000].lower()

    patterns = [
        (r"in\s+millions", 1_000_000),
        (r"\(?\s*in\s+millions\s*\)?", 1_000_000),
        (r"dollars\s+in\s+millions", 1_000_000),
        (r"amounts?\s+in\s+millions", 1_000_000),
        (r"in\s+thousands", 1_000),
        (r"\(?\s*in\s+thousands\s*\)?", 1_000),
        (r"in\s+billions", 1_000_000_000),
    ]

    for pattern, multiplier in patterns:
        if re.search(pattern, header_text):
            return multiplier

    # Also search around "consolidated balance sheet" or "financial statements"
    for pattern, multiplier in patterns:
        if re.search(pattern, text[:50000].lower()):
            return multiplier

    return 1


def scrape_filing(html: str, ticker: str, fiscal_year: int) -> dict:
    """Extract all target financial fields from a 10-K filing."""
    result = {"ticker": ticker, "fiscal_year": fiscal_year}

    text = html_to_text(html)
    multiplier = detect_unit_multiplier(text)
    logger.info(f"    Unit multiplier: {multiplier:,}x")
    logger.info(f"    Text length: {len(text):,} chars")

    for field, patterns in TARGET_FIELDS.items():
        best_match = None

        for pattern in patterns:
            matches = extract_number_near_label(text, pattern)
            if matches:
                # If multiple matches (e.g. "Total assets" appears in multiple tables),
                # prefer the one with the largest absolute value — that's likely the
                # consolidated balance sheet, not a subsidiary or segment table.
                candidate = max(matches, key=lambda x: abs(x["value"]))

                if best_match is None or abs(candidate["value"]) > abs(best_match["value"]):
                    best_match = candidate

        if best_match:
            final_value = best_match["value"] * multiplier
            result[field] = final_value
            logger.info(f"    ✓ {field}: {best_match['raw']} × {multiplier:,} = {final_value:,.0f}")
            logger.info(f"      Context: \"{best_match['context']}\"")
        else:
            result[field] = np.nan
            logger.warning(f"    ✗ {field}: NOT FOUND")

    return result


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────

def main():
    rows = []

    for firm_name, info in FIRMS.items():
        cik = info["cik"]
        ticker = info["ticker"]
        logger.info(f"\n{'='*60}\n{firm_name} ({ticker}, CIK {cik})\n{'='*60}")

        submissions = get_submissions(cik)
        filings = find_10k_filings(submissions, cik, START_YEAR, END_YEAR)
        logger.info(f"  Found {len(filings)} 10-K filings in range")

        for filing in filings:
            fy = filing["fiscal_year"]
            logger.info(f"\n  --- FY{fy} (filed {filing['filing_date']}) ---")

            html = download_filing_html(cik, ticker, filing)
            if not html:
                rows.append({"firm": firm_name, "ticker": ticker, "status": info["status"], "fiscal_year": fy})
                continue

            result = scrape_filing(html, ticker, fy)
            result["firm"] = firm_name
            result["status"] = info["status"]
            result["filing_date"] = filing["filing_date"]
            result["accession"] = filing["accession"]
            rows.append(result)

    df = pd.DataFrame(rows)

    # Reorder columns
    id_cols = ["firm", "ticker", "status", "fiscal_year", "filing_date", "accession"]
    data_cols = list(TARGET_FIELDS.keys())
    df = df[id_cols + [c for c in data_cols if c in df.columns]]

    out_path = PROCESSED_DIR / "financials.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved {len(df)} rows → {out_path}")

    # Coverage report
    logger.info(f"\n{'='*60}\nCOVERAGE REPORT\n{'='*60}")
    for firm in df["firm"].unique():
        fdf = df[df["firm"] == firm]
        years = sorted(fdf["fiscal_year"].tolist())
        logger.info(f"\n  {firm} (FY{min(years)}-{max(years)}):")
        for col in data_cols:
            if col in fdf.columns:
                found = fdf[col].notna().sum()
                total = len(fdf)
                icon = "✓" if found == total else "△" if found > 0 else "✗"
                logger.info(f"    {icon} {col}: {found}/{total}")

    # Flag gaps
    existing_data_cols = [c for c in data_cols if c in df.columns]
    if existing_data_cols:
        gaps = df[existing_data_cols].isna().sum(axis=1)
        if gaps.any():
            logger.info(f"\n{'='*60}\nMANUAL REVIEW NEEDED\n{'='*60}")
            for idx, row in df[gaps > 0].iterrows():
                missing = [c for c in existing_data_cols if pd.isna(row.get(c))]
                logger.info(f"  {row['firm']} FY{row['fiscal_year']}: missing {missing}")


if __name__ == "__main__":
    main()