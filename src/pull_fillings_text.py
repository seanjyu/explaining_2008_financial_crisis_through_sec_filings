"""
pull_filings_text.py — Extract Risk Factors (Item 1A) and MD&A (Item 7)
sections from locally cached 10-K HTML files.

Run pull_financials.py first — it downloads the HTML files to data/raw/html/.

Output:
    data/texts/{ticker}/{year}_risk_factors.txt
    data/texts/{ticker}/{year}_mda.txt
"""

import re
import logging
from config import FIRMS, TEXTS_DIR, RAW_DIR, START_YEAR, END_YEAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HTML_CACHE = RAW_DIR / "html"


# ──────────────────────────────────────────────
# 1. Text cleaning — do this BEFORE section search
# ──────────────────────────────────────────────

def html_to_text(raw: str) -> str:
    """Convert raw HTML to searchable plain text."""
    text = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)    # preserve line breaks
    text = re.sub(r"</(p|div|tr|li|h\d)>", "\n", text, flags=re.IGNORECASE)  # block elements
    text = re.sub(r"<[^>]+>", " ", text)           # strip remaining tags
    text = re.sub(r"&nbsp;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"&amp;", "&", text, flags=re.IGNORECASE)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)      # other HTML entities
    text = re.sub(r"&#\d+;", " ", text)            # numeric entities
    text = re.sub(r"\xa0", " ", text)              # non-breaking spaces
    text = re.sub(r"[ \t]+", " ", text)            # collapse horizontal whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)        # collapse blank lines
    return text.strip()


def clean_section_text(text: str) -> str:
    """Final cleanup for an extracted section."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# 2. Section extraction — multiple strategies
# ──────────────────────────────────────────────

# Each section has multiple start/end pattern pairs to try, from most specific to broadest
SECTION_CONFIGS = {
    "risk_factors": {
        "start_patterns": [
            r"item\s+1a[\.\s\-—:]+risk\s+factors",
            r"item\s*1a\b.*?risk\s+factors",
            r"risk\s+factors\s*\n",
        ],
        "end_patterns": [
            r"item\s+1b[\.\s\-—:]",
            r"item\s+2[\.\s\-—:]+prop",
            r"item\s+2[\.\s\-—:]",
        ],
    },
    "mda": {
        "start_patterns": [
            r"item\s+7[\.\s\-—:]+management",
            r"item\s*7\b.*?management.*?discussion",
            r"management\s*['\u2019]?\s*s?\s+discussion\s+and\s+analysis",
        ],
        "end_patterns": [
            r"item\s+7a[\.\s\-—:]",
            r"item\s+8[\.\s\-—:]",
            r"item\s+8\b",
        ],
    },
}


def extract_section(text: str, section_name: str) -> str | None:
    """
    Try multiple pattern combinations to find a section.
    Searches the plain text (not raw HTML).
    Uses the LAST match of start pattern to skip table of contents entries.
    """
    config = SECTION_CONFIGS.get(section_name)
    if not config:
        return None

    for start_pat in config["start_patterns"]:
        starts = list(re.finditer(start_pat, text, re.IGNORECASE))
        if not starts:
            continue

        # Use the last match to skip TOC entries
        start_match = starts[-1]
        logger.info(f"      Start matched: '{start_pat}' at position {start_match.start()}")

        # Try each end pattern
        for end_pat in config["end_patterns"]:
            end_match = re.search(end_pat, text[start_match.end():], re.IGNORECASE)
            if end_match:
                section = text[start_match.start():start_match.end() + end_match.start()]
                section = clean_section_text(section)

                if len(section) > 500:
                    logger.info(f"      End matched: '{end_pat}', extracted {len(section):,} chars")
                    return section
                else:
                    logger.info(f"      End matched but section too short ({len(section)} chars), trying next...")

        # No end pattern worked — take a generous chunk
        section = text[start_match.start():start_match.start() + 200_000]
        section = clean_section_text(section)

        if len(section) > 500:
            logger.info(f"      No end boundary found, took {len(section):,} chars from start")
            return section

    return None


# ──────────────────────────────────────────────
# 3. Main
# ──────────────────────────────────────────────

def main():
    sections = ["risk_factors", "mda"]
    summary = []

    for firm_name, info in FIRMS.items():
        ticker = info["ticker"]
        logger.info(f"\n{'='*50}\n{firm_name} ({ticker})\n{'='*50}")

        firm_dir = TEXTS_DIR / ticker
        firm_dir.mkdir(parents=True, exist_ok=True)

        for year in range(START_YEAR, END_YEAR + 1):
            html_path = HTML_CACHE / f"{ticker}_{year}.html"

            if not html_path.exists():
                logger.warning(f"  FY{year}: No HTML file found ({html_path.name})")
                for section in sections:
                    summary.append({"firm": firm_name, "year": year, "section": section, "status": "no_html"})
                continue

            logger.info(f"  FY{year}: Reading {html_path.name} ({html_path.stat().st_size:,} bytes)")
            raw = html_path.read_text(encoding="utf-8", errors="replace")

            # Convert HTML to plain text ONCE, then search for both sections
            text = html_to_text(raw)
            logger.info(f"    Plain text length: {len(text):,} chars")

            for section in sections:
                out_path = firm_dir / f"{year}_{section}.txt"

                # Check cache
                if out_path.exists() and out_path.stat().st_size > 500:
                    logger.info(f"    {section}: cached ({out_path.stat().st_size:,} bytes)")
                    summary.append({"firm": firm_name, "year": year, "section": section, "status": "cached"})
                    continue

                logger.info(f"    Extracting {section}...")
                extracted = extract_section(text, section)

                if extracted:
                    out_path.write_text(extracted, encoding="utf-8")
                    logger.info(f"    ✓ {section}: {len(extracted):,} chars")
                    summary.append({"firm": firm_name, "year": year, "section": section, "status": "ok"})
                else:
                    # Log what headers we CAN see to help debug
                    item_headers = re.findall(r"item\s+\d+[a-z]?[\.\s\-—:]+\w+", text[:50000], re.IGNORECASE)
                    unique_headers = list(dict.fromkeys(item_headers[:15]))  # dedupe, keep order
                    logger.warning(f"    ✗ {section}: NOT FOUND")
                    logger.warning(f"      Headers found in filing: {unique_headers[:8]}")
                    summary.append({"firm": firm_name, "year": year, "section": section, "status": "failed"})

    # Print summary
    logger.info(f"\n{'='*50}\nExtraction Summary\n{'='*50}")
    ok = sum(1 for e in summary if e["status"] in ("ok", "cached"))
    failed = sum(1 for e in summary if e["status"] == "failed")
    no_html = sum(1 for e in summary if e["status"] == "no_html")

    for entry in summary:
        icon = "✓" if entry["status"] in ("ok", "cached") else "—" if entry["status"] == "no_html" else "✗"
        logger.info(f"  {icon} {entry['firm']} FY{entry['year']} {entry['section']}: {entry['status']}")

    logger.info(f"\n  Total: {ok} extracted, {failed} failed, {no_html} missing HTML")

    if failed:
        logger.warning(f"\n  {failed} extraction(s) failed.")
        logger.warning("  Check the 'Headers found in filing' log lines above to see")
        logger.warning("  what section headers exist — you may need to add patterns.")


if __name__ == "__main__":
    main()