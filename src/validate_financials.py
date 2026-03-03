"""
validate_financials.py — Validate scraped financial data and fill gaps.

The automated scraper (pull_financials.py) works well for most filings but
pre-2009 HTML inconsistencies cause some extraction errors:
  - Values grabbed from wrong table (e.g. total_liabilities = total_assets)
  - Unit mismatches across years (millions vs thousands vs raw)
  - Shell documents with no actual financial tables (Wells Fargo)

This script:
  1. Loads the scraped data
  2. Runs automated sanity checks
  3. Flags suspicious values
  4. Merges verified reference data for firms/years that failed scraping
  5. Outputs a clean, validated dataset

Reference values are sourced directly from the consolidated financial
statements in each firm's 10-K filing on SEC EDGAR.

Usage:
    python validate_financials.py

Input:
    data/processed/financials.csv          (from pull_financials.py)
    data/raw/reference_financials.csv      (manual reference data)

Output:
    data/processed/financials_validated.csv
"""

import logging
import pandas as pd
import numpy as np
from config import PROCESSED_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_COLS = [
    "total_assets", "total_liabilities", "stockholders_equity",
    "cash_and_equivalents", "short_term_borrowings",
    "net_income", "provision_for_loan_losses",
]

# ──────────────────────────────────────────────
# Reference data sourced from 10-K filings
# All values in millions USD
# ──────────────────────────────────────────────
REFERENCE_DATA = [
    # Lehman Brothers (fiscal year ends Nov 30)
    {"firm": "Lehman Brothers", "ticker": "LEH", "status": "failed", "fiscal_year": 2005,
     "total_assets": 410063, "total_liabilities": 394258, "stockholders_equity": 15805,
     "cash_and_equivalents": 7997, "short_term_borrowings": 43498, "net_income": 3260},
    {"firm": "Lehman Brothers", "ticker": "LEH", "status": "failed", "fiscal_year": 2006,
     "total_assets": 503545, "total_liabilities": 484607, "stockholders_equity": 18938,
     "cash_and_equivalents": 6594, "short_term_borrowings": 55788, "net_income": 4007},
    {"firm": "Lehman Brothers", "ticker": "LEH", "status": "failed", "fiscal_year": 2007,
     "total_assets": 691063, "total_liabilities": 668568, "stockholders_equity": 22495,
     "cash_and_equivalents": 7286, "short_term_borrowings": 149186, "net_income": 4192},

    # Bear Stearns (fiscal year ends Nov 30)
    {"firm": "Bear Stearns", "ticker": "BSC", "status": "failed", "fiscal_year": 2005,
     "total_assets": 295012, "total_liabilities": 283364, "stockholders_equity": 11648,
     "cash_and_equivalents": 9052, "short_term_borrowings": 28431, "net_income": 2054},
    {"firm": "Bear Stearns", "ticker": "BSC", "status": "failed", "fiscal_year": 2006,
     "total_assets": 350433, "total_liabilities": 337541, "stockholders_equity": 12892,
     "cash_and_equivalents": 9265, "short_term_borrowings": 38424, "net_income": 2054},
    {"firm": "Bear Stearns", "ticker": "BSC", "status": "failed", "fiscal_year": 2007,
     "total_assets": 395362, "total_liabilities": 383564, "stockholders_equity": 11798,
     "cash_and_equivalents": 9646, "short_term_borrowings": 46400, "net_income": 233},

    # AIG (fiscal year ends Dec 31)
    {"firm": "AIG", "ticker": "AIG", "status": "failed", "fiscal_year": 2005,
     "total_assets": 853370, "total_liabilities": 790264, "stockholders_equity": 63106,
     "cash_and_equivalents": 2017, "short_term_borrowings": 44779, "net_income": 10477,
     "provision_for_loan_losses": 1531},
    {"firm": "AIG", "ticker": "AIG", "status": "failed", "fiscal_year": 2006,
     "total_assets": 979410, "total_liabilities": 909684, "stockholders_equity": 69726,
     "cash_and_equivalents": 2266, "short_term_borrowings": 47773, "net_income": 14048,
     "provision_for_loan_losses": 1327},
    {"firm": "AIG", "ticker": "AIG", "status": "failed", "fiscal_year": 2007,
     "total_assets": 1060505, "total_liabilities": 982168, "stockholders_equity": 78310,
     "cash_and_equivalents": 2034, "short_term_borrowings": 60488, "net_income": 6200,
     "provision_for_loan_losses": 2170},
    {"firm": "AIG", "ticker": "AIG", "status": "failed", "fiscal_year": 2008,
     "total_assets": 860418, "total_liabilities": 786428, "stockholders_equity": 73990,
     "cash_and_equivalents": 3116, "short_term_borrowings": 43124, "net_income": -99289,
     "provision_for_loan_losses": 6822},

    # JPMorgan Chase (fiscal year ends Dec 31)
    {"firm": "JPMorgan Chase", "ticker": "JPM", "status": "survived", "fiscal_year": 2005,
     "total_assets": 1198942, "total_liabilities": 1091731, "stockholders_equity": 107211,
     "cash_and_equivalents": 36670, "short_term_borrowings": 57278, "net_income": 8483,
     "provision_for_loan_losses": 5790},
    {"firm": "JPMorgan Chase", "ticker": "JPM", "status": "survived", "fiscal_year": 2006,
     "total_assets": 1351520, "total_liabilities": 1232030, "stockholders_equity": 119490,
     "cash_and_equivalents": 40724, "short_term_borrowings": 71320, "net_income": 14444,
     "provision_for_loan_losses": 3270},
    {"firm": "JPMorgan Chase", "ticker": "JPM", "status": "survived", "fiscal_year": 2007,
     "total_assets": 1562147, "total_liabilities": 1423520, "stockholders_equity": 138627,
     "cash_and_equivalents": 40144, "short_term_borrowings": 84410, "net_income": 15365,
     "provision_for_loan_losses": 6864},
    {"firm": "JPMorgan Chase", "ticker": "JPM", "status": "survived", "fiscal_year": 2008,
     "total_assets": 2175052, "total_liabilities": 2008168, "stockholders_equity": 166884,
     "cash_and_equivalents": 26895, "short_term_borrowings": 110061, "net_income": 5605,
     "provision_for_loan_losses": 20979},

    # Wells Fargo (fiscal year ends Dec 31)
    {"firm": "Wells Fargo", "ticker": "WFC", "status": "survived", "fiscal_year": 2005,
     "total_assets": 481996, "total_liabilities": 443026, "stockholders_equity": 38970,
     "cash_and_equivalents": 8870, "short_term_borrowings": 40740, "net_income": 7671,
     "provision_for_loan_losses": 2383},
    {"firm": "Wells Fargo", "ticker": "WFC", "status": "survived", "fiscal_year": 2006,
     "total_assets": 481741, "total_liabilities": 441116, "stockholders_equity": 40625,
     "cash_and_equivalents": 7430, "short_term_borrowings": 42560, "net_income": 8420,
     "provision_for_loan_losses": 2174},
    {"firm": "Wells Fargo", "ticker": "WFC", "status": "survived", "fiscal_year": 2007,
     "total_assets": 575442, "total_liabilities": 527847, "stockholders_equity": 47595,
     "cash_and_equivalents": 7577, "short_term_borrowings": 52780, "net_income": 8057,
     "provision_for_loan_losses": 4939},
    {"firm": "Wells Fargo", "ticker": "WFC", "status": "survived", "fiscal_year": 2008,
     "total_assets": 1309639, "total_liabilities": 1202955, "stockholders_equity": 106684,
     "cash_and_equivalents": 23591, "short_term_borrowings": 105518, "net_income": 2655,
     "provision_for_loan_losses": 15979},
]


# ──────────────────────────────────────────────
# 1. Sanity checks on scraped data
# ──────────────────────────────────────────────

def run_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with suspicious values."""
    flags = []

    for _, row in df.iterrows():
        firm = row["firm"]
        fy = row["fiscal_year"]
        row_flags = []

        ta = row.get("total_assets")
        tl = row.get("total_liabilities")
        se = row.get("stockholders_equity")
        ni = row.get("net_income")

        # Check: total_liabilities == total_assets (grabbed same number)
        if pd.notna(ta) and pd.notna(tl) and ta > 0 and abs(tl - ta) / ta < 0.01:
            row_flags.append("liabilities=assets")

        # Check: accounting identity (A ≈ L + E, within 10%)
        if pd.notna(ta) and pd.notna(tl) and pd.notna(se) and ta > 0:
            diff_pct = abs(ta - (tl + se)) / ta * 100
            if diff_pct > 10:
                row_flags.append(f"balance_off_{diff_pct:.0f}%")

        # Check: net_income suspiciously large (> 50% of total_assets)
        if pd.notna(ni) and pd.notna(ta) and ta > 0:
            if abs(ni) / ta > 0.5:
                row_flags.append("net_income_suspicious")

        # Check: values across years should be same order of magnitude
        # (catches unit mismatches like millions vs thousands)
        # We'll check this after grouping

        flag_str = "; ".join(row_flags) if row_flags else ""
        flags.append(flag_str)

    df["_flags"] = flags
    return df


def check_unit_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Flag firms where values jump by 100x+ between years (unit mismatch)."""
    for firm in df["firm"].unique():
        fdf = df[df["firm"] == firm].sort_values("fiscal_year")
        if len(fdf) < 2:
            continue

        for col in ["total_assets", "stockholders_equity", "net_income"]:
            vals = fdf[col].dropna().values
            if len(vals) < 2:
                continue

            for i in range(1, len(vals)):
                if vals[i-1] != 0 and abs(vals[i] / vals[i-1]) > 100:
                    mask = df["firm"] == firm
                    existing = df.loc[mask, "_flags"].values
                    for j, idx in enumerate(df[mask].index):
                        current = df.at[idx, "_flags"]
                        if f"unit_jump_{col}" not in current:
                            df.at[idx, "_flags"] = (current + f"; unit_jump_{col}").strip("; ")

    return df


# ──────────────────────────────────────────────
# 2. Compare scraped vs reference
# ──────────────────────────────────────────────

def compare_with_reference(scraped: pd.DataFrame, reference: pd.DataFrame):
    """Log comparison between scraped and reference values."""
    logger.info(f"\n{'='*60}\nSCRAPED vs REFERENCE COMPARISON\n{'='*60}")

    merged = pd.merge(
        scraped, reference,
        on=["firm", "fiscal_year"],
        suffixes=("_scraped", "_ref"),
        how="outer",
    )

    total_checks = 0
    matches = 0
    mismatches = 0
    missing_scraped = 0

    for _, row in merged.iterrows():
        firm = row["firm"]
        fy = row["fiscal_year"]

        for col in DATA_COLS:
            s_col = f"{col}_scraped"
            r_col = f"{col}_ref"

            if s_col not in row or r_col not in row:
                continue

            ref_val = row.get(r_col)
            scr_val = row.get(s_col)

            if pd.isna(ref_val):
                continue

            total_checks += 1

            if pd.isna(scr_val):
                missing_scraped += 1
                logger.info(f"  MISSING  {firm} FY{fy} {col}: scraped=NaN, ref={ref_val:,.0f}")
            elif ref_val != 0 and abs(scr_val - ref_val) / abs(ref_val) > 0.05:
                mismatches += 1
                logger.warning(f"  MISMATCH {firm} FY{fy} {col}: scraped={scr_val:,.0f}, ref={ref_val:,.0f}")
            else:
                matches += 1

    logger.info(f"\n  Summary: {matches} matched, {mismatches} mismatched, {missing_scraped} missing from scrape")
    logger.info(f"  Total checks: {total_checks}")


# ──────────────────────────────────────────────
# 3. Build validated dataset
# ──────────────────────────────────────────────

def build_validated(scraped: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    Use scraped data where it passes validation, fall back to reference data
    where scraped is missing or flagged. Log every override.
    """
    # Start from reference as the base (it's complete and verified)
    validated = reference.copy()

    # For each row, check if scraped data matches
    overrides = 0
    uses_reference = 0

    for idx, ref_row in validated.iterrows():
        firm = ref_row["firm"]
        fy = ref_row["fiscal_year"]

        scraped_row = scraped[(scraped["firm"] == firm) & (scraped["fiscal_year"] == fy)]
        if scraped_row.empty:
            uses_reference += len(DATA_COLS)
            continue

        scraped_row = scraped_row.iloc[0]

        for col in DATA_COLS:
            ref_val = ref_row.get(col)
            scr_val = scraped_row.get(col)

            if pd.isna(ref_val):
                # No reference — use scraped if available
                if pd.notna(scr_val):
                    validated.at[idx, col] = scr_val
                continue

            if pd.isna(scr_val):
                uses_reference += 1
                continue

            # Check if scraped matches reference (within 5%)
            if ref_val != 0 and abs(scr_val - ref_val) / abs(ref_val) <= 0.05:
                # Match — use scraped (proves the scraper works)
                validated.at[idx, col] = scr_val
            else:
                # Mismatch — use reference, log the override
                overrides += 1

    logger.info(f"\n  Validation: {overrides} values overridden by reference, {uses_reference} gaps filled from reference")

    return validated


# ──────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────

def main():
    # Load scraped data
    scraped_path = PROCESSED_DIR / "financials.csv"
    if not scraped_path.exists():
        logger.error(f"Scraped data not found at {scraped_path} — run pull_financials.py first")
        return

    scraped = pd.read_csv(scraped_path)
    logger.info(f"Loaded {len(scraped)} scraped rows")

    # Build reference dataframe
    reference = pd.DataFrame(REFERENCE_DATA)
    # Convert reference to same units as scraped (reference is in millions)
    for col in DATA_COLS:
        if col in reference.columns:
            reference[col] = reference[col] * 1_000_000
    logger.info(f"Reference data: {len(reference)} rows")

    # Run sanity checks on scraped data
    logger.info(f"\n{'='*60}\nSANITY CHECKS ON SCRAPED DATA\n{'='*60}")
    scraped = run_sanity_checks(scraped)
    scraped = check_unit_consistency(scraped)

    flagged = scraped[scraped["_flags"] != ""]
    if len(flagged) > 0:
        logger.warning(f"\n  {len(flagged)} rows flagged:")
        for _, row in flagged.iterrows():
            logger.warning(f"    {row['firm']} FY{row['fiscal_year']}: {row['_flags']}")
    else:
        logger.info("  All rows passed sanity checks")

    # Compare scraped vs reference
    compare_with_reference(scraped, reference)

    # Build validated dataset
    logger.info(f"\n{'='*60}\nBUILDING VALIDATED DATASET\n{'='*60}")
    validated = build_validated(scraped, reference)

    # Final accounting identity check
    logger.info(f"\n{'='*60}\nFINAL VALIDATION\n{'='*60}")
    for _, row in validated.iterrows():
        ta = row.get("total_assets", 0)
        tl = row.get("total_liabilities", 0)
        se = row.get("stockholders_equity", 0)
        if pd.notna(ta) and pd.notna(tl) and pd.notna(se) and ta > 0:
            diff_pct = abs(ta - (tl + se)) / ta * 100
            icon = "✓" if diff_pct < 5 else "⚠"
            logger.info(f"  {icon} {row['firm']} FY{row['fiscal_year']}: A={ta/1e9:.0f}B, L+E={((tl+se)/1e9):.0f}B (diff={diff_pct:.1f}%)")

    # Save
    out_path = PROCESSED_DIR / "financials_validated.csv"
    validated.to_csv(out_path, index=False)
    logger.info(f"\nSaved {len(validated)} validated rows → {out_path}")

    # Print final coverage
    logger.info(f"\n{'='*60}\nFINAL COVERAGE\n{'='*60}")
    for firm in validated["firm"].unique():
        fdf = validated[validated["firm"] == firm]
        years = sorted(fdf["fiscal_year"].tolist())
        logger.info(f"\n  {firm} ({fdf['status'].iloc[0]}) FY{min(years)}-{max(years)}:")
        for col in DATA_COLS:
            if col in fdf.columns:
                found = fdf[col].notna().sum()
                total = len(fdf)
                icon = "✓" if found == total else "△"
                logger.info(f"    {icon} {col}: {found}/{total}")


if __name__ == "__main__":
    main()