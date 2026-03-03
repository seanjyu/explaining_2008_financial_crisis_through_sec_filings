"""
features.py — Feature engineering from structured financials and filing text.

Usage:
    python -m src.features

Input:
    data/processed/financials.csv
    data/texts/{ticker}/{year}_{section}.txt

Output:
    data/processed/features_quantitative.csv
    data/processed/features_nlp.csv
    data/processed/features_combined.csv
"""

import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    FIRMS, PROCESSED_DIR, TEXTS_DIR,
    CRISIS_TERMS, HEDGING_WORDS,
    START_YEAR, END_YEAR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# PART A: QUANTITATIVE FEATURES
# ═══════════════════════════════════════════════

def build_quantitative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute financial ratio features from hand-collected 10-K data.
    Input df should have columns matching the financials CSV.
    """
    feat = df[["fiscal_year", "firm", "status", "ticker"]].copy()

    # --- Leverage Ratio ---
    # Total Assets / Stockholders' Equity
    # Higher = more fragile. Lehman was ~30:1, JPM ~12:1
    feat["leverage_ratio"] = safe_divide(df["total_assets"], df["stockholders_equity"])

    # --- Cash Coverage ---
    # Cash / Short-Term Borrowings
    # How many times over can you cover short-term debt with cash on hand?
    feat["cash_coverage"] = safe_divide(df["cash_and_equivalents"], df["short_term_borrowings"])

    # --- Level 3 Asset Concentration ---
    # Level 3 (mark-to-model) assets as % of total assets
    # Only available from FY2007+ (FAS 157 effective date) and not all firms report it
    if "fair_value_level3_assets" in df.columns:
        feat["level3_pct"] = safe_divide(df["fair_value_level3_assets"], df["total_assets"]) * 100
    else:
        feat["level3_pct"] = np.nan

    # --- Provision for Loan Losses (raw) ---
    # Keep the raw value for plotting; we'll compute growth below
    # Not all firms report this (investment banks like Lehman/Bear Stearns don't)
    if "provision_for_loan_losses" in df.columns:
        feat["provision_for_losses"] = df["provision_for_loan_losses"]
    else:
        feat["provision_for_losses"] = np.nan

    # --- Net Income (raw) ---
    feat["net_income"] = df["net_income"]

    # --- YoY Growth Rates (within each firm) ---
    feat = feat.sort_values(["firm", "fiscal_year"])
    for col in ["provision_for_losses", "net_income"]:
        feat[f"{col}_yoy_pct"] = (
            feat.groupby("firm")[col]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            * 100
        )

    # --- Equity-to-Assets Ratio (inverse of leverage, sometimes more intuitive) ---
    feat["equity_to_assets_pct"] = safe_divide(df["stockholders_equity"], df["total_assets"]) * 100

    return feat


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Division that returns NaN instead of inf/error on zero denominators."""
    return numerator / denominator.replace(0, np.nan)


# ═══════════════════════════════════════════════
# PART B: NLP FEATURES
# ═══════════════════════════════════════════════

def load_section_text(ticker: str, year: int, section: str) -> str | None:
    """Load an extracted filing section from disk."""
    path = TEXTS_DIR / ticker / f"{year}_{section}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def count_term_occurrences(text: str, terms: list[str]) -> int:
    """Count total occurrences of a list of terms in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(text_lower.count(term.lower()) for term in terms)


def compute_nlp_features_for_filing(ticker: str, year: int) -> dict:
    """
    Compute all NLP features for a single firm-year filing.
    Returns a dict of feature values.
    """
    risk_text = load_section_text(ticker, year, "risk_factors")
    mda_text = load_section_text(ticker, year, "mda")

    features = {
        "ticker": ticker,
        "year": year,
    }

    # --- Risk Factors Section Features ---
    if risk_text:
        features["risk_section_wc"] = len(risk_text.split())

        # Crisis keyword density (per 1000 words)
        total_words = len(risk_text.split())
        for category, terms in CRISIS_TERMS.items():
            count = count_term_occurrences(risk_text, terms)
            features[f"risk_{category}_count"] = count
            features[f"risk_{category}_density"] = (count / total_words * 1000) if total_words > 0 else 0
    else:
        features["risk_section_wc"] = np.nan

    # --- MD&A Section Features ---
    if mda_text:
        features["mda_section_wc"] = len(mda_text.split())

        total_words = len(mda_text.split())

        # Crisis keyword density in MD&A
        for category, terms in CRISIS_TERMS.items():
            count = count_term_occurrences(mda_text, terms)
            features[f"mda_{category}_count"] = count
            features[f"mda_{category}_density"] = (count / total_words * 1000) if total_words > 0 else 0

        # Hedging language percentage
        hedging_count = count_term_occurrences(mda_text, HEDGING_WORDS)
        features["hedging_language_pct"] = (hedging_count / total_words * 100) if total_words > 0 else 0
    else:
        features["mda_section_wc"] = np.nan
        features["hedging_language_pct"] = np.nan

    return features


def build_nlp_features() -> pd.DataFrame:
    """Build NLP features for all firms across all years."""
    rows = []

    for firm_name, info in FIRMS.items():
        ticker = info["ticker"]
        for year in range(START_YEAR, END_YEAR + 1):
            logger.info(f"  NLP features: {firm_name} FY{year}")
            features = compute_nlp_features_for_filing(ticker, year)
            features["firm"] = firm_name
            features["status"] = info["status"]
            rows.append(features)

    df = pd.DataFrame(rows)

    # --- YoY Growth in Risk Section Word Count ---
    df = df.sort_values(["firm", "year"])
    df["risk_section_wc_yoy_pct"] = (
        df.groupby("firm")["risk_section_wc"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    # --- Combined crisis term density (sum across all categories) ---
    risk_density_cols = [c for c in df.columns if c.startswith("risk_") and c.endswith("_density")]
    df["risk_total_crisis_density"] = df[risk_density_cols].sum(axis=1)

    mda_density_cols = [c for c in df.columns if c.startswith("mda_") and c.endswith("_density")]
    df["mda_total_crisis_density"] = df[mda_density_cols].sum(axis=1)

    return df


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    # --- Quantitative Features ---
    logger.info("Building quantitative features...")
    financials = pd.read_csv(PROCESSED_DIR / "financials_validated.csv")

    quant = build_quantitative_features(financials)
    quant.to_csv(PROCESSED_DIR / "features_quantitative.csv", index=False)
    logger.info(f"  Saved {len(quant)} rows → features_quantitative.csv")

    # --- NLP Features ---
    logger.info("\nBuilding NLP features...")
    nlp = build_nlp_features()
    nlp.to_csv(PROCESSED_DIR / "features_nlp.csv", index=False)
    logger.info(f"  Saved {len(nlp)} rows → features_nlp.csv")

    # --- Combined (merge on firm + year) ---
    combined = pd.merge(
        quant, nlp,
        left_on=["firm", "fiscal_year", "status", "ticker"],
        right_on=["firm", "year", "status", "ticker"],
        how="outer",
        suffixes=("_quant", "_nlp"),
    )
    # Clean up duplicate year columns
    if "year" in combined.columns:
        combined["fiscal_year"] = combined["fiscal_year"].fillna(combined["year"])
        combined = combined.drop(columns=["year"])

    combined.to_csv(PROCESSED_DIR / "features_combined.csv", index=False)
    logger.info(f"\n  Saved {len(combined)} rows → features_combined.csv")

    # Quick feature summary
    logger.info("\n--- Feature Summary ---")
    feature_cols = [c for c in combined.columns if c not in ["firm", "fiscal_year", "status", "ticker"]]
    for col in feature_cols:
        non_null = combined[col].notna().sum()
        logger.info(f"  {col}: {non_null}/{len(combined)} non-null")


if __name__ == "__main__":
    main()