"""
sentiment.py — FinBERT sentiment analysis on Risk Factors and MD&A sections.

Improvements over naive approach:
  1. Scores BOTH Risk Factors and MD&A (Risk Factors extracted well for all firms)
  2. Sentence-level scoring (FinBERT was trained on sentences, not arbitrary chunks)
  3. Filters out junk: lines that are mostly numbers, too short, or boilerplate
  4. Reports per-section results for comparison


Requires:
    pip install transformers torch

Output:
    data/processed/sentiment_scores.csv
    figures/09_sentiment_trajectory.png
    figures/10_negative_ratio.png
"""

import re
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from config import (
    FIRMS, TEXTS_DIR, PROCESSED_DIR, FIGURES_DIR,
    START_YEAR, END_YEAR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MIN_PARAGRAPHS = 10


# ──────────────────────────────────────────────
# 1. Load FinBERT
# ──────────────────────────────────────────────

def load_finbert():
    logger.info("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
    )
    logger.info("FinBERT loaded ✓")
    return pipe


# ──────────────────────────────────────────────
# 2. Sentence splitting + quality filtering
# ──────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Split on period/exclamation/question followed by space and capital letter
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in raw if s.strip()]


def is_quality_sentence(sentence: str) -> bool:
    """Filter out junk: too short, mostly numbers, boilerplate."""
    words = sentence.split()

    # Too short — not enough context for sentiment
    if len(words) < 8:
        return False

    # Too long — probably a malformed chunk
    if len(words) > 200:
        return False

    # Mostly numbers (table remnants)
    alpha_chars = sum(1 for c in sentence if c.isalpha())
    total_chars = len(sentence)
    if total_chars > 0 and alpha_chars / total_chars < 0.5:
        return False

    # Common boilerplate patterns
    boilerplate = [
        r"^item\s+\d",
        r"^table of contents",
        r"^see\s+note",
        r"^page\s+\d",
        r"^\(\d+\)\s*$",
        r"^the\s+following\s+table",
        r"^in\s+millions",
        r"^dollars\s+in",
    ]
    for pattern in boilerplate:
        if re.match(pattern, sentence, re.IGNORECASE):
            return False

    return True


def extract_quality_sentences(text: str) -> list[str]:
    """Split into sentences and filter to quality content."""
    sentences = split_into_sentences(text)
    quality = [s for s in sentences if is_quality_sentence(s)]
    return quality


# ──────────────────────────────────────────────
# 3. Score sentences with FinBERT
# ──────────────────────────────────────────────

def score_sentences(pipe, sentences: list[str]) -> list[dict]:
    """Score each sentence with FinBERT in batches."""
    results = []
    batch_size = 32

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        scores = pipe(batch)

        for text, score in zip(batch, scores):
            results.append({
                "text": text[:120],
                "label": score["label"],
                "score": score["score"],
            })

    return results


def aggregate_scores(scored: list[dict]) -> dict:
    """Aggregate sentence-level scores into document-level metrics."""
    if not scored:
        return {
            "n_sentences": 0,
            "positive_ratio": np.nan,
            "negative_ratio": np.nan,
            "neutral_ratio": np.nan,
            "avg_positive_conf": np.nan,
            "avg_negative_conf": np.nan,
            "sentiment_score": np.nan,
        }

    labels = [s["label"] for s in scored]
    n = len(labels)

    pos = labels.count("positive") / n
    neg = labels.count("negative") / n
    neu = labels.count("neutral") / n

    pos_confs = [s["score"] for s in scored if s["label"] == "positive"]
    neg_confs = [s["score"] for s in scored if s["label"] == "negative"]

    return {
        "n_sentences": n,
        "positive_ratio": pos,
        "negative_ratio": neg,
        "neutral_ratio": neu,
        "avg_positive_conf": np.mean(pos_confs) if pos_confs else 0,
        "avg_negative_conf": np.mean(neg_confs) if neg_confs else 0,
        "sentiment_score": pos - neg,
    }


# ──────────────────────────────────────────────
# 4. Process all filings
# ──────────────────────────────────────────────

def process_section(pipe, ticker: str, year: int, section: str) -> dict:
    """Score a single section for a firm-year."""
    path = TEXTS_DIR / ticker / f"{year}_{section}.txt"

    if not path.exists():
        return {"section": section, "n_sentences": 0}

    text = path.read_text(encoding="utf-8")
    sentences = extract_quality_sentences(text)

    if len(sentences) < MIN_PARAGRAPHS:
        logger.info(f"    {section}: only {len(sentences)} quality sentences (skipping)")
        return {"section": section, "n_sentences": len(sentences)}

    logger.info(f"    {section}: scoring {len(sentences)} sentences...")
    scored = score_sentences(pipe, sentences)
    agg = aggregate_scores(scored)
    agg["section"] = section

    logger.info(
        f"      {agg['positive_ratio']:.1%} pos, "
        f"{agg['negative_ratio']:.1%} neg, "
        f"{agg['neutral_ratio']:.1%} neutral "
        f"(score: {agg['sentiment_score']:+.3f})"
    )

    # Log top negative sentence
    neg_sents = sorted(
        [s for s in scored if s["label"] == "negative"],
        key=lambda x: x["score"], reverse=True
    )
    if neg_sents:
        logger.info(f"      Most negative: \"{neg_sents[0]['text']}...\"")

    return agg


def main():
    pipe = load_finbert()
    rows = []

    sections = ["risk_factors", "mda"]

    for firm_name, info in FIRMS.items():
        ticker = info["ticker"]
        logger.info(f"\n{'='*50}\n{firm_name} ({ticker})\n{'='*50}")

        for year in range(START_YEAR, END_YEAR + 1):
            logger.info(f"  FY{year}:")

            for section in sections:
                agg = process_section(pipe, ticker, year, section)

                row = {
                    "firm": firm_name, "ticker": ticker,
                    "status": info["status"], "fiscal_year": year,
                    **agg,
                }
                rows.append(row)

    # Save all results
    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "sentiment_scores.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved {len(df)} rows → {out_path}")

    # ──────────────────────────────────────────
    # Summary tables
    # ──────────────────────────────────────────

    has_data = df[df["n_sentences"] >= MIN_PARAGRAPHS].copy()

    for section in sections:
        sec_df = has_data[has_data["section"] == section]
        if sec_df.empty:
            continue

        print(f"\n{'='*60}")
        print(f"SENTIMENT SCORE — {section.upper().replace('_', ' ')}")
        print(f"  (Firms with >= {MIN_PARAGRAPHS} quality sentences)")
        print("=" * 60)
        pivot = sec_df.pivot_table(index="firm", columns="fiscal_year", values="sentiment_score", aggfunc="first")
        print(pivot.to_string())

        print(f"\nNEGATIVE RATIO — {section.upper().replace('_', ' ')}")
        pivot_neg = sec_df.pivot_table(index="firm", columns="fiscal_year", values="negative_ratio", aggfunc="first")
        print(pivot_neg.to_string())

        print(f"\nSENTENCE COUNTS — {section.upper().replace('_', ' ')}")
        pivot_n = sec_df.pivot_table(index="firm", columns="fiscal_year", values="n_sentences", aggfunc="first")
        print(pivot_n.to_string())

    # ──────────────────────────────────────────
    # Charts
    # ──────────────────────────────────────────

    COLORS = {
        "Lehman Brothers": "#d62728",
        "Bear Stearns": "#ff7f0e",
        "AIG": "#e377c2",
        "JPMorgan Chase": "#1f77b4",
        "Wells Fargo": "#2ca02c",
    }
    MARKERS = {
        "Lehman Brothers": "o",
        "Bear Stearns": "s",
        "AIG": "D",
        "JPMorgan Chase": "^",
        "Wells Fargo": "v",
    }

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Generate one pair of charts per section
    for section in sections:
        sec_df = has_data[has_data["section"] == section]
        if sec_df.empty:
            continue

        section_label = "Risk Factors" if section == "risk_factors" else "MD&A"

        # Sentiment trajectory
        fig, ax = plt.subplots()
        for firm in sec_df["firm"].unique():
            fdf = sec_df[sec_df["firm"] == firm].sort_values("fiscal_year")
            if fdf["sentiment_score"].isna().all():
                continue
            status = fdf["status"].iloc[0]
            ax.plot(
                fdf["fiscal_year"], fdf["sentiment_score"],
                color=COLORS[firm], marker=MARKERS[firm],
                linewidth=2, markersize=7, label=firm,
                linestyle="--" if status == "failed" else "-",
            )

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.axvline(x=2007, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axvline(x=2008, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Fiscal Year")
        ax.set_ylabel("Sentiment Score (positive − negative ratio)")
        ax.set_title(f"FinBERT Sentiment in {section_label} Sections",
                     fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks([2005, 2006, 2007, 2008])
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()

        fname = f"09_sentiment_{section}.png"
        path = FIGURES_DIR / fname
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {path}")

        # Negative ratio
        fig, ax = plt.subplots()
        for firm in sec_df["firm"].unique():
            fdf = sec_df[sec_df["firm"] == firm].sort_values("fiscal_year")
            if fdf["negative_ratio"].isna().all():
                continue
            status = fdf["status"].iloc[0]
            ax.plot(
                fdf["fiscal_year"], fdf["negative_ratio"] * 100,
                color=COLORS[firm], marker=MARKERS[firm],
                linewidth=2, markersize=7, label=firm,
                linestyle="--" if status == "failed" else "-",
            )

        ax.axvline(x=2007, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axvline(x=2008, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Fiscal Year")
        ax.set_ylabel("% of Sentences with Negative Sentiment")
        ax.set_title(f"Negative Sentiment in {section_label} Sections",
                     fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks([2005, 2006, 2007, 2008])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}%"))
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()

        fname = f"10_negative_{section}.png"
        path = FIGURES_DIR / fname
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()