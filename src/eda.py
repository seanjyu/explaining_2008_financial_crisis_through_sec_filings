"""
01_eda.py — Exploratory Data Analysis: 2008 Financial Crisis via SEC EDGAR

Generates 6 annotated charts comparing failed vs. surviving firms
across quantitative and NLP-derived features from 10-K filings.

Usage:
    python 01_eda.py

Output:
    figures/01_leverage_ratio.png
    figures/02_cash_coverage.png
    figures/03_net_income.png
    figures/04_provision_growth.png
    figures/05_crisis_keyword_density.png
    figures/06_risk_section_growth.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from config import PROCESSED_DIR, FIGURES_DIR, CRISIS_EVENTS

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

combined = pd.read_csv(PROCESSED_DIR / "features_combined.csv")

# Color scheme: reds for failed, blues for survived
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

# Key crisis dates for annotations
EVENTS = {
    2007: "BNP Paribas\nfreezes funds\n(Aug '07)",
    2008: "Lehman\nbankruptcy\n(Sep '08)",
}

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def add_crisis_annotations(ax, y_pos=None):
    """Add vertical lines and labels for key crisis events."""
    for year, label in EVENTS.items():
        ax.axvline(x=year, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        if y_pos is not None:
            ax.text(year + 0.05, y_pos, label, fontsize=7, color="gray", va="top")


def plot_lines(df, y_col, title, ylabel, filename, y_fmt=None, y_pos_annot=None):
    """Generic line chart: one line per firm, colored by status."""
    # Print the data table
    pivot = df.pivot_table(index="firm", columns="fiscal_year", values=y_col, aggfunc="first")
    pivot["status"] = df.drop_duplicates("firm").set_index("firm")["status"]
    pivot = pivot.sort_values("status", ascending=True)
    print(f"\n  Data ({y_col}):")
    print(pivot.to_string())
    print()

    fig, ax = plt.subplots()

    for firm in df["firm"].unique():
        fdf = df[df["firm"] == firm].sort_values("fiscal_year")
        status = fdf["status"].iloc[0]
        ax.plot(
            fdf["fiscal_year"], fdf[y_col],
            color=COLORS[firm], marker=MARKERS[firm],
            linewidth=2, markersize=7, label=firm,
            linestyle="--" if status == "failed" else "-",
        )

    add_crisis_annotations(ax, y_pos=y_pos_annot)
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks([2005, 2006, 2007, 2008])
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    if y_fmt == "comma":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    elif y_fmt == "billions":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x/1e9:.0f}B"))
    elif y_fmt == "pct":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1f}%"))

    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Chart 1: Leverage Ratio
# ──────────────────────────────────────────────
print("\nChart 1: Leverage Ratio")
plot_lines(
    combined, "leverage_ratio",
    title="Failed Firms Were 2–3× More Leveraged Than Survivors",
    ylabel="Leverage Ratio (Total Assets / Equity)",
    filename="01_leverage_ratio.png",
    y_fmt="comma",
    y_pos_annot=combined["leverage_ratio"].max() * 0.9,
)

# ──────────────────────────────────────────────
# Chart 2: Cash Coverage
# ──────────────────────────────────────────────
print("Chart 2: Cash Coverage")
plot_lines(
    combined, "cash_coverage",
    title="Liquidity Buffers Were Thin — and Shrinking — at Failed Firms",
    ylabel="Cash / Short-Term Borrowings",
    filename="02_cash_coverage.png",
    y_fmt="comma",
    y_pos_annot=combined["cash_coverage"].max() * 0.9,
)

# ──────────────────────────────────────────────
# Chart 3: Net Income Trajectory
# ──────────────────────────────────────────────
print("Chart 3: Net Income")
plot_lines(
    combined, "net_income",
    title="Earnings Collapsed for Failed Firms — AIG Lost $99B in 2008",
    ylabel="Net Income (USD)",
    filename="03_net_income.png",
    y_fmt="billions",
    y_pos_annot=combined["net_income"].max() * 0.9,
)

# ──────────────────────────────────────────────
# Chart 4: Provision for Loan Losses
# ──────────────────────────────────────────────
print("Chart 4: Provision for Loan Losses")
# Only firms that report provisions (commercial banks + AIG)
prov_df = combined[combined["provision_for_losses"].notna()]
if len(prov_df) > 0:
    plot_lines(
        prov_df, "provision_for_losses",
        title="Loan Loss Provisions Surged — But the Reaction Came Late",
        ylabel="Provision for Loan Losses (USD)",
        filename="04_provision_growth.png",
        y_fmt="billions",
        y_pos_annot=prov_df["provision_for_losses"].max() * 0.9,
    )

# ──────────────────────────────────────────────
# Chart 5: Crisis Keyword Density in Risk Factors
# ──────────────────────────────────────────────
print("Chart 5: Crisis Keyword Density")
plot_lines(
    combined, "risk_total_crisis_density",
    title="Crisis Language Spiked in Filings Before Collapse",
    ylabel="Crisis Term Density (per 1,000 words in Risk Factors)",
    filename="05_crisis_keyword_density.png",
    y_fmt="comma",
    y_pos_annot=combined["risk_total_crisis_density"].max() * 0.9,
)

# ──────────────────────────────────────────────
# Chart 6: Risk Factors Section Word Count Growth
# ──────────────────────────────────────────────
print("Chart 6: Risk Section Growth")
plot_lines(
    combined, "risk_section_wc",
    title="Risk Factor Disclosures Ballooned Before the Crisis",
    ylabel="Risk Factors Section (word count)",
    filename="06_risk_section_growth.png",
    y_fmt="comma",
    y_pos_annot=combined["risk_section_wc"].max() * 0.9,
)

# ──────────────────────────────────────────────
# Bonus: Combined Dashboard (subplots)
# ──────────────────────────────────────────────
print("\nBonus: Combined Dashboard")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
charts = [
    ("leverage_ratio", "Leverage Ratio\n(Assets / Equity)", None),
    ("cash_coverage", "Cash Coverage\n(Cash / ST Borrowings)", None),
    ("net_income", "Net Income", "billions"),
    ("provision_for_losses", "Loan Loss Provisions", "billions"),
    ("risk_total_crisis_density", "Crisis Keyword Density\n(per 1k words)", None),
    ("risk_section_wc", "Risk Section\nWord Count", "comma"),
]

for ax, (col, ylabel, fmt) in zip(axes.flat, charts):
    for firm in combined["firm"].unique():
        fdf = combined[combined["firm"] == firm].sort_values("fiscal_year")
        if fdf[col].isna().all():
            continue
        status = fdf["status"].iloc[0]
        ax.plot(
            fdf["fiscal_year"], fdf[col],
            color=COLORS[firm], marker=MARKERS[firm],
            linewidth=1.5, markersize=5, label=firm,
            linestyle="--" if status == "failed" else "-",
        )

    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks([2005, 2006, 2007, 2008])
    ax.tick_params(labelsize=8)

    if fmt == "billions":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x/1e9:.0f}B"))
    elif fmt == "comma":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Crisis line
    ax.axvline(x=2007, color="gray", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.axvline(x=2008, color="gray", linestyle="--", alpha=0.4, linewidth=0.6)

# Single legend for all subplots
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, frameon=False)

fig.suptitle(
    "What Could You Have Seen in Public SEC Filings Before the 2008 Collapse?",
    fontsize=14, fontweight="bold", y=1.02,
)
plt.tight_layout()
path = FIGURES_DIR / "00_dashboard.png"
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

print("\nDone! All charts saved to figures/")