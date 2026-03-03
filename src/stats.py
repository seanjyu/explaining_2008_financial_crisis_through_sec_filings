"""
02_statistics.py — Statistical comparison: failed vs. survived firms.

Tests whether the differences visible in the EDA charts are statistically
meaningful, given the small sample size.

Usage:
    python 02_statistics.py

Output:
    figures/07_correlation_matrix.png
    figures/08_group_comparison.png
    Prints test results to console
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import PROCESSED_DIR, FIGURES_DIR

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

combined = pd.read_csv(PROCESSED_DIR / "features_combined.csv")

# ──────────────────────────────────────────────
# 1. Group Comparison Tests (Failed vs Survived)
# ──────────────────────────────────────────────

print("=" * 60)
print("GROUP COMPARISON: Failed vs Survived")
print("=" * 60)

failed = combined[combined["status"] == "failed"]
survived = combined[combined["status"] == "survived"]

test_features = [
    ("leverage_ratio", "Leverage Ratio"),
    ("cash_coverage", "Cash Coverage"),
    ("equity_to_assets_pct", "Equity / Assets %"),
    ("risk_total_crisis_density", "Crisis Keyword Density (Risk Factors)"),
    ("mda_total_crisis_density", "Crisis Keyword Density (MD&A)"),
    ("risk_section_wc", "Risk Section Word Count"),
    ("hedging_language_pct", "Hedging Language %"),
]

results = []

for col, label in test_features:
    f_vals = failed[col].dropna()
    s_vals = survived[col].dropna()

    if len(f_vals) < 2 or len(s_vals) < 2:
        print(f"\n  {label}: insufficient data (failed={len(f_vals)}, survived={len(s_vals)})")
        continue

    # Welch's t-test (doesn't assume equal variance)
    t_stat, p_value = stats.ttest_ind(f_vals, s_vals, equal_var=False)

    # Cohen's d (effect size)
    pooled_std = np.sqrt((f_vals.std()**2 + s_vals.std()**2) / 2)
    cohens_d = (f_vals.mean() - s_vals.mean()) / pooled_std if pooled_std > 0 else 0

    # Mann-Whitney U (non-parametric alternative for small samples)
    u_stat, u_p = stats.mannwhitneyu(f_vals, s_vals, alternative="two-sided")

    sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""

    print(f"\n  {label}:")
    print(f"    Failed:   mean={f_vals.mean():.2f}, std={f_vals.std():.2f}, n={len(f_vals)}")
    print(f"    Survived: mean={s_vals.mean():.2f}, std={s_vals.std():.2f}, n={len(s_vals)}")
    print(f"    Welch's t={t_stat:.3f}, p={p_value:.4f} {sig}")
    print(f"    Mann-Whitney U={u_stat:.0f}, p={u_p:.4f}")
    print(f"    Cohen's d={cohens_d:.3f}")

    results.append({
        "Feature": label,
        "Failed Mean": f_vals.mean(),
        "Survived Mean": s_vals.mean(),
        "t-stat": t_stat,
        "p-value": p_value,
        "Cohen's d": cohens_d,
        "Significant": sig,
    })

# Summary table
if results:
    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    results_df.to_csv(PROCESSED_DIR / "statistical_tests.csv", index=False)
    print(f"\n  Saved: {PROCESSED_DIR / 'statistical_tests.csv'}")


# ──────────────────────────────────────────────
# 2. Group Comparison Bar Chart
# ──────────────────────────────────────────────

print(f"\n{'='*60}")
print("Generating group comparison chart...")

# Normalize features for comparison (z-score within each feature)
plot_features = [col for col, _ in test_features if col in combined.columns]
plot_labels = [label for col, label in test_features if col in combined.columns]

fig, axes = plt.subplots(1, len(plot_features), figsize=(3 * len(plot_features), 5))
if len(plot_features) == 1:
    axes = [axes]

for ax, col, label in zip(axes, plot_features, plot_labels):
    f_vals = failed[col].dropna()
    s_vals = survived[col].dropna()

    means = [f_vals.mean(), s_vals.mean()]
    sems = [f_vals.sem(), s_vals.sem()]
    bars = ax.bar(["Failed", "Survived"], means, yerr=sems,
                  color=["#d62728", "#1f77b4"], alpha=0.8, capsize=5)

    ax.set_title(label, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=8)

    # Add p-value from results
    matching = [r for r in results if r["Feature"] == label]
    if matching:
        p = matching[0]["p-value"]
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(0.5, 0.95, p_str, transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="gray")

fig.suptitle("Failed vs. Survived: Feature Means with Standard Error",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = FIGURES_DIR / "08_group_comparison.png"
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# 3. Correlation Matrix
# ──────────────────────────────────────────────

print(f"\n{'='*60}")
print("Generating correlation matrix...")

corr_cols = [
    "leverage_ratio", "cash_coverage", "equity_to_assets_pct",
    "risk_total_crisis_density", "mda_total_crisis_density",
    "risk_section_wc", "hedging_language_pct",
    "net_income", "provision_for_losses",
]

# Only use columns that exist and have variance
corr_cols = [c for c in corr_cols if c in combined.columns and combined[c].notna().sum() > 3]
corr_labels = {
    "leverage_ratio": "Leverage",
    "cash_coverage": "Cash Coverage",
    "equity_to_assets_pct": "Equity/Assets",
    "risk_total_crisis_density": "Crisis Keywords\n(Risk)",
    "mda_total_crisis_density": "Crisis Keywords\n(MD&A)",
    "risk_section_wc": "Risk Section\nLength",
    "hedging_language_pct": "Hedging\nLanguage",
    "net_income": "Net Income",
    "provision_for_losses": "Loan Loss\nProvisions",
}

corr_data = combined[corr_cols].dropna(how="all")
corr_matrix = corr_data.corr(method="spearman")

# Rename for display
corr_matrix.index = [corr_labels.get(c, c) for c in corr_matrix.index]
corr_matrix.columns = [corr_labels.get(c, c) for c in corr_matrix.columns]

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5, ax=ax,
    annot_kws={"size": 9},
)
ax.set_title("Spearman Correlation: Financial & NLP Features",
             fontsize=13, fontweight="bold", pad=15)

plt.tight_layout()
path = FIGURES_DIR / "07_correlation_matrix.png"
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# 4. Year-over-Year: Did NLP features lead?
# ──────────────────────────────────────────────

print(f"\n{'='*60}")
print("LEAD/LAG ANALYSIS: NLP vs Quantitative")
print("=" * 60)

# Simple question: did crisis keyword density increase BEFORE leverage spiked?
for firm in combined["firm"].unique():
    fdf = combined[combined["firm"] == firm].sort_values("fiscal_year")
    if len(fdf) < 2:
        continue

    nlp_col = "risk_total_crisis_density"
    fin_col = "leverage_ratio"

    nlp_vals = fdf[nlp_col].values
    fin_vals = fdf[fin_col].values

    if np.isnan(nlp_vals).all() or np.isnan(fin_vals).all():
        continue

    # Year-over-year changes
    nlp_changes = np.diff(nlp_vals)
    fin_changes = np.diff(fin_vals)

    print(f"\n  {firm}:")
    years = fdf["fiscal_year"].values
    for i in range(len(nlp_changes)):
        nlp_dir = "↑" if nlp_changes[i] > 0 else "↓"
        fin_dir = "↑" if fin_changes[i] > 0 else "↓"
        print(f"    {years[i]}→{years[i+1]}: Crisis keywords {nlp_dir} ({nlp_changes[i]:+.2f}), Leverage {fin_dir} ({fin_changes[i]:+.2f})")


print(f"\n{'='*60}")
print("Done! All statistical outputs saved.")