# from pull_financials import *
#
# html_path = RAW_DIR / "html" / "AIG_2007.html"
# html = html_path.read_text(encoding="utf-8", errors="replace")
# tables = extract_tables(html)
# print(f"Found {len(tables)} tables")
#
# best = find_best_tables(tables)
# print(f"Balance sheet rows: {len(best['balance_sheet']) if best['balance_sheet'] is not None else 'None'}")
#
# # Try to find total assets manually
# if best["balance_sheet"] is not None:
#     print(best["balance_sheet"].to_string()[:2000])

# Quick peek at Wells Fargo's text
path = r"data\raw\html\WFC_2007.html"
with open(path, encoding="utf-8", errors="replace") as f:
    text = f.read()

# Show first 500 chars
print("=== FIRST 500 CHARS ===")
print(text[:500])

# Search for key terms
import re
for term in ["total assets", "total liabilities", "net income", "stockholders", "shareholders", "cash and"]:
    matches = [(m.start(), text[max(0,m.start()-20):m.end()+50]) for m in re.finditer(term, text, re.IGNORECASE)]
    print(f"\n'{term}': {len(matches)} matches")
    for pos, ctx in matches[:3]:
        print(f"  @{pos}: ...{ctx.strip()[:80]}...")