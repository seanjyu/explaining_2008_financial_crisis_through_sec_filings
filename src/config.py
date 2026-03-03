"""
config.py — Central configuration for the 2008 Financial Crisis EDGAR project.
All CIK codes, date ranges, XBRL tags, and NLP dictionaries live here.
"""

from pathlib import Path

# --- Project Paths ---
ROOT = Path(__file__).resolve().parent.parent  # assumes src/ is one level below project root
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEXTS_DIR = DATA_DIR / "texts"
FIGURES_DIR = ROOT / "figures"

for d in [RAW_DIR, PROCESSED_DIR, TEXTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- SEC EDGAR API ---
# Required by SEC fair access policy: https://www.sec.gov/os/accessing-edgar-data
USER_AGENT = "YourName your.email@example.com"  # <-- CHANGE THIS
BASE_URL = "https://data.sec.gov"
RATE_LIMIT_SEC = 0.12  # ~8 req/sec, staying under SEC's 10/sec limit

# --- Firms ---
# Failed / bailed out
# Survived
FIRMS = {
    "Lehman Brothers":   {"cik": "0000806085", "ticker": "LEH",  "status": "failed"},
    "Bear Stearns":      {"cik": "0000777001", "ticker": "BSC",  "status": "failed"},
    "AIG":               {"cik": "0000005272", "ticker": "AIG",  "status": "failed"},
    "JPMorgan Chase":    {"cik": "0000019617", "ticker": "JPM",  "status": "survived"},
    "Wells Fargo":       {"cik": "0000072971", "ticker": "WFC",  "status": "survived"},
}

# --- Time Window ---
START_YEAR = 2005
END_YEAR = 2008

# --- XBRL Tags to Pull ---
# Maps a readable name -> (taxonomy, tag, unit)
# These are the us-gaap tags we need for feature engineering.
# Some tags changed names over time; alternates are listed for fallback.
XBRL_FIELDS = {
    "total_assets": {
        "taxonomy": "us-gaap",
        "tags": ["Assets"],
        "unit": "USD",
    },
    "total_liabilities": {
        "taxonomy": "us-gaap",
        "tags": ["Liabilities"],
        "unit": "USD",
    },
    "stockholders_equity": {
        "taxonomy": "us-gaap",
        "tags": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "unit": "USD",
    },
    "cash_and_equivalents": {
        "taxonomy": "us-gaap",
        "tags": [
            "CashAndCashEquivalentsAtCarryingValue",
            "Cash",
        ],
        "unit": "USD",
    },
    "short_term_borrowings": {
        "taxonomy": "us-gaap",
        "tags": [
            "ShortTermBorrowings",
            "CommercialPaper",
        ],
        "unit": "USD",
    },
    "net_income": {
        "taxonomy": "us-gaap",
        "tags": [
            "NetIncomeLoss",
            "ProfitLoss",
        ],
        "unit": "USD",
    },
    "provision_for_loan_losses": {
        "taxonomy": "us-gaap",
        "tags": [
            "ProvisionForLoanAndLeaseLosses",
            "ProvisionForLoanLeaseAndOtherLosses",
            "ProvisionForCreditLosses",
        ],
        "unit": "USD",
    },
    "fair_value_level3_assets": {
        "taxonomy": "us-gaap",
        "tags": [
            "FairValueMeasuredOnRecurringBasisAssetsLevel3",
            "AssetsLevel3",
            "FairValueAssetsMeasuredOnRecurringBasisUnobservableInputReconciliationByAssetClassDomain",
        ],
        "unit": "USD",
    },
}

# --- NLP: Crisis Keyword Dictionary ---
CRISIS_TERMS = {
    "subprime": ["subprime", "sub-prime", "sub prime"],
    "liquidity_risk": [
        "liquidity risk", "funding pressure", "liquidity facility",
        "liquidity position", "funding liquidity",
    ],
    "defaults": [
        "default", "delinquency", "delinquent",
        "nonperforming", "non-performing", "foreclosure",
    ],
    "counterparty": ["counterparty risk", "counterparty exposure", "counterparty credit"],
    "writedown": ["write-down", "writedown", "write down", "impairment", "impaired"],
    "off_balance_sheet": [
        "off-balance sheet", "off balance sheet",
        "variable interest entity", "variable interest entities",
        "VIE", "special purpose entity", "special purpose entities",
        "structured investment vehicle", "SIV", "conduit",
    ],
}

# Words used to measure hedging / uncertainty language in MD&A
HEDGING_WORDS = [
    "may", "might", "could", "would", "should",
    "uncertain", "uncertainty", "no assurance",
    "potentially", "possible", "possibly",
    "approximate", "approximately", "estimate", "estimated",
    "subject to", "not guaranteed", "cannot predict",
]

# --- Key Crisis Event Dates (for chart annotations) ---
CRISIS_EVENTS = {
    "2007-02-07": "HSBC subprime write-down",
    "2007-08-09": "BNP Paribas freezes funds",
    "2008-03-16": "Bear Stearns rescue",
    "2008-09-15": "Lehman bankruptcy",
    "2008-09-16": "AIG bailout",
    "2008-10-03": "TARP signed into law",
}