"""
Artha - Module 1: Transaction Narration Parser
================================================
Rules-based (regex/keyword) classification of raw bank narrations into
categories: salary, business_income, emi, rent, discretionary, bounce.

Deliberately NOT an LLM call -- this needs to run over millions of rows,
cheaply, deterministically, and in a way that's auditable to a bank's
compliance function. See DATA_DICTIONARY.md for why.

We validate against `true_category` (present only in our synthetic data,
standing in for a manually-labeled validation set in production) to report
parser accuracy -- a real deliverable for the pitch.
"""

import re
import pandas as pd  
import kagglehub
from kagglehub import KaggleDatasetAdapter

# ---------------------------------------------------------------------------
# Rule definitions -- ordered; first match wins.
# In production these would be tuned against a labeled sample of real
# narrations and expanded iteratively.
# ---------------------------------------------------------------------------
RULES = [
    ("bounce",          re.compile(r"RETURN|INSUFFICIENT|BOUNCE|ECS.?RETURN", re.I)),
    ("emi",             re.compile(r"\bEMI\b|LOAN\d+|ECS-EMI", re.I)),
    ("rent",             re.compile(r"\bRENT\b|LANDLORD", re.I)),
    ("salary",           re.compile(r"\bSAL(ARY)?\b|SALARY CREDIT", re.I)),
    ("window_dressing", re.compile(r"TRANSFER-IN", re.I)),
    ("business_income",  re.compile(r"CLIENTPAY|BUSINESS RECEIPT", re.I)),
    ("discretionary",    re.compile(r"PAYMENT-\d+|UPI-(SWIGGY|ZOMATO|AMAZON|FLIPKART|BIGBASKET|MYNTRA|IRCTC|RELIANCE DIGITAL|DMART|PVR CINEMAS)", re.I)),
]

FALLBACK_CATEGORY = "discretionary"  # safest default for unmatched narrations


def classify_narration(narration: str) -> str:
    """Apply ordered rules to a single narration string."""
    for category, pattern in RULES:
        if pattern.search(narration):
            return category
    return FALLBACK_CATEGORY


def parse_transactions(df: pd.DataFrame, narration_col: str = "narration_raw") -> pd.DataFrame:
    """Vectorized-ish classification across the full ledger."""
    df = df.copy()
    df["predicted_category"] = df[narration_col].apply(classify_narration)
    return df


def evaluate_parser(df: pd.DataFrame, true_col: str = "true_category",
                     pred_col: str = "predicted_category") -> dict:
    """Compare predicted vs ground truth (only possible because this is synthetic data)."""
    accuracy = (df[true_col] == df[pred_col]).mean()
    confusion = pd.crosstab(df[true_col], df[pred_col])
    per_class_recall = df.groupby(true_col).apply(
        lambda g: (g[pred_col] == g.name).mean()
    )
    return {
        "overall_accuracy": accuracy,
        "confusion_matrix": confusion,
        "per_class_recall": per_class_recall,
    }


if __name__ == "__main__":
    print("Loading transaction ledger...")
    txns = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                                  "shuchismitamallick/loan-underwriting-and-customer-behavior-dataset",
                                  "transaction_ledger.csv")

    print("Running narration parser...")
    txns = parse_transactions(txns)

    print("\nEvaluating parser accuracy against ground truth...")
    results = evaluate_parser(txns)

    print(f"\nOverall parser accuracy: {results['overall_accuracy']:.4f}")
    print("\nPer-class recall:")
    print(results["per_class_recall"].round(4))
    print("\nConfusion matrix:")
    print(results["confusion_matrix"])

    # Save parsed ledger with predicted_category for downstream feature engineering.
    # (We keep predicted_category, not true_category, as the "real" signal a
    # production system would use -- true_category exists only for this validation step.)
    out_path = ".//outputs/transaction_ledger_parsed.csv"
    txns.drop(columns=["true_category"]).to_csv(out_path, index=False)
    print(f"\nSaved parsed ledger to {out_path}")
