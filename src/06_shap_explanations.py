"""
Artha - Module 6: Per-Customer SHAP Explanations
==================================================
Computes SHAP contributions for EVERY customer (not just a holdout sample)
on both models, and extracts the top contributing features per customer.

This powers two dashboard features:
  1. "Score drivers" - why a lead scored what it did (works for any lead,
     including top-ranked ones).
  2. "Why not this lead" - the same mechanism, just naturally shows
     negative contributions when applied to a low-scoring customer.
     No separate logic needed -- it's the same explanation, read either way.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import json

DATA_DIR = "/home/claude/artha_project/data"
OUT_DIR = "/home/claude/artha_project/outputs"
MODEL_DIR = "/home/claude/artha_project/models"

INCOME_FEATURE_COLS = [
    "avg_monthly_credit", "income_cv", "income_stability_index",
    "emi_to_income_ratio", "rent_to_income_ratio", "discretionary_to_income_ratio",
    "bounce_count", "window_dressing_flag", "total_txn_count", "avg_txn_amount",
    "count_salary", "count_business_income",
]

INCOME_LABELS = {
    "avg_monthly_credit": "Average monthly credit inflow",
    "income_cv": "Income volatility month-to-month",
    "income_stability_index": "Income stability index",
    "emi_to_income_ratio": "Existing EMI burden",
    "rent_to_income_ratio": "Rent burden",
    "discretionary_to_income_ratio": "Discretionary spending ratio",
    "bounce_count": "Bounced or overdraft transactions",
    "window_dressing_flag": "Pre-application balance spike",
    "total_txn_count": "Overall transaction activity",
    "avg_txn_amount": "Average transaction size",
    "count_salary": "Salary credit count",
    "count_business_income": "Business income transaction count",
}

NUMERIC_INTENT_FEATURES = [
    "login_count", "calculator_use_count_30d", "calculator_use_count_total",
    "product_page_view_count_30d", "product_page_view_count_total",
    "customer_care_chat_count", "sms_click_count", "product_view_concentration",
    "days_since_last_intent_event", "avg_session_duration",
    "age", "existing_loan_count", "relationship_tenure_years", "bureau_score_proxy",
]
CATEGORICAL_INTENT_FEATURES = ["preferred_product", "occupation_type", "city_tier", "residence_type"]

INTENT_LABELS = {
    "login_count": "App login frequency",
    "calculator_use_count_30d": "Loan calculator use, last 30 days",
    "calculator_use_count_total": "Loan calculator use, total",
    "product_page_view_count_30d": "Product page views, last 30 days",
    "product_page_view_count_total": "Product page views, total",
    "customer_care_chat_count": "Customer care chat interactions",
    "sms_click_count": "SMS click-throughs",
    "product_view_concentration": "Focus on a single product",
    "days_since_last_intent_event": "Recency of last research activity",
    "avg_session_duration": "Average app session duration",
    "age": "Age",
    "existing_loan_count": "Existing loan count",
    "relationship_tenure_years": "Banking relationship tenure",
    "bureau_score_proxy": "Credit bureau score",
    "preferred_product_enc": "Preferred loan product",
    "occupation_type_enc": "Occupation type",
    "city_tier_enc": "City tier",
    "residence_type_enc": "Residence type",
}

TOP_N_DRIVERS = 4


def top_drivers(shap_row, feature_names, labels, raw_values=None):
    """Return top-N features by |SHAP|, signed, human-readable."""
    order = np.argsort(-np.abs(shap_row))[:TOP_N_DRIVERS]
    drivers = []
    for idx in order:
        fname = feature_names[idx]
        drivers.append({
            "feature": fname,
            "label": labels.get(fname, fname),
            "shap": round(float(shap_row[idx]), 4),
            "direction": "positive" if shap_row[idx] >= 0 else "negative",
        })
    return drivers


def main():
    print("=== Income model SHAP (all customers) ===")
    income_model = joblib.load(f"{MODEL_DIR}/income_model.pkl")
    aff_feats = pd.read_csv(f"{OUT_DIR}/affordability_features.csv")
    X_income = aff_feats[INCOME_FEATURE_COLS]

    explainer_income = shap.TreeExplainer(income_model)
    shap_income = explainer_income.shap_values(X_income)
    print(f"Computed SHAP for {shap_income.shape[0]} customers x {shap_income.shape[1]} features")

    income_explanations = {}
    for i, cid in enumerate(aff_feats["customer_id"]):
        income_explanations[cid] = top_drivers(shap_income[i], INCOME_FEATURE_COLS, INCOME_LABELS)

    print("\n=== Intent model SHAP (all customers) ===")
    intent_model = joblib.load(f"{MODEL_DIR}/intent_model.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/intent_encoders.pkl")
    intent_feats = pd.read_csv(f"{OUT_DIR}/intent_features.csv")

    for col in CATEGORICAL_INTENT_FEATURES:
        le = encoders[col]
        # handle unseen categories gracefully (shouldn't occur here, but safe)
        intent_feats[col + "_enc"] = intent_feats[col].astype(str).map(
            lambda v: le.transform([v])[0] if v in le.classes_ else -1
        )

    feature_cols = NUMERIC_INTENT_FEATURES + [c + "_enc" for c in CATEGORICAL_INTENT_FEATURES]
    X_intent = intent_feats[feature_cols]

    explainer_intent = shap.TreeExplainer(intent_model)
    shap_intent = explainer_intent.shap_values(X_intent)
    if isinstance(shap_intent, list):
        shap_intent = shap_intent[1]
    print(f"Computed SHAP for {shap_intent.shape[0]} customers x {shap_intent.shape[1]} features")

    intent_explanations = {}
    for i, cid in enumerate(intent_feats["customer_id"]):
        intent_explanations[cid] = top_drivers(shap_intent[i], feature_cols, INTENT_LABELS)

    print("\nMerging into combined explanations file...")
    combined = {}
    all_ids = set(income_explanations.keys()) | set(intent_explanations.keys())
    for cid in all_ids:
        combined[cid] = {
            "affordability_drivers": income_explanations.get(cid, []),
            "intent_drivers": intent_explanations.get(cid, []),
        }

    out_path = f"{OUT_DIR}/lead_explanations.json"
    with open(out_path, "w") as f:
        json.dump(combined, f)

    print(f"Saved {len(combined)} customer explanations to {out_path}")

    # Sanity print for one customer
    sample_id = list(combined.keys())[0]
    print(f"\nSample explanation for {sample_id}:")
    print(json.dumps(combined[sample_id], indent=2))


if __name__ == "__main__":
    main()
