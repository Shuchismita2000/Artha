"""
Artha - Module 2: Feature Engineering
=======================================
Builds two feature sets from the parsed ledger, behavioral log, and
customer master:

  1. Affordability features  -> feed the Income/Repayment-Capacity model
  2. Intent features         -> feed the Loan-Intent Propensity model

Both are keyed on customer_id and can be joined to labels.csv for training.
"""

import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

OBS_END = pd.Timestamp("2025-06-30")

OUT_DIR = ".//outputs"


def build_affordability_features(txns: pd.DataFrame) -> pd.DataFrame:
    """One row per customer_id. Uses predicted_category from the narration parser."""
    txns = txns.copy()
    txns["date"] = pd.to_datetime(txns["date"])
    txns["month"] = txns["date"].dt.to_period("M")

    # --- Monthly credit totals -> income stability ---
    credits = txns[txns["direction"] == "credit"]
    monthly_credit = credits.groupby(["customer_id", "month"])["amount"].sum().reset_index()
    stability = monthly_credit.groupby("customer_id")["amount"].agg(["mean", "std", "count"]).reset_index()
    stability.columns = ["customer_id", "avg_monthly_credit", "std_monthly_credit", "n_active_months"]
    stability["income_cv"] = (stability["std_monthly_credit"] / stability["avg_monthly_credit"]).fillna(0)
    stability["income_stability_index"] = 1 / (1 + stability["income_cv"])  # higher = more stable

    # --- Category-wise aggregates ---
    cat_totals = txns.groupby(["customer_id", "predicted_category"])["amount"].agg(
        total="sum", count="count"
    ).reset_index()
    cat_pivot_total = cat_totals.pivot(index="customer_id", columns="predicted_category", values="total").fillna(0)
    cat_pivot_total.columns = [f"total_{c}" for c in cat_pivot_total.columns]
    cat_pivot_count = cat_totals.pivot(index="customer_id", columns="predicted_category", values="count").fillna(0)
    cat_pivot_count.columns = [f"count_{c}" for c in cat_pivot_count.columns]

    feats = stability.merge(cat_pivot_total, on="customer_id", how="left")
    feats = feats.merge(cat_pivot_count, on="customer_id", how="left")
    feats = feats.fillna(0)

    total_credit = feats["avg_monthly_credit"] * feats["n_active_months"]
    total_credit = total_credit.replace(0, np.nan)

    # EMI load: EMI totals are negative (debits) -> take abs
    feats["emi_to_income_ratio"] = (feats.get("total_emi", 0).abs() / total_credit).fillna(0)
    feats["rent_to_income_ratio"] = (feats.get("total_rent", 0).abs() / total_credit).fillna(0)
    feats["discretionary_to_income_ratio"] = (feats.get("total_discretionary", 0).abs() / total_credit).fillna(0)
    feats["bounce_count"] = feats.get("count_bounce", 0)
    feats["window_dressing_flag"] = (feats.get("count_window_dressing", 0) > 0).astype(int)

    # Credit frequency & average transaction size (overall behavior signal)
    txn_freq = txns.groupby("customer_id").agg(
        total_txn_count=("amount", "count"),
        avg_txn_amount=("amount", lambda x: x.abs().mean()),
    ).reset_index()
    feats = feats.merge(txn_freq, on="customer_id", how="left")

    keep_cols = [
        "customer_id", "avg_monthly_credit", "income_cv", "income_stability_index",
        "emi_to_income_ratio", "rent_to_income_ratio", "discretionary_to_income_ratio",
        "bounce_count", "window_dressing_flag", "total_txn_count", "avg_txn_amount",
        "count_salary", "count_business_income",
    ]
    for c in keep_cols:
        if c not in feats.columns:
            feats[c] = 0

    return feats[keep_cols]


def build_intent_features(behav: pd.DataFrame) -> pd.DataFrame:
    """One row per customer_id, from behavioral_log."""
    behav = behav.copy()
    behav["timestamp"] = pd.to_datetime(behav["timestamp"])
    behav["days_before_end"] = (OBS_END - behav["timestamp"]).dt.days

    def agg_customer(g):
        logins = (g["event_type"] == "app_login").sum()
        calc_all = (g["event_type"] == "loan_calculator_used").sum()
        calc_30d = ((g["event_type"] == "loan_calculator_used") & (g["days_before_end"] <= 30)).sum()
        pv_all = (g["event_type"] == "product_page_view").sum()
        pv_30d = ((g["event_type"] == "product_page_view") & (g["days_before_end"] <= 30)).sum()
        chat = (g["event_type"] == "customer_care_chat").sum()
        sms = (g["event_type"] == "sms_click").sum()

        intent_events = g[g["event_type"].isin(["loan_calculator_used", "product_page_view"])]
        if len(intent_events) > 0:
            product_counts = intent_events["product_viewed"].value_counts()
            preferred_product = product_counts.idxmax()
            product_concentration = product_counts.max() / len(intent_events)
            days_since_last_intent_event = intent_events["days_before_end"].min()
        else:
            preferred_product = "none"
            product_concentration = 0.0
            days_since_last_intent_event = 999

        avg_session_duration = g["session_duration_seconds"].mean()

        return pd.Series({
            "login_count": logins,
            "calculator_use_count_30d": calc_30d,
            "calculator_use_count_total": calc_all,
            "product_page_view_count_30d": pv_30d,
            "product_page_view_count_total": pv_all,
            "customer_care_chat_count": chat,
            "sms_click_count": sms,
            "preferred_product": preferred_product,
            "product_view_concentration": product_concentration,
            "days_since_last_intent_event": days_since_last_intent_event,
            "avg_session_duration": avg_session_duration,
        })

    feats = behav.groupby("customer_id").apply(agg_customer, include_groups=False).reset_index()
    return feats


if __name__ == "__main__":
    print("Loading data...")
    txns = pd.read_csv(f"{OUT_DIR}/transaction_ledger_parsed.csv")
    behav = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                                  "shuchismitamallick/loan-underwriting-and-customer-behavior-dataset",
                                  "behavioral_log.csv")
    customers = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                                  "shuchismitamallick/loan-underwriting-and-customer-behavior-dataset",
                                  "customer_master.csv")

    print("Building affordability features...")
    affordability_feats = build_affordability_features(txns)
    print(f"  {affordability_feats.shape}")

    print("Building intent features...")
    intent_feats = build_intent_features(behav)
    print(f"  {intent_feats.shape}")

    # Customers with zero behavioral events won't appear in intent_feats -- fill them in.
    all_ids = customers[["customer_id"]]
    intent_feats = all_ids.merge(intent_feats, on="customer_id", how="left")
    intent_feats["preferred_product"] = intent_feats["preferred_product"].fillna("none")
    numeric_intent_cols = intent_feats.select_dtypes(include=[np.number]).columns
    intent_feats[numeric_intent_cols] = intent_feats[numeric_intent_cols].fillna(0)
    intent_feats["days_since_last_intent_event"] = intent_feats["days_since_last_intent_event"].replace(0, 999)

    affordability_feats = all_ids.merge(affordability_feats, on="customer_id", how="left").fillna(0)

    # Merge customer_master demographic features in for the intent model
    intent_feats_full = intent_feats.merge(
        customers[["customer_id", "age", "occupation_type", "city_tier",
                   "existing_loan_count", "residence_type", "relationship_tenure_years",
                   "bureau_score_proxy"]],
        on="customer_id", how="left"
    )

    affordability_feats.to_csv(f"{OUT_DIR}/features/affordability_features.csv", index=False)
    intent_feats_full.to_csv(f"{OUT_DIR}/features/intent_features.csv", index=False)

    print(f"\nSaved affordability_features.csv: {affordability_feats.shape}")
    print(f"Saved intent_features.csv: {intent_feats_full.shape}")
    print("\nSample affordability features:")
    print(affordability_feats.head(3))
    print("\nSample intent features:")
    print(intent_feats_full.head(3))
