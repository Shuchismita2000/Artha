"""
Artha - Module 5: Composite Lead Priority Score
=================================================
Combines the Income/Affordability model + Intent/Propensity model into a
single ranked score, applies a simple product-fit rule, and backtests
precision@top-k against ACTUAL conversion (labels.converted_flag) --
this is the number that answers the ">30% conversion" requirement in the
problem statement.
"""

import json

import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

DATA_DIR = ".//data"
OUT_DIR = ".//outputs"


def product_fit_multiplier(row):
    """
    Simple, explainable business rule (not a model) connecting behavior
    patterns to loan-product fit. Kept as a rule, not learned, so it's
    directly editable by the bank's product team without retraining.
    """
    product = row["preferred_product"]
    if product == "home" and row["rent_to_income_ratio"] > 0.15:
        return 1.15  # currently paying rent + browsing home loans -> strong fit signal
    if product == "personal" and row["existing_loan_count"] == 0:
        return 1.05
    if product == "auto":
        return 1.0
    if product == "mortgage" and row["estimated_monthly_income"] > 100000:
        return 1.10
    return 1.0


def main():
    print("Loading model outputs...")
    income_est = pd.read_csv(f"{OUT_DIR}/score/income_estimates_all_customers.csv")
    intent_scores = pd.read_csv(f"{OUT_DIR}/score/intent_scores_all_customers.csv")
    affordability_feats = pd.read_csv(f"{OUT_DIR}/features/affordability_features.csv")
    customers = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                                  "shuchismitamallick/loan-underwriting-and-customer-behavior-dataset",
                                  "customer_master.csv")
    labels = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                                  "shuchismitamallick/loan-underwriting-and-customer-behavior-dataset",
                                  "labels.csv")

    df = (income_est
          .merge(intent_scores, on="customer_id", how="left")
          .merge(affordability_feats[["customer_id", "rent_to_income_ratio", "emi_to_income_ratio"]],
                 on="customer_id", how="left")
          .merge(customers[["customer_id", "existing_loan_count"]], on="customer_id", how="left")
          .merge(labels[["customer_id", "applied_flag", "converted_flag"]], on="customer_id", how="left"))

    # --- Normalize affordability into a 0-1 score ---
    # Higher estimated income + lower EMI burden = higher affordability.
    df["affordability_raw"] = df["estimated_monthly_income"] * (1 - df["emi_to_income_ratio"].clip(0, 1))
    df["affordability_score"] = (
        (df["affordability_raw"] - df["affordability_raw"].min())
        / (df["affordability_raw"].max() - df["affordability_raw"].min())
    )

    # --- Product fit rule ---
    df["product_fit_multiplier"] = df.apply(product_fit_multiplier, axis=1)

    # --- Composite score ---
    df["lead_priority_score"] = (
        df["intent_probability"] * df["affordability_score"] * df["product_fit_multiplier"]
    )

    # Rank
    df = df.sort_values("lead_priority_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["percentile"] = (1 - df["rank"] / len(df)) * 100

    print(f"\nTotal customers scored: {len(df)}")
    print(f"Overall base conversion rate: {df['converted_flag'].mean():.3f}")

    print("\n--- Precision@top-k against ACTUAL conversion (converted_flag) ---")
    for k in [0.05, 0.10, 0.20, 0.30]:
        n = int(len(df) * k)
        top_k = df.head(n)
        precision_k = top_k["converted_flag"].mean()
        applied_k = top_k["applied_flag"].mean()
        print(f"Top {int(k*100):>2}% ({n:>4} leads): conversion={precision_k:.3f}  application_rate={applied_k:.3f}")

    baseline = df["converted_flag"].mean()
    top20_precision = df.head(int(len(df) * 0.20))["converted_flag"].mean()
    lift = (top20_precision / baseline - 1) * 100
    print(f"\nTop-20% lift over random baseline: +{lift:.1f}%")
    print(f"Claim for pitch: 'Top-ranked 20% of leads convert at {top20_precision*100:.1f}% "
          f"vs {baseline*100:.1f}% baseline ({lift:.0f}% lift)'")

    # --- Save ranked leads (for the dashboard) ---
    output_cols = [
        "customer_id", "rank", "percentile", "lead_priority_score",
        "intent_probability", "affordability_score", "estimated_monthly_income",
        "preferred_product", "product_fit_multiplier", "occupation_type",
        "applied_flag", "converted_flag",
    ]
    df[output_cols].to_csv(f"{OUT_DIR}/ranked_leads.csv", index=False)
    print(f"\nSaved ranked_leads.csv: {df.shape[0]} rows")
    print(df[output_cols].head(10))
    # Convert column names to snake_case (optional, if needed)
    df[output_cols].columns = (
        df[output_cols].columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace(".", "", regex=False)
    )

    # Convert DataFrame to list of dictionaries
    records = df[output_cols].to_dict(orient="records")

    # Save as JSON
    with open(f"{OUT_DIR}/ranked_leads.json", "w") as f:
        json.dump(records, f, indent=4)

    print("JSON file saved as ranked_leads.json")


if __name__ == "__main__":
    main()
