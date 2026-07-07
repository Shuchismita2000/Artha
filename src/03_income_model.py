"""
Artha - Module 3: Income / Repayment-Capacity Model
=====================================================
Key design decision: we CANNOT train directly against a "true income" label,
because in the real world that's exactly what's unobserved for self-employed
and business-owner customers -- that's the whole problem.

Instead we do what a real income-verification model does:
  1. Train on SALARIED customers, where declared_annual_income is reliable
     (payslip-verifiable) -- this becomes our supervised label.
  2. Learn the mapping: cash-flow behavior features -> monthly income.
  3. Apply that learned mapping to self-employed / business-owner customers,
     whose declared income is unreliable, to get a more trustworthy estimate.
  4. Compare the model's estimate against their self-declared figure to
     surface over/under-statement -- this is the actual underwriting value-add.

Model: XGBoost regression. SHAP for explainability.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import joblib
import kagglehub
from kagglehub import KaggleDatasetAdapter

OUT_DIR = ".//outputs"
MODEL_DIR = ".//models"

FEATURE_COLS = [
    "avg_monthly_credit", "income_cv", "income_stability_index",
    "emi_to_income_ratio", "rent_to_income_ratio", "discretionary_to_income_ratio",
    "bounce_count", "window_dressing_flag", "total_txn_count", "avg_txn_amount",
    "count_salary", "count_business_income",
]


def main():
    print("Loading features + customer master...")
    feats = pd.read_csv(f"{OUT_DIR}/features/affordability_features.csv")
    customers = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                                  "shuchismitamallick/loan-underwriting-and-customer-behavior-dataset",
                                  "customer_master.csv")

    df = feats.merge(customers[["customer_id", "occupation_type", "declared_annual_income"]],
                      on="customer_id", how="left")
    df["declared_monthly_income"] = df["declared_annual_income"] / 12

    # --- Training set: salaried only (reliable label) ---
    train_df = df[df["occupation_type"] == "salaried"].copy()
    apply_df = df[df["occupation_type"] != "salaried"].copy()

    print(f"Training rows (salaried): {len(train_df)}")
    print(f"Apply rows (self-employed/business): {len(apply_df)}")

    X = train_df[FEATURE_COLS]
    y = train_df["declared_monthly_income"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    print(f"\n--- Holdout performance (salaried test set) ---")
    print(f"MAE:  Rs.{mae:,.0f}/month")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")

    # --- Apply to self-employed / business owners ---
    X_apply = apply_df[FEATURE_COLS]
    apply_df["estimated_monthly_income"] = model.predict(X_apply)
    apply_df["income_gap_pct"] = (
        (apply_df["estimated_monthly_income"] - apply_df["declared_monthly_income"])
        / apply_df["declared_monthly_income"] * 100
    )

    print(f"\n--- Self-employed / business income re-estimation ---")
    print(f"Median declared monthly income:  Rs.{apply_df['declared_monthly_income'].median():,.0f}")
    print(f"Median estimated monthly income: Rs.{apply_df['estimated_monthly_income'].median():,.0f}")
    print(f"Median gap (estimated vs declared): {apply_df['income_gap_pct'].median():.1f}%")
    understated = (apply_df["income_gap_pct"] > 15).mean() * 100
    print(f"% of self-employed customers understating income by >15%: {understated:.1f}%")

    # --- SHAP explainability ---
    print("\nComputing SHAP values (holdout sample)...")
    explainer = shap.TreeExplainer(model)
    shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(shap_sample)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop features driving income estimate (mean |SHAP|):")
    print(mean_abs_shap.round(2))

    # --- Save everything ---
    apply_df[["customer_id", "declared_monthly_income", "estimated_monthly_income", "income_gap_pct"]].to_csv(
        f"{OUT_DIR}/score/income_reestimation_self_employed.csv", index=False
    )

    # Score EVERY customer (salaried get model applied too, for consistency downstream in
    # the composite score -- their model estimate should closely match declared income anyway)
    df["estimated_monthly_income"] = model.predict(df[FEATURE_COLS])
    df[["customer_id", "occupation_type", "declared_monthly_income", "estimated_monthly_income"]].to_csv(
        f"{OUT_DIR}/income_estimates_all_customers.csv", index=False
    )

    joblib.dump(model, f"{MODEL_DIR}/income_model.pkl")
    mean_abs_shap.to_csv(f"{OUT_DIR}/shap/income_model_shap_importance.csv")

    print(f"\nSaved model to {MODEL_DIR}/income_model.pkl")
    print(f"Saved estimates to {OUT_DIR}/score/income_estimates_all_customers.csv")


if __name__ == "__main__":
    main()
