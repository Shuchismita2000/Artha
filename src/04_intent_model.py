"""
Artha - Module 4: Loan Intent / Propensity Model
==================================================
Predicts probability of loan application (applied_flag) from behavioral +
demographic features. LightGBM classifier, with SHAP explainability.

Target: labels.applied_flag (ground truth for backtesting/training only).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import shap
import joblib
import kagglehub
from kagglehub import KaggleDatasetAdapter

DATA_DIR = ".//data"
OUT_DIR = ".//outputs"
MODEL_DIR = ".//models"

NUMERIC_FEATURES = [
    "login_count", "calculator_use_count_30d", "calculator_use_count_total",
    "product_page_view_count_30d", "product_page_view_count_total",
    "customer_care_chat_count", "sms_click_count", "product_view_concentration",
    "days_since_last_intent_event", "avg_session_duration",
    "age", "existing_loan_count", "relationship_tenure_years", "bureau_score_proxy",
]
CATEGORICAL_FEATURES = ["preferred_product", "occupation_type", "city_tier", "residence_type"]


def main():
    print("Loading features + labels...")
    feats = pd.read_csv(f"{OUT_DIR}/features/intent_features.csv")
    labels = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                                  "shuchismitamallick/loan-underwriting-and-customer-behavior-dataset",
                                  "labels.csv")


    df = feats.merge(labels[["customer_id", "applied_flag", "converted_flag"]], on="customer_id", how="left")

    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = NUMERIC_FEATURES + [c + "_enc" for c in CATEGORICAL_FEATURES]

    X = df[feature_cols]
    y = df["applied_flag"]

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, df["customer_id"], test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    print(f"\n--- Holdout performance ---")
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    # Precision at top-k -- the number that matters for the "30% conversion" claim
    test_results = pd.DataFrame({"customer_id": ids_test, "y_true": y_test.values, "proba": proba})
    test_results = test_results.sort_values("proba", ascending=False)
    for k in [0.1, 0.2, 0.3]:
        n = int(len(test_results) * k)
        precision_at_k = test_results.head(n)["y_true"].mean()
        print(f"Precision@top-{int(k*100)}%: {precision_at_k:.3f}  (baseline rate: {y_test.mean():.3f})")

    # --- SHAP ---
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    print("\nTop features driving intent score (mean |SHAP|):")
    print(mean_abs_shap.round(4))

    # --- Score ALL customers for downstream composite score ---
    df["intent_probability"] = model.predict_proba(df[feature_cols])[:, 1]
    df[["customer_id", "intent_probability", "preferred_product"]].to_csv(
        f"{OUT_DIR}/intent_scores_all_customers.csv", index=False
    )

    joblib.dump(model, f"{MODEL_DIR}/intent_model.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/intent_encoders.pkl")
    mean_abs_shap.to_csv(f"{OUT_DIR}/shap/intent_model_shap_importance.csv")

    print(f"\nSaved model to {MODEL_DIR}/intent_model.pkl")
    print(f"Saved scores to {OUT_DIR}/score/intent_scores_all_customers.csv")


if __name__ == "__main__":
    main()
