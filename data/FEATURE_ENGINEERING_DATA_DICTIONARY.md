# Feature Engineering Data Dictionary

## Overview

This document describes the engineered features created from transaction and behavioral datasets for loan underwriting and customer intent modeling.

---

# 1. Affordability Features

These features are engineered from bank transaction data to estimate repayment capacity, financial stability, spending behavior, and affordability.

| Feature | Formula / Calculation | Purpose | Business Use |
|---|---|---|---|
| customer_id | Unique customer identifier | Links customer records | Join key |
| avg_monthly_credit | Mean of monthly credit totals | Estimate monthly income | Repayment capacity |
| income_cv | std_monthly_credit / avg_monthly_credit | Measure income variability | Income stability assessment |
| income_stability_index | 1 / (1 + income_cv) | Stability score (higher is better) | Creditworthiness |
| emi_to_income_ratio | abs(total_emi) / total_credit | Existing debt burden | Affordability assessment |
| rent_to_income_ratio | abs(total_rent) / total_credit | Housing expense burden | Fixed obligation estimation |
| discretionary_to_income_ratio | abs(total_discretionary) / total_credit | Lifestyle spending relative to income | Spending discipline |
| bounce_count | Count of bounce transactions | Failed payment history | Financial distress indicator |
| window_dressing_flag | 1 if window dressing exists else 0 | Detect temporary balance inflation | Fraud/risk indicator |
| total_txn_count | Count of all transactions | Banking activity level | Behavioral signal |
| avg_txn_amount | Mean absolute transaction amount | Typical transaction size | Spending profile |
| count_salary | Count of salary transactions | Salary frequency | Stable income indicator |
| count_business_income | Count of business income transactions | Business income frequency | Self-employed income indicator |

### Intermediate Features

| Feature | Formula | Purpose |
|---|---|---|
| monthly_credit | Monthly sum of credit transactions | Monthly income estimation |
| std_monthly_credit | Standard deviation of monthly credits | Income volatility |
| n_active_months | Months with at least one credit | Observation duration |
| total_credit | avg_monthly_credit × n_active_months | Approximate observed income |
| total_emi | Sum of EMI transactions | Debt calculation |
| total_rent | Sum of rent transactions | Housing expense calculation |
| total_discretionary | Sum of discretionary spending | Lifestyle expense calculation |
| count_window_dressing | Count of window dressing events | Fraud detection |

---

# 2. Customer Intent Features

These features are engineered from digital behavioral logs to estimate customer engagement, product interest, and loan application intent.

| Feature | Formula / Calculation | Purpose | Business Use |
|---|---|---|---|
| customer_id | Unique customer identifier | Links customer records | Join key |
| login_count | Count of app_login events | App engagement | Customer activity |
| calculator_use_count_30d | Loan calculator usage in last 30 days | Recent borrowing intent | Conversion prediction |
| calculator_use_count_total | Total loan calculator usage | Overall interest | Intent scoring |
| product_page_view_count_30d | Product page views in last 30 days | Recent exploration | Purchase readiness |
| product_page_view_count_total | Total product page views | Overall product interest | Product affinity |
| customer_care_chat_count | Count of customer care chats | Information seeking | Assisted conversion |
| sms_click_count | Count of SMS clicks | Marketing responsiveness | Campaign effectiveness |
| preferred_product | Most frequently viewed product | Product preference | Personalization |
| product_view_concentration | Max product views / Total intent events | Focus of interest | Intent strength |
| days_since_last_intent_event | Days since most recent intent event | Engagement recency | Propensity modeling |
| avg_session_duration | Mean session duration | User engagement | Customer quality signal |

### Intermediate Feature

| Feature | Formula | Purpose |
|---|---|---|
| days_before_end | Observation End Date − Event Timestamp | Enables recency-based features |

---

# Feature Engineering Objectives

The engineered features are designed to support machine learning models for:

- Loan underwriting
- Credit risk assessment
- Affordability analysis
- Loan propensity prediction
- Customer intent modeling
- Personalized product recommendation
- Fraud and anomaly detection
- Marketing campaign optimization

## Feature Groups

### Affordability
- Income estimation
- Income stability
- Debt burden
- Fixed obligations
- Spending behaviour
- Banking activity
- Fraud indicators

### Customer Intent
- Digital engagement
- Loan exploration
- Product preference
- Marketing responsiveness
- Engagement recency
- Conversion propensity

Together, these feature groups provide a comprehensive representation of a customer's financial health, repayment capacity, behavioral patterns, and purchase intent for downstream machine learning models.
