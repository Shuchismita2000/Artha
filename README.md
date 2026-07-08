# Artha — Retail Lending Lead Scoring & Income Assessment

PoC for IDBI Innovate Hackathon. Combines an income/repayment-capacity model
and a loan-intent propensity model into a single explainable Lead Priority
Score, backtested against a synthetic ground truth.

## Headline result

**Top-ranked 20% of leads convert at 50.8%, vs. 31.1% baseline — a 63% lift**,
comfortably clearing the >30% conversion target in the problem statement.
(See `outputs/ranked_leads.csv` and the console output of `05_composite_score.py`.)

## Project structure

```text
Artha/
├── app/                                       # Frontend web application
│   ├── artha_landing_page.html                # Project landing page
│   ├── artha_rm_dashboard.html                # Interactive Relationship Manager dashboard
│   └── artha_rm_eda_dashboard.html            # Exploratory Data Analysis (EDA) dashboard
│
├── data/                                      # Sample datasets and documentation
│   ├── behavioral_log_sample.csv              # Sample customer behavioral events
│   ├── customer_master_sample.csv             # Sample customer master records
│   ├── DATA_DICTIONARY.md                     # Description of dataset schema and fields
│   ├── FEATURE_ENGINEERING_DATA_DICTIONARY.md # Engineered feature descriptions
│   ├── labels_sample.csv                      # Sample ground-truth loan labels (backtesting only)
│   └── transaction_ledger_sample.csv          # Sample banking transaction history
│
├── models/                                    # Trained ML model artifacts (.pkl)
│
├── outputs/                                   # Generated outputs from the ML pipeline
│   ├── explainability/                        # Feature contribution reports
│   ├── features/                              # Engineered feature datasets
│   ├── ranked_leads.csv                       # Final prioritized lead list
│   ├── ranked_leads.json                      # Dashboard-ready lead data
│   ├── score/                                 # Income, intent, and composite scores
│   ├── shap/                                  # SHAP explainability outputs
│   └── transaction_ledger_parsed.csv          # Parsed and categorized transactions
│
├── src/                                       # Machine Learning pipeline
│   ├── 01_narration_parser.py                 # Rule-based transaction narration parser
│   ├── 02_feature_engineering.py              # Builds affordability & intent features
│   ├── 03_income_model.py                     # XGBoost income estimation model
│   ├── 04_intent_model.py                     # LightGBM loan intent prediction model
│   ├── 05_composite_score.py                  # Generates explainable lead priority score
│   └── 06_shap_explanations.py                # Produces SHAP explanations for predictions
│
├── index.html                                 # Redirects to the live application
├── README.md                                  # Project documentation
└── requirements.txt                           # Python dependencies
```
## 📂 Dataset Availability

To keep this repository lightweight and easy to navigate, only **sample datasets** are included here for demonstrating the project structure, schema, and workflows.

The **complete synthetic dataset** used for model development and experimentation—including:

- 5,000 customer profiles
- 1M+ banking transactions
- 489K+ behavioral events
- Ground-truth loan application labels

is publicly available on Kaggle.

👉 **Download the full dataset from Kaggle**

https://www.kaggle.com/datasets/shuchismitamallick/loan-underwriting-and-customer-behavior-dataset

## 🌐 Live Demo

Experience the interactive RM Dashboard directly in your browser.

**Live Website:**  
👉 https://shuchismita2000.github.io/Artha

The dashboard demonstrates:

- AI-powered lead prioritization
- Income estimation and affordability assessment
- Loan intent prediction
- Explainable lead scoring
- Interactive "Ask Artha" assistant
- Customer insights and recommendations

No installation is required to explore the dashboard.

## 🎥 Demo Video

A 3-minute walkthrough of the complete solution is available here:

👉 [https://DEMO_VIDEO_LINK](https://github.com/Shuchismita2000/Artha/blob/main/Artha%20Demo%20Video.mp4)

## Design decisions

1. **No LLM/API calls anywhere in the core pipeline.** Narration parsing is
   rules-based (regex), both prediction models are gradient-boosted trees.
   This is deliberate: cheaper at scale, deterministic, and defensible to a
   bank's compliance function in a way an LLM's internal reasoning isn't.

2. **The income model is trained on salaried customers only** (where declared
   income is payslip-verifiable, i.e. a trustworthy label), then *applied* to
   self-employed/business-owner customers, whose declared income is
   unreliable. The model surfaces a median 17.8% income understatement for
   that segment — this is the actual underwriting value-add, not just a
   number for its own sake.

3. **Labels were generated from latent scores + independent noise**, not
   copied from any single input feature — so the models had to learn
   genuine patterns rather than exploit a leak. This is worth stating
   explicitly to reviewers: the ~60% AUC on the intent model reflects a
   realistically noisy label, not a weak model.

4. **SHAP explainability on both models** — every score is traceable to the
   features that drove it, which is what makes this auditable to
   underwriting/compliance rather than a black box.

5. **Product-fit multiplier is a rule, not a learned parameter** — kept
   editable by the bank's product team without retraining, e.g. "renting +
   browsing home loans" boosts fit for a home loan.

## What's next (explicitly deferred, not built)

- Account Aggregator integration replacing the simulated transaction feed
- Bureau data fusion alongside behavioral/transactional scoring
- GenAI-assisted SHAP-to-plain-English explanation notes for RMs (only once
  the numeric pipeline above is validated on real sandbox data)
