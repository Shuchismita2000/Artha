# Artha — Retail Lending Lead Scoring & Income Assessment

PoC for IDBI Innovate Hackathon. Combines an income/repayment-capacity model
and a loan-intent propensity model into a single explainable Lead Priority
Score, backtested against a synthetic ground truth.

## Headline result

**Top-ranked 20% of leads convert at 50.8%, vs. 31.1% baseline — a 63% lift**,
comfortably clearing the >30% conversion target in the problem statement.
(See `outputs/ranked_leads.csv` and the console output of `05_composite_score.py`.)

## Project structure

```
artha_project/
├── data/                          synthetic input tables (see DATA_DICTIONARY.md)
│   ├── customer_master.csv
│   ├── transaction_ledger.csv
│   ├── behavioral_log.csv
│   └── labels.csv                 (ground truth — backtest only, never a model input)
├── src/
│   ├── 01_narration_parser.py     rules-based transaction categorization
│   ├── 02_feature_engineering.py  builds affordability + intent feature sets
│   ├── 03_income_model.py         XGBoost regression: income/repayment capacity
│   ├── 04_intent_model.py         LightGBM classification: loan intent propensity
│   └── 05_composite_score.py      combines both models, backtests precision@top-k
├── models/                         saved .pkl model artifacts
├── outputs/                        all generated features, scores, and the dashboard
│   └── artha_rm_dashboard.html    standalone RM-facing lead console (open in any browser)
└── README.md                      this file
```

## How to run (in order)

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap joblib --break-system-packages

cd src
python3 01_narration_parser.py       # -> outputs/transaction_ledger_parsed.csv
python3 02_feature_engineering.py    # -> outputs/affordability_features.csv, intent_features.csv
python3 03_income_model.py           # -> models/income_model.pkl, outputs/income_estimates_all_customers.csv
python3 04_intent_model.py           # -> models/intent_model.pkl, outputs/intent_scores_all_customers.csv
python3 05_composite_score.py        # -> outputs/ranked_leads.csv
```

Then open `outputs/artha_rm_dashboard.html` directly in a browser — no server
needed, all lead data is embedded.

## Design decisions worth highlighting in the pitch

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
