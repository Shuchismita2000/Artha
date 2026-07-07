# Artha — Synthetic Dataset: Data Dictionary & Generation Rules

Generated for the IDBI Innovate PoC. Four linked tables, joined on `customer_id`.

| Table | Rows | Purpose |
|---|---|---|
| customer_master.csv | 5,000 | Demographic / KYC base profile |
| transaction_ledger.csv | 1,033,406 | 18 months of transaction history (Jan 2024 – Jun 2025) |
| behavioral_log.csv | 489,663 | 6 months of app/digital engagement (Jan 2025 – Jun 2025) |
| labels.csv | 5,000 | Ground truth for backtesting ONLY — never fed to models as an input feature |

---

## 1. customer_master.csv

| Field | Type | Description |
|---|---|---|
| customer_id | string | Unique ID, format `CUSTxxxxxx` |
| age | int | 21–60 |
| occupation_type | categorical | `salaried` (60%) / `self_employed` (25%) / `business_owner` (15%) |
| employment_tenure_months | int | Months in current job/business |
| city_tier | categorical | `metro` (40%) / `tier2` (35%) / `tier3` (25%) |
| declared_annual_income | float | Self-declared income (INR/year) |
| existing_loan_count | int | 0–4 active loans |
| existing_loan_types | string | `\|`-separated list, e.g. `home\|auto`; blank = none |
| residence_type | categorical | `owned` / `rented` |
| relationship_tenure_years | float | Years banking with IDBI |
| bureau_score_proxy | int | 300–900, adjusted down for customers with frequent bounce events |

**Business rules baked into generation:**
- Employment tenure scales with age; self-employed/business owners skew ~1.0–1.3x longer (family businesses).
- Income multiplier by city tier: metro ×1.4, tier2 ×1.0, tier3 ×0.75.
- **Declared income is a *noisy observation* of a hidden true income, not the truth itself.** Salaried customers: declared ≈ true income ±5% (payslip-verifiable). Self-employed/business owners: declared income is systematically **understated on average (~25% lower)** with much higher variance (±30%) — this gap is exactly what the income-estimation model is designed to close using transaction behavior instead of declared figures.
- Residence: probability of owning increases with age.
- Bureau score proxy is nudged down 25 points per bounce/overdraft event recorded for that customer.

---

## 2. transaction_ledger.csv

| Field | Type | Description |
|---|---|---|
| transaction_id | string | Unique ID, format `TXNxxxxxxxx` |
| customer_id | string | FK to customer_master |
| date | date | Transaction date (2024-01-01 to 2025-06-30) |
| amount | float | Signed amount — positive = credit, negative = debit |
| direction | categorical | `credit` / `debit` |
| channel | categorical | `NEFT` / `UPI` / `ECS` |
| narration_raw | string | Deliberately messy, templated bank narration (multiple formats per category to stress-test the parser) |
| true_category | categorical | **Ground truth only** — `salary`, `business_income`, `emi`, `rent`, `discretionary`, `bounce`, `window_dressing`. This column exists so we can measure the narration-parser's accuracy against ground truth; a production parser would have to infer this from `narration_raw` alone. |

**Category distribution generated:** discretionary 66%, business_income 19%, emi 5%, salary 5%, rent 4%, bounce 0.3%, window_dressing 0.01%.

**Business rules baked into generation:**
- **Salaried**: exactly 1 salary credit/month, date jitter ±3 days, amount = true monthly income ±5% noise.
- **Self-employed / business owner**: 3–8 irregular credits/month, month-to-month total volatility ±30% (deliberately volatile — a naive average would misjudge these customers; the income-stability feature is built to handle this correctly).
- **EMI debits**: fixed date (5th of month), fixed-ish amount (8–18% of monthly income per active loan), only generated for customers with `existing_loan_count > 0`.
- **Rent debits**: only for `residence_type == rented`, 20–30% of monthly income, ~3rd of month.
- **Discretionary spend**: Poisson-distributed UPI transactions, count and size scaled by income.
- **Bounce/overdraft events**: 2–5% probability per customer per month — a negative affordability signal.
- **Window-dressing pattern (injected deliberately)**: 3% of customers receive one artificial large one-time credit 3–7 days before the observation-window cutoff (1.5–3x monthly income). This exists specifically to demonstrate the income model does **not** get fooled by a single balance spike — a genuine robustness talking point for the pitch.

---

## 3. behavioral_log.csv

| Field | Type | Description |
|---|---|---|
| event_id | string | Unique ID, format `EVTxxxxxxxx` |
| customer_id | string | FK to customer_master |
| timestamp | datetime | Within Jan 2025 – Jun 2025 |
| event_type | categorical | `app_login` / `loan_calculator_used` / `product_page_view` / `customer_care_chat` / `sms_click` |
| product_viewed | categorical | `personal` / `home` / `auto` / `mortgage` / `none` (populated only for calculator/page-view/some chat events) |
| session_duration_seconds | int | Engagement depth |

**Event type distribution generated:** app_login 86%, loan_calculator_used 6%, product_page_view 6%, sms_click 0.9%, customer_care_chat 0.8%.

**Business rules baked into generation:**
- Every customer gets a baseline of app logins (Poisson, rate increases mildly with underlying intent — realistic since even browsing customers use the app for other reasons).
- A **latent intent score** (0–1, right-skewed — most customers are low-intent) drives calculator-use and product-page-view frequency, **clustered in the final 30 days** of the window (recency signal).
- Each intent-driven customer concentrates activity on **one preferred product**, simulating a real customer researching one specific loan type rather than browsing everything.
- Customer care chat / SMS click events are sparse noise, weighted slightly by intent.

**Important design note:** the latent intent score itself is *not* in this file — only the resulting event counts are. Downstream models must learn intent from the observable event pattern, not from a hidden variable, avoiding trivial label leakage.

---

## 4. labels.csv (ground truth — backtesting only)

| Field | Type | Description |
|---|---|---|
| customer_id | string | FK |
| applied_flag | binary | Did the customer apply for a loan in the follow-up window |
| converted_flag | binary | Was the loan disbursed (0 if not applied) |
| loan_type | categorical | Product applied for (`none` if not applied) |
| application_amount | float | Amount applied for (0 if not applied) |

**Observed rates in this run:** application rate 39.8%, conversion rate 31.1%.

**Business rules baked into generation (label-leakage safeguard):**
- `applied_flag` probability = weighted combination of latent intent (55%) + latent affordability (10%) + baseline (15%) + **random noise**, then thresholded stochastically — not a deterministic rule.
- Latent affordability = `1 / (1 + income_coefficient_of_variation)`, computed from the *actual* generated transaction volatility — i.e., derived independently from the transaction ledger, not copy-pasted from any single input field.
- `converted_flag` (conditional on applied) is weighted more heavily toward affordability (65%) than intent (15%) — reflecting that once someone applies, whether the loan disburses depends more on real repayment capacity than on how interested they were.
- Both probabilities include independent Gaussian noise (σ=0.15), so labels are **not perfectly recoverable** from any single feature — models have to combine multiple engineered signals, same as they would need to on real data.
- `application_amount` is a rough multiple (8–30x) of the customer's *true* hidden monthly income, not their declared income — consistent with how real underwriting sizing works off verified income.

---

## Downstream usage map

| Model | Primary input table(s) | Key engineered features |
|---|---|---|
| Income / affordability model | transaction_ledger (aggregated) | Income stability index (CV of monthly credits), EMI-to-income ratio, bounce count (trailing 90d), debit-to-credit ratio |
| Intent / propensity model | behavioral_log + customer_master | Calculator-use count (30d), page-view recency, product-view concentration, relationship tenure |
| Composite lead score | Outputs of both models | intent_score × affordability_score × product-fit multiplier |

## What's real vs. simulated (stated for reviewer transparency)
- Calendar windows, business-rule ratios (EMI %, rent %, bounce probability) are informed estimates, not sourced from IDBI's actual portfolio — this is explicitly a **methodology demonstration**, to be recalibrated against real sandbox data once available.
- Labels are generated from latent scores with intentional noise — precision@top-k results on this dataset validate the *scoring logic*, not a guaranteed real-world conversion number.
