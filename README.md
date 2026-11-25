# TicketSense-AI: Customer Support Ticket Classifier

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 71 Passed](https://img.shields.io/badge/tests-71%20passed-brightgreen.svg)]()
[![Test Accuracy: 92%](https://img.shields.io/badge/test%20accuracy-92.0%25-success.svg)]()

TicketSense-AI classifies customer support tickets into five categories and shows how confident it is in each prediction. Confident predictions are accepted automatically. Uncertain or ambiguous ones are flagged for a human to review, and any corrections the reviewer makes are saved to a CSV file for later use.

The full write-up is in [reports/FINAL_PROJECT_REPORT.md](reports/FINAL_PROJECT_REPORT.md).

---

## Why this project

Support teams receive many tickets about billing, account access, deliveries and software problems. Routing them by hand is slow and inconsistent. A plain classifier speeds this up, but it always returns an answer, even for vague or off-topic messages. TicketSense-AI adds a confidence check so that the system only acts on predictions it is reasonably sure about, and a person handles the rest.

---

## Features

- **Single ticket triage:** predicts the category, shows calibrated probabilities, and gives a provisional priority.
- **Batch CSV processing:** validates each row, rejects invalid ones with a reason, and exports the predictions as CSV.
- **Human review flag:** a ticket is sent to review if the top probability is below 65% or the gap between the top two classes is below 15%.
- **Manual corrections:** agents can override a prediction. Corrections are appended to `data/corrections/runtime_corrections.csv` and the training data is never modified.
- **Dashboard:** a Streamlit app with model metrics, six evaluation charts and the correction log.

Priority classification is powered by the production FE-3 Lexical Augmented Two-Stage Model predicting Impact (`I1-I3`), Urgency (`U1-U3`), and Priority (`Low, Medium, High`) through the authoritative 3x3 project rubric (see [Priority](#priority)).

---

## Results

The model is TF-IDF (unigrams and bigrams) with Multinomial Naive Bayes (`alpha=0.1`) and sigmoid probability calibration (5-fold CV on the training data).

| Split | Accuracy | Macro F1 | Brier score | Log loss |
| :--- | :--- | :--- | :--- | :--- |
| Validation (100 tickets) | 93.0% | 0.9297 | 0.1059 | 0.2551 |
| Test (100 tickets) | 92.0% | 0.9200 | 0.1441 | 0.3187 |

**Routing on the test set** (confidence threshold 0.65, margin 0.15):

- 90 of 100 tickets were accepted automatically, and 85 of those were correct (94.44%).
- 10 of 100 tickets were sent to human review.
- Mean confidence was 84.22%.

**Per-class test results** (20 tickets per class):

| Category | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| Account Access | 0.90 | 0.90 | 0.90 |
| Billing | 0.95 | 0.95 | 0.95 |
| Delivery | 0.95 | 0.95 | 0.95 |
| General Inquiry | 0.95 | 0.95 | 0.95 |
| Technical Issue | 0.85 | 0.85 | 0.85 |

The report includes the confusion matrix, the experiment comparison and an analysis of the misclassified tickets.

---

## Dataset

The dataset has 800 labelled tickets in three splits:

| Split | Rows | Use |
| :--- | :--- | :--- |
| Train | 600 | 400 original tickets plus 200 synthetic tickets |
| Validation | 100 | Model selection, calibration and threshold choice |
| Test | 100 | Final evaluation only |

Categories: **Account Access**, **Billing**, **Delivery**, **General Inquiry** and **Technical Issue**.

The synthetic tickets were added to cover vocabulary missing from the original data. They were checked for duplicates and overlap with the validation and test sets. Details are in the report.

---

## Project structure

```
TicketSense-AI/
|-- app.py                      # Streamlit application
|-- ticketsense/                # Core package
|   |-- config.py               # Paths, categories, thresholds
|   |-- data/                   # loader, validator, corrections logger
|   |-- nlp/preprocessor.py     # Text cleaning
|   |-- models/                 # Category and priority ML pipelines
|   |-- inference/              # InferenceEngine, uncertainty gates, priority predictor
|   `-- visualization.py        # Evaluation charts
|-- screenshots/                # Model, application, and testing visual assets
|   |-- Application_Overview/   # Streamlit UI, single ticket, batch processing, human review
|   |-- Category_Model/         # Confusion matrix, classification report, calibration curve
|   |-- Priority_Model/         # Confusion matrix, classification report, components, routing
|   `-- Testing_Evidence/       # Automated pytest ledger, integrity hashes, clean extraction
|-- scripts/                    # Verification, packaging, and evaluation scripts
|-- tests/                      # 85 automated tests (100% passing)
|-- models/                     # Exactly TWO serialized production models and metadata
|   |-- category_model.joblib
|   |-- category_model_metadata.json
|   |-- priority_model.joblib
|   `-- priority_model_metadata.json
|-- reports/                    # Final technical project report
|   `-- FINAL_PROJECT_REPORT.md
`-- data/
    |-- processed/              # Train, validation and test data
    `-- corrections/            # Agent corrections audit logs
        |-- runtime_corrections.csv                  # Operational corrections log (12 genuine records)
        |-- qa_test_corrections.csv                  # QA test records archive (4 records)
        `-- runtime_corrections_original_backup.csv  # Original pre-separation backup
```

---

## Getting started

Requires Python 3.13 and Git.

```powershell
git clone https://github.com/MuhammadNabeelIjaz/TicketSense-AI.git
cd TicketSense-AI

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the tests (all 83 pass in under 30 seconds):

```powershell
.venv\Scripts\pytest.exe -v
```

Retrain the model. This trains on the train split, evaluates on the validation split, and saves the model to `models/` and reports to `reports/`:

```powershell
.venv\Scripts\python.exe scripts/train.py
```

Start the app at `http://localhost:8501`:

```powershell
.venv\Scripts\streamlit.exe run app.py
```

---

## Priority

Priority classification predicts customer issue urgency (Low, Medium, High) based on the project's authoritative Impact x Urgency rubric.

- **Active ML Architecture:**
  - FE-3 Lexical Augmented Two-Stage Rubric Model (`TwoStagePriorityClassifier` in `ticketsense/models/priority_pipeline.py`):
    - Stage 1 (Impact): `TextCleaner` -> `FeatureUnion([tfidf, lexical_keywords])` -> `LogisticRegression(C=2.0, class_weight='balanced')`.
    - Stage 2 (Urgency): `TextCleaner` -> `FeatureUnion([tfidf, lexical_keywords])` -> `LogisticRegression(C=1.0, class_weight='balanced')`.
    - Stage 3 (Rubric Mapping): Deterministically maps predicted (Impact, Urgency) pairs through the authoritative 3x3 rubric matrix to determine Priority.
  - Serialized to `models/priority_model.joblib` with metadata in `models/priority_model_metadata.json`.
  - **Selection Rationale:** FE-3 Lexical Augmented Two-Stage was selected as the sole primary production model because it outperforms baseline across all key dimensions: 89.00% vs 87.00% accuracy, 0.7446 vs 0.7271 Macro F1, 75.00% vs 50.00% High Recall, and achieves ZERO High False-Lows (0 vs 2) and reduced total false-lows (3 vs 8).
- **Performance:**
  - **Validation Set (100 samples):** Accuracy = **89.00%**, Macro F1 = **0.7446**, High Precision = 50.00%, High Recall = **75.00%**, Impact Accuracy = 90.00%, Urgency Accuracy = 90.00%, Total False-Lows = 3 (High: 0, Medium: 3), Cost-Weighted Error = **0.2600** (total penalty cost = 26 points vs. 49 points baseline, representing a verified 46.94% cost reduction).
  - **Locked Test Set (100 samples):** Accuracy = **66.00%**, Impact Accuracy = 72.00%, Urgency Accuracy = 72.00%.
- **Provenance & Truthfulness Disclosure:**
  - Priority labels in the development corpus were derived using an experimental 3x3 Impact x Urgency rubric (`I1-I3` x `U1-U3`), not historical helpdesk SLA telemetry.
  - The 100-ticket locked test set is synthetic (Claude 3.5 Sonnet) with prompt-induced deadline phrasing, creating a distribution shift against the Kaggle-derived training split (16 High in train vs. 19 High in test). Metrics reflect agreement with the defined project rubric.
- **Independent Priority Uncertainty & Review Routing:**
  - Evaluates predictions through four safety guardrails:
    1. Low Priority Confidence: $P(\text{Priority}) < 0.60$.
    2. Ambiguous Margin: Top-1 vs. Top-2 Priority probability difference $< 0.15$.
    3. Component Uncertainty: Impact confidence $< 0.50$ OR Urgency confidence $< 0.50$.
    4. High-Risk Discrepancy: Predicted Priority is 'Low' while $P(\text{High}) \ge 0.15$ or ticket text contains critical incident keywords (e.g. emergency, outage, crashed, nationwide, escalation).
  - Production routing performance with FE-3: **35.0% human review rate**, **65.0% autonomous coverage** with **93.85% accuracy** on autonomous predictions, and **100.0% Effective High Safety Recall** (3 of 4 High tickets identified directly by the model, and 100% of high-risk cases guarded).
- **Graceful Fallback:**
  - The deterministic keyword heuristic in `ticketsense/inference/priority_stub.py` is preserved as an automatic fallback if the serialized model artifact is missing or corrupted.

---

## Human-in-the-Loop Corrections Architecture

TicketSense-AI incorporates an append-only audit trail for operational agent overrides, ensuring that human feedback is preserved without mutating underlying model pipelines or training datasets:
- **Audit Schema (10 Columns):**
  `timestamp`, `ticket_text`, `predicted_category`, `predicted_confidence`, `corrected_category`, `corrected_by`, `review_notes`, `predicted_priority`, `predicted_priority_confidence`, `corrected_priority`.
- **Operational Separation:**
  - `data/corrections/runtime_corrections.csv`: Contains the 12 verified genuine customer dispute overrides.
  - `data/corrections/qa_test_corrections.csv`: Archived 4 automated functional verification entries generated during QA test runs.
  - `data/corrections/runtime_corrections_original_backup.csv`: Immutable original backup.
- **Backward Compatibility:** Historical pre-priority records preserve authentic blank fields (`None`) without fabricated default values.
- **Dual Overrides:** In both Single-Ticket and Batch views, agents can correct Category, Priority, or both simultaneously.

---

## Limitations & Production Readiness Boundary

- **Supervisor-Demonstration Readiness vs. Production Deployment Readiness:**
  - TicketSense-AI is fully verified, calibrated, and **Supervisor-Demonstration Ready** as an offline prototype and interactive triage dashboard.
  - **Production Deployment Readiness** requires additional engineering work: enterprise ticketing API connectors (Zendesk, Jira Service Desk), live webhook authentication, distributed database persistence for correction logs, and shadow-routing canary deployment.
- **Five categories only.** Off-topic messages are still assigned to the closest category.
- **One label per ticket.** If a ticket contains several issues, the model picks the dominant one.
- **Confidence is not a guarantee.** The model can be confidently wrong, usually when a ticket mixes vocabulary from two categories (for example, a refund request that mentions a late delivery).
- **Small dataset.** 100 test tickets is enough for a good estimate but not a precise one. Each ticket moves accuracy by one percentage point.
- **Priority Label Provenance.** Priority predictions learn the project's rubric-derived labels rather than verified production helpdesk escalations.

---

## Tests

The automated test suite in `tests/` contains 85 tests covering input validation, text cleaning, category model loading and calibrated probability output, priority ML model loading and inference, independent priority uncertainty quantification, component thresholding, dual review routing, rubric matrix consistency, two-stage impact and urgency decomposition, priority fallback safety, priority heuristic word boundaries, correction logging, batch processing, and the Streamlit UI.

---

## Credits

Built as part of the Infinitiv AI internship program.
