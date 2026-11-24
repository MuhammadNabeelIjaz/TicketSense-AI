# TicketSense-AI: Customer Support Ticket Classifier

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 45 Passed](https://img.shields.io/badge/tests-45%20passed-brightgreen.svg)]()
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

Priority is currently a keyword-based heuristic, not a trained model (see [Priority](#priority)).

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
|   |-- models/                 # Pipeline, evaluation, model saving/loading
|   |-- inference/              # Prediction engine, uncertainty check, priority heuristic
|   `-- visualization.py        # Evaluation charts
|-- scripts/                    # Training, experiments, evaluation and QA scripts
|-- tests/                      # 45 automated tests
|-- models/                     # Saved model and metadata
|-- reports/                    # Final report, experiment log, charts
`-- data/
    |-- processed/              # Train, validation and test data
    `-- corrections/            # Agent corrections log
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

Run the tests (all 45 pass in under 20 seconds):

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

Priority prediction is an optional extension in the project brief, and supervised training for it is paused. Only 6.5% of the historical priority labels are High, and 26 rows are still waiting for review in `data/processed/dataset_review_required.csv`. Training on these labels would likely under-predict urgent tickets such as security incidents. Until the Technical Supervisor (Abdullah Ramzan) approves this at Gate G1, the app shows a keyword-based indicator (Low, Medium or High) as a hint only.

---

## Limitations

- **Five categories only.** Off-topic messages are still assigned to the closest category.
- **One label per ticket.** If a ticket contains several issues, the model picks the dominant one.
- **Confidence is not a guarantee.** The model can be confidently wrong, usually when a ticket mixes vocabulary from two categories (for example, a refund request that mentions a late delivery).
- **Small dataset.** 100 test tickets is enough for a good estimate but not a precise one. Each ticket moves accuracy by one percentage point.

---

## Tests

The tests in `tests/` cover input validation, text cleaning, model loading and probability output, uncertainty flags, correction logging, batch processing, the Streamlit UI (single and batch flows) and chart generation.

---

## Credits

Built as part of the Infinitiv AI internship program.
