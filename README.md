# TicketSense-AI: Smart Customer Support Ticket Classifier

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 37 Passed](https://img.shields.io/badge/tests-37%20passed-brightgreen.svg)]()
[![Val Accuracy: 93%](https://img.shields.io/badge/val%20accuracy-93.0%25-blueviolet.svg)]()
[![Test Accuracy: 92%](https://img.shields.io/badge/test%20accuracy-92.0%25-success.svg)]()
[![Priority Model: Paused](https://img.shields.io/badge/priority%20training-PAUSED%20(Gate%20G1)-orange.svg)]()

TicketSense-AI is a modular, production-engineered machine learning system designed to automatically classify customer support tickets into operational categories, quantify prediction uncertainty, triage high-risk messages, and support human-in-the-loop review without mutating underlying historical datasets.

---

## Canonical Project Documentation

The complete, evidence-based project report with full methodology, validation logs, independent audit verification, and deployment recommendations is available in:
- [FINAL_PROJECT_REPORT.md](reports/FINAL_PROJECT_REPORT.md)

Historical handover documents, project briefs, and milestone tracking records are preserved in:
- [docs/archive/](docs/archive/)

---

## Project Overview

Incoming customer support tickets are frequently ambiguous, misspelled, or operationally sensitive. TicketSense-AI provides:
- Real-Time Single Ticket Triage: Text normalization, category prediction, confidence scoring, and issue summarization.
- Batch CSV Processing: High-throughput prediction on uploaded datasets with interactive filtering and downloadable enriched CSVs.
- Uncertainty Quantification: Flags low-confidence predictions (< 65%) and ambiguous class margins (< 15%) for mandatory human verification.
- Auditable Human Correction Logging: Support agent overrides are appended to an immutable audit log (`data/corrections/runtime_corrections.csv`) to enable active learning without altering historical training records.
- Governance Guardrails: Priority machine learning training is explicitly paused pending Gate G1 supervisor sign-off due to class imbalance (6.5% High priority) and active review queues.

---

## System Architecture

```
TicketSense-AI/
|-- app.py                         # Production Streamlit web application
|-- pytest.ini                     # Pytest configuration with pythonpath
|-- requirements.txt               # Pinned dependencies
|-- ticketsense/                   # Core modular package
|   |-- config.py                  # Project paths, taxonomy, and thresholds
|   |-- data/                      # Ingestion, validation, and corrections
|   |   |-- loader.py              # Split loader (train, validation, test)
|   |   |-- validator.py           # Text & batch CSV schema validation
|   |   `-- corrections.py         # Append-only human correction logger
|   |-- nlp/                       # Natural language processing
|   |   `-- preprocessor.py        # Text cleaning, contraction expansion, normalization
|   |-- models/                    # ML pipeline & evaluation
|   |   |-- category_pipeline.py   # TF-IDF + Classifier pipeline builder
|   |   |-- evaluator.py           # Metrics, confusion matrix, and error analysis
|   |   `-- persistence.py         # Joblib serialization with JSON metadata
|   `-- inference/                 # Inference runtime
|       |-- engine.py              # Unified single & batch inference engine
|       |-- uncertainty.py         # Decision uncertainty quantification
|       `-- priority_stub.py       # Priority governance stub (PAUSED per Gate G1)
|-- scripts/                       # Training, optimization, and evaluation CLI tools
|   |-- train.py                   # CLI runner for reproducible model training
|   |-- generate_synthetic_expansion.py # Balanced training data expansion generator
|   |-- validate_synthetic_expansion.py # Leakage & quality validation script
|   |-- run_optimization_experiments.py # Multi-model optimization & calibration runner
|   |-- evaluate_threshold_tradeoffs.py # Routing trade-off analysis script
|   |-- evaluate_final_test_set.py      # Untouched benchmark test evaluator
|   `-- run_qa_manual_checklist.py      # 10-point automated QA test runner
|-- tests/                         # Comprehensive automated test suite (37 tests)
|   |-- conftest.py                # Pytest fixtures
|   |-- test_app_ui.py             # Streamlit AppTest headless UI tests
|   |-- test_batch.py              # Batch processing tests
|   |-- test_corrections.py        # Corrections logging & immutability tests
|   |-- test_pipeline.py           # Model loading, output shapes, calibration
|   |-- test_preprocessing.py      # Contractions, URLs, emails, normalization
|   |-- test_uncertainty.py        # Low-confidence & ambiguity flags
|   `-- test_validation.py         # String boundaries & schema validation
|-- models/                        # Serialized model artifacts
|   |-- category_model.joblib      # Production Calibrated Multinomial NB artifact
|   `-- category_model_metadata.json # Hyperparameters, timestamps, metrics
|-- reports/                       # Validation & test evaluation artifacts
|   |-- FINAL_PROJECT_REPORT.md    # Single canonical final project report
|   |-- CATEGORY_MODEL_EVALUATION.md # Validation split evaluation breakdown
|   |-- EXPERIMENT_RESULTS_LOG.json # Machine-readable benchmark records
|   |-- category_confusion_matrix.png # Confusion matrix plot
|   |-- category_validation_errors.csv # Validation error analysis
|   `-- visualizations/            # Dedicated production visualization folder (6 chart PNGs)
|-- docs/                          # Project documentation and milestones
|   |-- README.md                  # Documentation index
|   |-- archive/                   # Archived historical notes, audit logs, and legacy reports
|   |   |-- final_independent_audit/ # Independent verification audit logs
|   |   |-- MODEL_OPTIMIZATION_EXPERIMENTS.md # Historical optimization experiment log
|   |   |-- HANDOVER.md
|   |   |-- HANDOVER_2026-11-23_LEGACY.md
|   |   `-- legacy_project_parameters.txt
|   `-- 01_Project_Brief/ ...      # Course briefs and planning artifacts
`-- data/
    |-- processed/                 # Immutable datasets (600 train, 100 val, 100 test)
    `-- corrections/               # Runtime agent overrides log
```

---

## Dataset and Category Taxonomy

The operational dataset contains 800 total customer support records partitioned into three distinct splits:
- Train (600 rows): Supervised model fitting (400 original verified records + 200 high-diversity synthetic additions).
- Validation (100 rows): Hyperparameter tuning, calibration fitting, and threshold selection.
- Test (100 rows): Isolated benchmark set reserved exclusively for final unblinded evaluation.

### Approved Category Classes
1. Account Access: Login failures, password resets, 2FA errors, MFA lockouts.
2. Billing: Invoices, charges, unauthorized debits, refund disputes, subscription fees.
3. Delivery: Shipment tracking, damaged goods, carrier delays, missing parcels.
4. General Inquiry: Business hours, product capabilities, policy questions.
5. Technical Issue: Software bugs, HTTP 500 errors, UI rendering glitches, crashing.

---

## Model Architecture and Evaluation Results

The production category classifier utilizes sublinear TF-IDF n-grams (1, 2) coupled with Multinomial Naive Bayes (`alpha=0.1`) and Sigmoid Probability Calibration (5-fold cross-validation).

### Summary Performance Metrics

| Evaluation Split | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Validation Set (100 samples) | 93.00% | 93.03% | 93.00% | 92.97% | 0.1382 |
| Untouched Test Set (100 samples) | 92.00% | 0.9234 | 0.9200 | 0.9200 | 0.1441 |

### Operational Routing on Test Set (Threshold = 0.65, Margin = 0.15)
- Autonomous Acceptance Coverage: 90.00% (90 / 100 tickets)
- Autonomous Prediction Accuracy: 94.44% (85 / 90 correct)
- Human Review Routing Rate: 10.00% (10 / 100 tickets flagged)
- Incorrect Autonomous Acceptance Rate: 5.56% (5 / 90 accepted tickets)

### Per-Class Validation Breakdown (100 samples)
- Delivery: Precision 0.952 | Recall 1.000 | F1 0.976
- General Inquiry: Precision 1.000 | Recall 1.000 | F1 1.000
- Billing: Precision 0.947 | Recall 0.900 | F1 0.923
- Account Access: Precision 0.857 | Recall 0.900 | F1 0.878
- Technical Issue: Precision 0.895 | Recall 0.850 | F1 0.872

---

## Governance and Gate G1 Priority Policy

Priority Model Training is PAUSED.
Historical priority labels in the training set exhibit severe class imbalance (6.5% High priority) and are subject to an active 26-row review queue (`data/processed/dataset_review_required.csv`). Training supervised priority models before policy calibration risks false negatives on critical security incidents. The system provides rule-based heuristic triage indicators while supervised priority training awaits Gate G1 sign-off by Technical Supervisor Abdullah Ramzan.

---

## Getting Started

### 1. Prerequisites
- Python 3.13.x
- Git

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/MuhammadNabeelIjaz/TicketSense-AI.git
cd TicketSense-AI

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests
```powershell
.venv\Scripts\pytest.exe -v
```
All 37 unit, AppTest UI, and integration tests pass in under 3.0 seconds.

### 4. Reproduce Model Training
```powershell
.venv\Scripts\python.exe scripts/train.py
```
This script trains the model on the expanded train split, validates on the validation split, persists `models/category_model.joblib`, and outputs evaluation artifacts to `reports/`.

### 5. Launch the Streamlit Application
```powershell
.venv\Scripts\streamlit.exe run app.py
```
The application launches locally at `http://localhost:8501`.

---

## Automated Testing Verification Summary

The automated test suite in `tests/` covers:
- `test_validation.py`: Null values, whitespace strings, length bounds (5 - 5000 chars), CSV schema validation.
- `test_preprocessing.py`: Negation-preserving contraction expansion, email/URL scrubbing, symbol normalization.
- `test_pipeline.py`: Model deserialization, probability distribution validity, sigmoid calibration preservation.
- `test_uncertainty.py`: Low-confidence and ambiguity thresholds, human-review explanation reasoning.
- `test_corrections.py`: Append-only logging, column preservation, dataset immutability.
- `test_batch.py`: Batch prediction flow, output enrichment, aggregation metrics.
- `test_app_ui.py`: Streamlit AppTest headless execution covering single ticket triage and batch CSV prediction.

---

## License and Maintainers
Developed for the Infinitiv AI / TicketSense-AI Internship Program.
Maintained by the TicketSense-AI Machine Learning Engineering Team.
