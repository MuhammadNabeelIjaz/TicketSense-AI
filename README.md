# TicketSense-AI: Smart Customer Support Ticket Classifier

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 37 Passed](https://img.shields.io/badge/tests-37%20passed-brightgreen.svg)]()
[![Model Accuracy: 80%](https://img.shields.io/badge/val%20accuracy-80.0%25-blueviolet.svg)]()
[![Priority Model: Paused](https://img.shields.io/badge/priority%20training-PAUSED%20(Gate%20G1)-orange.svg)]()

TicketSense-AI is a modular, production-engineered machine learning system designed to automatically classify customer support tickets into operational categories, quantify prediction uncertainty, triage high-risk messages, and support human-in-the-loop review without mutating underlying historical datasets.

---

## 📌 Project Overview

Incoming customer support tickets are frequently ambiguous, misspelled, or operationally sensitive. TicketSense-AI provides:
- **Real-Time Single Ticket Triage:** Text normalization, category prediction, confidence scoring, and issue summarization.
- **Batch CSV Processing:** High-throughput prediction on uploaded datasets with interactive filtering and downloadable enriched CSVs.
- **Uncertainty Quantification:** Flags low-confidence predictions ($< 65\%$) and ambiguous class margins ($< 15\%$) for mandatory human verification.
- **Auditable Human Correction Logging:** Support agent overrides are appended to an immutable audit log (`data/corrections/runtime_corrections.csv`) to enable active learning without altering historical training records.
- **Governance Guardrails:** Priority machine learning training is explicitly paused pending Gate G1 supervisor sign-off due to class imbalance ($6.5\%$ High priority) and active review queues.

---

## 🏗️ System Architecture

```
TicketSense-AI/
├── app.py                         # Production Streamlit web application
├── pytest.ini                     # Pytest configuration with pythonpath
├── requirements.txt               # Pinned dependencies
├── ticketsense/                   # Core modular package
│   ├── config.py                  # Project paths, taxonomy, and thresholds
│   ├── data/                      # Ingestion, validation, and corrections
│   │   ├── loader.py              # Split loader (train, validation, test)
│   │   ├── validator.py           # Text & batch CSV schema validation
│   │   └── corrections.py         # Append-only human correction logger
│   ├── nlp/                       # Natural language processing
│   │   └── preprocessor.py        # Text cleaning, contraction expansion, normalization
│   ├── models/                    # ML pipeline & evaluation
│   │   ├── category_pipeline.py   # TF-IDF + LogisticRegression pipeline
│   │   ├── evaluator.py           # Metrics, confusion matrix, and error analysis
│   │   └── persistence.py         # Joblib serialization with JSON metadata
│   └── inference/                 # Inference runtime
│       ├── engine.py              # Unified single & batch inference engine
│       ├── uncertainty.py         # Decision uncertainty quantification
│       └── priority_stub.py       # Priority governance stub (PAUSED per Gate G1)
├── scripts/
│   └── train.py                   # CLI runner for reproducible model training
├── tests/                         # Comprehensive automated test suite (25 tests)
│   ├── conftest.py                # Pytest fixtures
│   ├── test_batch.py              # Batch processing tests
│   ├── test_corrections.py        # Corrections logging & immutability tests
│   ├── test_pipeline.py           # Model loading, output shapes, calibration
│   ├── test_preprocessing.py     # Contractions, URLs, emails, normalization
│   ├── test_uncertainty.py        # Low-confidence & ambiguity flags
│   └── test_validation.py         # String boundaries & schema validation
├── models/                        # Serialized model artifacts
│   ├── category_model.joblib      # Trained pipeline artifact
│   └── category_model_metadata.json # Hyperparameters, timestamps, metrics
├── reports/                       # Validation evaluation artifacts
│   ├── CATEGORY_MODEL_EVALUATION.md
│   ├── category_confusion_matrix.png
│   └── category_validation_errors.csv
└── data/
    ├── processed/                 # Audited datasets (400 train, 100 val, 100 test)
    └── corrections/               # Runtime agent overrides log
```

---

## 📊 Dataset & Category Taxonomy

The processed dataset contains 600 verified customer support records partitioned into three distinct splits:
- **Train (400 rows):** Supervised model fitting.
- **Validation (100 rows):** Hyperparameter tuning and threshold calibration.
- **Test (100 rows):** Isolated synthetic benchmark for final unblinded testing.

### Approved Category Classes
1. **Account Access:** Login failures, password resets, 2FA errors, MFA lockouts.
2. **Billing:** Invoices, charges, unauthorized debits, refund disputes, subscription fees.
3. **Delivery:** Shipment tracking, damaged goods, carrier delays, missing parcels.
4. **General Inquiry:** Business hours, product capabilities, policy questions.
5. **Technical Issue:** Software bugs, HTTP 500 errors, UI rendering glitches, crashing.

---

## 📈 Model Performance & Validation Results

The baseline category classifier uses sublinear TF-IDF word n-grams $(1, 2)$ coupled with balanced-class regularized Logistic Regression.

| Metric | Validation Score |
| :--- | :--- |
| **Accuracy** | **80.00%** |
| **Macro Precision** | **81.04%** |
| **Macro Recall** | **80.00%** |
| **Macro F1-Score** | **79.97%** |
| **Weighted F1-Score** | **79.97%** |

### Per-Class Validation Breakdown (100 samples)
- **Delivery:** Precision `0.900` | Recall `0.900` | F1 `0.900`
- **Billing:** Precision `0.850` | Recall `0.850` | F1 `0.850`
- **General Inquiry:** Precision `0.810` | Recall `0.850` | F1 `0.829`
- **Account Access:** Precision `0.789` | Recall `0.750` | F1 `0.769`
- **Technical Issue:** Precision `0.650` | Recall `0.650` | F1 `0.650`

---

## 🛡️ Governance & Gate G1 Priority Policy

> [!IMPORTANT]
> **Priority Model Training is PAUSED.**
> Historical priority labels in the training set exhibit severe class imbalance ($6.5\%$ High priority) and are subject to an active 26-row review queue (`data/processed/dataset_review_required.csv`). Training supervised priority models before policy calibration risks false negatives on critical security incidents. The system provides rule-based heuristic triage indicators while supervised priority training awaits Gate G1 sign-off by Technical Supervisor Abdullah Ramzan.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python `3.13.x`
- Git

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/infinitiv-ai/TicketSense-AI.git
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
*(All 25 unit and integration tests pass in $< 1.0\text{s}$.)*

### 4. Reproduce Model Training
```powershell
.venv\Scripts\python.exe scripts/train.py
```
This script trains the model on the `train` split, validates on the `validation` split, persists `models/category_model.joblib`, and outputs reports in `reports/`.

### 5. Launch the Streamlit Application
```powershell
.venv\Scripts\streamlit.exe run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 Testing Verification Summary

The automated test suite in `tests/` covers:
- **`test_validation.py`:** Null values, whitespace strings, length bounds ($5 - 5000$), CSV column schemas.
- **`test_preprocessing.py`:** Negation-preserving contraction expansion, email/URL scrubbing, symbol normalization.
- **`test_pipeline.py`:** Model deserialization, probability distribution validity ($\sum P = 1.0$).
- **`test_uncertainty.py`:** Low-confidence and ambiguity thresholds, reasoning explanations.
- **`test_corrections.py`:** Append-only logging, column preservation, dataset immutability.
- **`test_batch.py`:** Batch prediction flow, output enrichment, and aggregation metrics.

---

## 📄 License & Maintainers
Developed for the Infinitiv AI / TicketSense-AI Internship Program.  
Maintained by the TicketSense-AI Machine Learning Engineering Team.
