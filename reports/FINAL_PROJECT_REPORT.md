# TicketSense-AI: Complete Technical and Empirical Project Report

**Author / Role:** Senior Machine Learning Engineer, NLP Engineer, and Project Auditor  
**Project Repository:** TicketSense-AI  
**Target Audience:** Academic Supervisor, Technical Reviewers, and Internship Evaluators  
**Canonical File:** `reports/FINAL_PROJECT_REPORT.md`  
**Operational Status:** Verification Complete — Ready for Supervisor Demonstration  

---

## 1. Executive Summary

TicketSense-AI is a production-grade natural language processing (NLP) application engineered to classify customer support tickets into five primary operational categories, calibrate multi-class probability scores, calculate deterministic confidence metrics, and implement confidence-aware routing. The platform automatically accepts high-confidence, non-ambiguous classifications while safely delegating low-confidence and boundary cases to human agents for manual review.

Prior to optimization, the baseline machine learning system suffered from severe probability dispersion: despite demonstrating an underlying test accuracy of 88.00%, the model produced an average prediction confidence of only 41.25%. Consequently, under standard operational triage thresholds (confidence threshold of 0.65 and margin threshold of 0.15), 93.00% of incoming tickets were flagged for manual review, yielding an autonomous coverage rate of only 7.00%.

Through root-cause investigation, targeted synthetic dataset expansion (adding 200 validated records covering underrepresented scenarios without modifying the locked baseline partitions), and a systematic 8-experiment evaluation matrix, an optimized classification pipeline was developed and deployed. The final production pipeline integrates custom text normalization, sublinear TF-IDF feature extraction, an optimized Multinomial Naive Bayes classifier, and 5-fold cross-validated Platt Sigmoid probability calibration.

On the isolated, untouched test dataset ($N = 100$), the optimized model achieves:
- Test Accuracy: 92.00% (an increase of 4.00 percentage points over baseline).
- Macro F1-Score: 0.9200 (up from 0.8802).
- Multiclass Brier Score: 0.1441 (reduced from 0.4883, representing a 70.49% error reduction in probability calibration).
- Test Mean Confidence: 84.22% (aligned with true empirical accuracy).
- Autonomous Prediction Coverage: 90.00% (up from 7.00%, processing 90 out of 100 tickets automatically).
- Human Review Rate: 10.00% (reduced from 93.00%, saving 83 percentage points of manual triage load).
- Autonomous Acceptance Accuracy: 94.44% (85 correct classifications out of 90 autonomously accepted tickets).

All project code, dataset partitions, serialized model objects, and Streamlit user interface modules were independently verified. All 45 automated unit tests passed, and the end-to-end quality assurance checklist verified all runtime workflows.

---

## 2. Problem Statement

Customer support centers handle thousands of customer tickets daily across multiple channels including email, web portals, live chat, and mobile applications. In standard manual workflows, human dispatchers must read every ticket, identify the underlying customer intent, assess urgency, and assign the ticket to the appropriate functional department (such as IT Support, Accounts Receivable, Warehouse Logistics, or General Inquiries).

Manual triage introduces significant operational bottlenecks:
1. High Latency: Initial triage delays mean critical tickets (e.g., account lockouts or payment failures) queue behind routine queries.
2. Human Error and Inconsistency: Subjective interpretations lead to inconsistent categorization and repeated re-assignment across queues.
3. Scaling Costs: Linear increases in ticket volume require proportional staffing increases for basic dispatching tasks.

Automated text classification systems mitigate these bottlenecks by routing tickets instantly. However, conventional automated classifiers present substantial operational risks when deployed blindly: when a model encounters out-of-vocabulary terms, ambiguous requests, or multi-topic complaints, standard multi-class models output forced, brittle predictions without alerting operators to underlying uncertainty.

TicketSense-AI implements a selective classification framework that balances automation against reliability:
- High-confidence, distinct predictions are processed autonomously without human intervention.
- Borderline, low-confidence, or ambiguous tickets are routed to human review queues with detailed diagnostic reasons.
- Human overrides are logged to an immutable audit file (`runtime_corrections.csv`) to support human-in-the-loop active learning.

---

## 3. Project Objectives and Scope

### 3.1 Primary Functional Objectives
1. Multi-Class Intent Classification: Automatically classify customer tickets into five predefined enterprise categories: Account Access, Billing, Delivery, General Inquiry, and Technical Issue.
2. Reliable Probability Calibration: Ensure that predicted probability scores accurately reflect true empirical accuracy, avoiding over-confident misclassifications and under-confident correct predictions.
3. Confidence and Margin Triage: Compute dual-factor uncertainty diagnostics combining minimum confidence thresholds with top-1 versus top-2 probability margins.
4. Human-in-the-Loop Corrections: Provide an intuitive interface allowing support agents to inspect predictions, override mistakes, and record corrections.
5. Batch Ingestion and Rejection Safeguards: Ingest batch CSV files while rejecting invalid, empty, or out-of-bounds records with transparent row-level auditing.

### 3.2 Implemented Features vs Original Requirements
- Implemented: End-to-end scikit-learn text pipeline with sublinear TF-IDF vectorization.
- Implemented: Multinomial Naive Bayes classifier with Platt Sigmoid calibration fit via 5-fold cross-validation.
- Implemented: Dual-factor uncertainty triage (`threshold = 0.65`, `margin = 0.15`).
- Implemented: Streamlit web application featuring Single Ticket triage, Batch CSV processing, Model Governance dashboards, and Audit Correction viewing.
- Implemented: Keyword-based heuristic priority indicator ("Low", "Medium", "High") operating as non-blocking provisional guidance.
- Implemented: 45 automated pytest unit and integration tests.

### 3.3 Out-of-Scope and Provisional Features
- Machine Learning Priority Training: Paused under Gate G1 supervisor governance. Due to a severe class imbalance in original priority labels (only 6.5% High priority tickets), machine learning priority training is paused (`PRIORITY_TRAINING_PAUSED = True`) to prevent biased minority predictions until rebalancing is approved.
- Multi-Label Classification: Customer tickets are mapped to a single primary category. Tickets expressing multiple unrelated complaints are routed based on the dominant semantic feature signal.
- Open-Domain Fallback Class: The model operates under a closed-world five-category assumption. Out-of-domain trivia questions map to the nearest semantic cluster rather than an explicit "Out-of-Scope" class.

---

## 4. System Architecture and Workflow

The system is structured as a modular, decoupled architecture where preprocessing, feature extraction, calibration, inference, and UI presentation are strictly separated.

```
+---------------------------------------------------------------------------------------------------+
|                                  TicketSense-AI Pipeline Architecture                             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ Customer Input ] ---> [ Validation Boundary ] ---> [ Preprocessing ] ---> [ Feature Extraction ]|
|  (Raw Support Text)      (Length: 5-5000 chars)       (clean_text /            (Sublinear TF-IDF   |
|                          (Null/Empty check)            TextCleaner)             Unigrams/Bigrams) |
|                                                                                       |           |
|                                                                                       v           |
|  [ Operational Decision ] <--- [ Uncertainty Triage ] <--- [ Probability ] <--- [ Base Classifier]|
|  - Autonomous (Conf >= 0.65     (top_1 vs top_2 Margin     Calibration          (MultinomialNB     |
|    & Margin >= 0.15)             Margin < 0.15 or           (Platt Sigmoid        alpha = 0.1)    |
|  - Human Review Flagged          Conf < 0.65 -> Review)     5-Fold CV)                            |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

### Component Details
1. Validation Boundary (`ticketsense.data.validator`):
   Validates character length between `MIN_TICKET_CHAR_LENGTH = 5` and `MAX_TICKET_CHAR_LENGTH = 5000`. Strips whitespace and blocks empty strings or non-string inputs.
2. Text Normalization (`ticketsense.nlp.preprocessor.TextCleaner`):
   Custom scikit-learn transformer executing contraction expansion (e.g., "can't" to "cannot", "won't" to "will not"), URL and email masking (`<URL>`, `<EMAIL>`), non-alphanumeric character removal, lowercasing, and whitespace compaction.
3. Feature Extraction (`sklearn.feature_extraction.text.TfidfVectorizer`):
   Converts normalized text into sparse n-gram feature vectors using sublinear term frequency scaling ($1 + \log(\text{tf})$) and token regex pattern `(?u)\b\w+\b`.
4. Base Classifier (`sklearn.naive_bayes.MultinomialNB`):
   Computes multinomial log-likelihoods over TF-IDF feature distributions with additive Laplace smoothing (`alpha = 0.1`).
5. Probability Calibration (`sklearn.calibration.CalibratedClassifierCV`):
   Applies Platt Sigmoid scaling via logistic regression fit over out-of-fold logit outputs across 5 stratified folds on the training split.
6. Uncertainty Assessment (`ticketsense.inference.uncertainty`):
   Evaluates top probability against threshold ($0.65$) and probability difference against margin ($0.15$). Identifies the top two competing categories.
7. Priority Heuristic (`ticketsense.inference.priority_stub`):
   Scans text for urgent heuristic tokens (e.g., "urgent", "immediately", "outage", "unauthorized") to assign provisional guidance without statistical bias.
8. Application Layer (`app.py`):
   Streamlit multi-tab dashboard displaying predicted category, calibrated confidence, uncertainty reasons, second-place alternatives, and manual override forms.

---

## 5. Dataset and Data Preparation

### 5.1 Dataset Origin and Structure
The initial project dataset was compiled from support ticket archives containing raw customer queries across five business functions:
- Account Access: Password resets, account lockouts, 2FA/MFA token problems, corporate portal access, and session expirations.
- Billing: Payment failures, subscription charges, invoice inaccuracies, credit card renewals, and refund processing.
- Delivery: Delayed parcels, missing shipments, tracking status gaps, damaged packages, and courier exceptions.
- General Inquiry: Product feature questions, operating hours, system requirements, documentation requests, and enterprise pricing.
- Technical Issue: Software crashes, HTTP server errors (500, 502, 429), blank rendering, broken exports, and API failures.

### 5.2 Partitioning and Hash Verification
To prevent data snooping and methodology leakage, the dataset was partitioned into three distinct splits. Cryptographic SHA-256 hashes confirm that the original files remain completely immutable:

| Dataset File | File Path | Record Count | Verified SHA-256 Hash | Partition Role |
| :--- | :--- | :--- | :--- | :--- |
| Primary Dataset | `data/processed/ticketsense_dataset.csv` | 500 | `ab28ce2070f91246cf7416b99ed732b3eac261643a61c27f9444a85b8e2e9893` | 400 Train / 100 Validation |
| Locked Test Dataset | `data/processed/ticketsense_test.csv` | 100 | `356bcd20c70d96daf07e3aa0a3378cebbf7a1b129b263cf08c687dc6d1ee8615` | Isolated Test Evaluation |
| Synthetic Expansion | `data/processed/ticketsense_synthetic_expansion_v1.csv` | 200 | `9be4a2cd58ae78c9d1df52b57f0230f6534571e7208d98d28a8d115e85bb7aa0` | Training Set Expansion |

### 5.3 Synthetic Expansion and Quality Verification
The initial 400 training samples contained vocabulary gaps that impaired generalization on modern support requests. Exactly 200 synthetic training records (40 per category) were generated using `scripts/generate_synthetic_expansion.py` and saved separately.

Quality Assurance Checks (`scripts/validate_synthetic_expansion.py`):
- Missing Values: Zero null or missing fields across all columns.
- Length Constraints: All records fall within valid bounds [54, 146] characters.
- Duplicate Audit: Zero internal exact duplicates; zero cross-dataset duplicates against training, validation, or test data.
- Semantic Similarity Audit: Zero pairs with cosine similarity > 0.85 against original training data (maximum observed similarity was 0.74).
- Zero Leakage: The validation ($N=100$) and locked test ($N=100$) datasets were strictly excluded from synthetic generation.
- Vocabulary Impact: Added 631 new unique words (+39.6% vocabulary increase, expanding unique training tokens from 1,593 to 2,224).

Combined Training Size: 600 records (Account Access: 118, Billing: 120, Delivery: 120, General Inquiry: 120, Technical Issue: 122).

---

## 6. Exploratory Analysis and Data Challenges

An exploratory analysis of the original dataset revealed four core technical challenges:

1. High Out-of-Vocabulary (OOV) Rate:
   The original 400 training records contained only 1,593 distinct vocabulary terms. Evaluating the 100-record validation split revealed that 27.10% of validation words (190 words) were out-of-vocabulary, causing unobserved terms to contribute zero weight during inference.
2. Underrepresented Support Scenarios:
   Modern enterprise workflows were absent from the initial training corpus. Specifically, queries mentioning "magic links", "2FA authenticator reset", "SAML token expiration", "overage spending caps", "VAT tax exemption", "HTTP 429 rate limits", and "customs clearance holds" were completely unrepresented.
3. Sentence Length Disparity:
   The General Inquiry category had an average length of only 11.4 words (range: 6 to 23 words), whereas Billing and Technical Issue tickets averaged 25.5 and 24.9 words respectively. This disparity caused shorter tickets to receive lower cumulative TF-IDF scores.
4. Softmax Probability Dispersion:
   When linear models (such as Logistic Regression) evaluate tickets containing unseen or weakly weighted terms, the raw logits for all five categories remain close to zero. The softmax denominator divides the exponential mass evenly, producing maximum probability values between 0.25 and 0.42. Because these values fall below the 0.65 threshold, 92% to 93% of tickets were flagged as uncertain despite selecting the correct category.

---

## 7. Text Preprocessing and Feature Engineering

### 7.1 Text Preprocessing Pipeline
Text cleaning is enforced identically across training, batch evaluation, and single-ticket prediction via `ticketsense.nlp.preprocessor`:
- Contraction Mapping: Expands colloquial English contractions (e.g., "don't" -> "do not", "can't" -> "cannot", "it's" -> "it is").
- Regex URL and Email Sanitization: Replaces web links and email addresses with `<URL>` and `<EMAIL>` tokens.
- Character Filtering: Strips non-alphanumeric punctuation while preserving intra-word characters.
- Casing and Whitespace: Converts text to lowercase and replaces multi-line tabs/spaces with single spaces.

### 7.2 TF-IDF Feature Engineering
TF-IDF feature extraction was selected over dense embeddings (e.g., Word2Vec, BERT) due to several technical advantages:
- Deterministic and Interpretable: Feature weights directly correspond to specific customer terminology.
- Low Latency: Sub-millisecond CPU vectorization without GPU dependencies.
- Resilience on Small Corpora: Avoids overfitting risks common to high-capacity deep learning models when trained on small enterprise datasets ($N=600$).

Final Vectorizer Configuration:
- `ngram_range = (1, 2)`: Captures both individual keywords ("refund", "login") and critical phrases ("cannot access", "not arrived", "charged twice").
- `sublinear_tf = True`: Replaces term frequency $\text{tf}$ with $1 + \log(\text{tf})$ to prevent repetitive phrases from disproportionately dominating document vectors.
- `min_df = 1`: Retains domain-specific technical vocabulary present even in single tickets.
- `max_df = 0.85`: Excludes ubiquitous corpus terms appearing in more than 85% of documents.
- `token_pattern = r"(?u)\b\w+\b"`: Ensures single-character tokens and alphanumeric codes (e.g., "500", "4b") are captured.

---

## 8. Model Selection and Experimentation

A systematic 8-experiment matrix was executed to evaluate algorithms, calibration techniques, and dataset sizes. All experiments used Stratified 5-Fold Cross-Validation on the training partition, followed by out-of-sample evaluation on the validation split ($N=100$). The test dataset was locked and untouched during this phase.

All runs were recorded in `reports/EXPERIMENT_RESULTS_LOG.json`.

| Experiment ID | Training Size | Algorithm and Configuration | 5-Fold CV Accuracy | Validation Accuracy | Validation Macro F1 | Multiclass Brier Score | Autonomous Coverage | Autonomous Accuracy | Review Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-00 | 400 | LogisticRegression (C=1.0, Uncalibrated) | 0.8900 | 0.8000 | 0.7997 | 0.4645 | 8.0% | 100.0% | 92.0% |
| EXP-01 | 400 | LogisticRegression (C=5.0, Sigmoid 5-CV) | 0.9025 | 0.8300 | 0.8286 | 0.2522 | 66.0% | 93.9% | 34.0% |
| EXP-02 | 600 | LogisticRegression (C=1.0, Uncalibrated) | 0.9050 | 0.8800 | 0.8801 | 0.4144 | 10.0% | 100.0% | 90.0% |
| EXP-03 | 600 | LogisticRegression (C=5.0, Sigmoid 5-CV) | 0.9217 | 0.9000 | 0.9014 | 0.1732 | 75.0% | 98.7% | 25.0% |
| EXP-04 | 600 | LinearSVC (C=1.0, Sigmoid 5-CV) | 0.9200 | 0.9200 | 0.9202 | 0.1513 | 76.0% | 98.7% | 24.0% |
| EXP-05 | 600 | MultinomialNB (alpha=0.1, Sigmoid 5-CV) | 0.9217 | 0.9300 | 0.9297 | 0.1059 | 87.0% | 98.9% | 13.0% |
| EXP-06 | 600 | SGDClassifier (loss=log_loss, Sigmoid 5-CV) | 0.9200 | 0.9000 | 0.9014 | 0.1699 | 73.0% | 97.3% | 27.0% |
| EXP-07 | 600 | LogisticRegression (C=5.0, Isotonic 5-CV) | 0.9117 | 0.9000 | 0.8994 | 0.1530 | 86.0% | 95.3% | 14.0% |
| EXP-08 | 600 | LogisticRegression (C=8.0, Sigmoid 5-CV) | 0.9200 | 0.9100 | 0.9109 | 0.1798 | 72.0% | 98.6% | 28.0% |

### Key Analysis of Experimental Findings
1. Dataset Expansion Impact: Comparing EXP-00 against EXP-02 demonstrates that adding 200 synthetic records increased uncalibrated accuracy from 80.00% to 88.00% (+8.00 percentage points). However, uncalibrated confidence remained dispersed, leaving review rates at 90.0%.
2. Probability Calibration Impact: Comparing EXP-02 (uncalibrated) with EXP-03 (calibrated) demonstrates that Sigmoid calibration reduced Brier Score from 0.4144 to 0.1732, expanding autonomous coverage from 10.0% to 75.0% while achieving 98.7% autonomous accuracy.
3. Algorithm Comparison:
   - Logistic Regression (EXP-03) achieved 90.00% validation accuracy.
   - Linear SVM (EXP-04) achieved 92.00% validation accuracy with low CV variance (std = 0.0100).
   - Multinomial Naive Bayes (EXP-05) achieved the highest overall validation accuracy (93.00%), highest Macro F1 (0.9297), lowest Brier Score (0.1059), and highest autonomous coverage (87.0%) with 98.85% autonomous accuracy.
   - Isotonic Regression (EXP-07) produced lower autonomous accuracy (95.3%) compared to Sigmoid scaling due to step-function binning on a 600-sample corpus.

---

## 9. Final Model Configuration

The final production model artifact was serialized to `models/category_model.joblib` with metadata saved to `models/category_model_metadata.json`:

```python
Pipeline(steps=[
    ('cleaner', TextCleaner()),
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.85,
        sublinear_tf=True,
        token_pattern=r'(?u)\b\w+\b'
    )),
    ('clf', MultinomialNB(alpha=0.1))
])
```

- Enclosing Estimator: `sklearn.calibration.CalibratedClassifierCV(method="sigmoid", cv=5)`
- Training Corpus: 600 records (original 400 + synthetic 200)
- Classes: `['Account Access', 'Billing', 'Delivery', 'General Inquiry', 'Technical Issue']`
- Uncertainty Configuration: `DEFAULT_UNCERTAINTY_THRESHOLD = 0.65`, `DEFAULT_UNCERTAINTY_MARGIN = 0.15`

---

## 10. Model Evaluation and Results

### 10.1 Validation Split Evaluation ($N = 100$)
Evaluated out-of-sample on the 100 validation records from `ticketsense_dataset.csv`:
- Accuracy: 93.00% (7 errors out of 100)
- Macro Precision: 0.9303
- Macro Recall: 0.9300
- Macro F1-Score: 0.9297
- Multiclass Brier Score: 0.1059
- Log Loss: 0.2551
- Mean Confidence: 82.70%
- Autonomous Coverage: 87.00% (87 tickets processed autonomously)
- Human Review Rate: 13.00% (13 tickets routed to review queue)
- Autonomous Accuracy: 98.85% (86 correct classifications out of 87 autonomous tickets; only 1 error)

Validation Per-Class Performance:
- Account Access: Precision 0.850, Recall 0.900, F1 0.874 (Support: 20)
- Billing: Precision 0.947, Recall 0.900, F1 0.923 (Support: 20)
- Delivery: Precision 0.952, Recall 1.000, F1 0.976 (Support: 20)
- General Inquiry: Precision 1.000, Recall 1.000, F1 1.000 (Support: 20)
- Technical Issue: Precision 0.900, Recall 0.850, F1 0.874 (Support: 20)

### 10.2 Final Isolated Test Set Evaluation ($N = 100$)
Evaluated on the locked test partition (`ticketsense_test.csv`) which was never used during training, tuning, or threshold selection:
- Accuracy: 92.00% (8 errors out of 100)
- Macro Precision: 0.9200
- Macro Recall: 0.9200
- Macro F1-Score: 0.9200
- Multiclass Brier Score: 0.1441
- Log Loss: 0.3187 (detailed in Section 12)
- Mean Confidence: 84.22%
- Autonomous Coverage: 90.00% (90 tickets processed autonomously)
- Human Review Rate: 10.00% (10 tickets routed to review queue)
- Autonomous Accuracy: 94.44% (85 correct classifications out of 90 autonomous tickets)

Test Set Confusion Matrix:

| True Category \ Predicted | Account Access | Billing | Delivery | General Inquiry | Technical Issue | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Account Access | 18 | 0 | 0 | 0 | 2 | 20 |
| Billing | 0 | 19 | 1 | 0 | 0 | 20 |
| Delivery | 1 | 0 | 19 | 0 | 0 | 20 |
| General Inquiry | 0 | 0 | 0 | 19 | 1 | 20 |
| Technical Issue | 1 | 1 | 0 | 1 | 17 | 20 |
| Total | 20 | 20 | 20 | 20 | 20 | 100 |

Test Per-Class Performance:
- Account Access: Precision 0.900, Recall 0.900, F1 0.900 (Support: 20)
- Billing: Precision 0.950, Recall 0.950, F1 0.950 (Support: 20)
- Delivery: Precision 0.950, Recall 0.950, F1 0.950 (Support: 20)
- General Inquiry: Precision 0.950, Recall 0.950, F1 0.950 (Support: 20)
- Technical Issue: Precision 0.850, Recall 0.850, F1 0.850 (Support: 20)

Interactive Empirical Visualizations:
The Streamlit application Performance & Governance tab directly renders six production visualizations powered by `ticketsense/visualization.py`:
1. Dataset Class Distribution: Category frequencies and percentages across the 5 operational classes.
2. Confusion Matrix Heatmap: Test set prediction grid with exact integer counts and class cross-talk.
3. Category-wise Performance: Grouped comparison of Precision, Recall, and F1 across all five categories.
4. Model Experiment Comparison: Validation Accuracy and Macro F1 trajectories across documented experiments.
5. Prediction Confidence Distribution: Test set confidence histogram with KDE density and 0.65 threshold.
6. Automatic vs Manual Review Routing: Donut chart displaying the verified 90% autonomous coverage and 10% human review split.

### 10.3 Metric Definitions in TicketSense-AI
- Accuracy: Proportion of all tickets whose predicted top category matches true ground truth.
- Macro F1: Unweighted arithmetic average of F1-scores across all 5 classes, treating each category with equal importance regardless of frequency.
- Multiclass Brier Score: Mean squared Euclidean distance between predicted probability vectors and one-hot ground truth vectors: $\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$. Lower scores indicate superior probability calibration.
- Autonomous Coverage: Percentage of tickets meeting autonomous routing criteria: $\frac{N_{\text{auto}}}{N_{\text{total}}}$.
- Autonomous Accuracy: Classification accuracy computed exclusively among autonomously accepted tickets: $\frac{N_{\text{auto, correct}}}{N_{\text{auto}}}$. This measures the reliability of the system when operating without human supervision.

---

## 11. Confidence-Based Human Review

### 11.1 Uncertainty Triage Mechanics
The uncertainty scoring function evaluates two criteria:
1. Low Confidence Criterion:
   If $\max_k(p_k) < \text{threshold}$ (default $0.65$), the top prediction lacks statistical certainty.
2. Ambiguity Criterion:
   If $p_{(1)} - p_{(2)} < \text{margin}$ (default $0.15$), the model cannot clearly distinguish between the two most likely categories.

If either condition is met, `is_uncertain` is set to `True`, triggering routing to manual review queues with a human-readable diagnosis string (e.g., `"Low confidence (58.4% < 65.0%)"` or `"Ambiguous prediction between Technical Issue and Account Access (margin 0.08 < 0.15)"`).

### 11.2 Coverage vs Reliability Trade-Off
A threshold sensitivity analysis was conducted on validation data to determine the optimal threshold:

| Threshold | Margin | Autonomous Coverage | Human Review Rate | Autonomous Accuracy | Routing Errors |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.50 | 0.15 | 96.0% | 4.0% | 95.8% | 4 tickets |
| 0.55 | 0.15 | 94.0% | 6.0% | 96.8% | 3 tickets |
| 0.60 | 0.15 | 91.0% | 9.0% | 98.9% | 1 ticket |
| 0.65 | 0.15 | 87.0% | 13.0% | 98.9% | 1 ticket |
| 0.70 | 0.15 | 79.0% | 21.0% | 98.7% | 1 ticket |
| 0.75 | 0.15 | 74.0% | 26.0% | 100.0% | 0 tickets |
| 0.80 | 0.15 | 69.0% | 31.0% | 100.0% | 0 tickets |

Operating at `threshold = 0.65` provides 87% to 90% autonomous throughput while maintaining over 94% to 98% accuracy on automatically routed customer issues.

### 11.3 Confidence Limitations
High prediction confidence does not guarantee correctness. If a customer query contains strong delivery phrasing while describing an overcharge (e.g., "Refund for my order has not arrived"), the model can output high confidence (>85%) for an incorrect category. Confidence reflects model certainty relative to its training distribution, not factual infallibility.

---

## 12. Independent Audit and Verification

An independent audit was conducted using `docs/archive/final_independent_audit/audit_runner.py`:

### 12.1 Hash Verification
- `ticketsense_dataset.csv`: Hash matches `ab28ce2070f91246cf7416b99ed732b3eac261643a61c27f9444a85b8e2e9893` (VERIFIED).
- `ticketsense_test.csv`: Hash matches `356bcd20c70d96daf07e3aa0a3378cebbf7a1b129b263cf08c687dc6d1ee8615` (VERIFIED).

### 12.2 Investigation of Metric Reporting Discrepancies
Two reporting variations between early draft logs and the independent audit were investigated:

1. Test Log Loss Discrepancy (0.3524 vs 0.3187):
   - Investigation: In the experimental script `scripts/evaluate_final_test_set.py`, a dynamic retrain of `m_nb` was performed without a pinned cross-validation split generator. In contrast, the independent audit runner loaded the actual serialized artifact (`models/category_model.joblib`) directly from disk and evaluated it against `ticketsense_test.csv`.
   - Finding: The actual serialized model artifact produces a test Log Loss of `0.3187`, which is lower (better calibrated) than the intermediate script's dynamically refit score of `0.3524`. The value `0.3187` is confirmed as the official metric for the production model.
2. Macro Precision Calculation (0.9213 vs 0.9200):
   - Investigation: The value `0.9213` in the initial summary table was an accidental carryover from EXP-04's validation macro precision table. On the test set, per-class precisions are exactly 0.90, 0.95, 0.95, 0.95, and 0.85. The arithmetic mean is $(0.90 + 0.95 + 0.95 + 0.95 + 0.85) / 5 = 0.9200$.
   - Finding: `0.9200` is the exact, verified Macro Precision for the test dataset.

### 12.3 Automated and Manual QA Verification
- Automated Pytest Suite: 45 passed (0 failures, 0 errors, 100% pass rate).
- Manual QA Checklist (`scripts/run_qa_manual_checklist.py`): Passed all boundary, ingestion, logging, and layout tests with final verdict `READY FOR SUPERVISOR DEMO`.

---

## 13. Error Analysis and Limitations

### 13.1 Audit of the Five Incorrect Autonomous Test Predictions
On the test set, five tickets met autonomous criteria (confidence >= 0.65 and margin >= 0.15) but were misclassified:

1. Record `TS-0512`:
   - Text: `"Refund for order 8802 still hasnt arrived after 3 weeks."`
   - True: Billing | Predicted: Delivery | Confidence: 88.83% | Margin: 0.8514
   - Root Cause: Lexical overlap. "order", "hasnt arrived", and "3 weeks" are dominant features for delayed shipments in the training corpus, overriding the financial token "Refund".
2. Record `TS-0543`:
   - Text: `"Never got the verification code by text."`
   - True: Account Access | Predicted: Technical Issue | Confidence: 75.94% | Margin: 0.6455
   - Root Cause: Missing telecommunication intent. The text describes an SMS delivery failure, triggering technical message-gateway defect features rather than identity lockout features.
3. Record `TS-0548`:
   - Text: `"Login page just refreshes when I press the button."`
   - True: Account Access | Predicted: Technical Issue | Confidence: 88.88% | Margin: 0.8359
   - Root Cause: Cross-cutting intent. The customer's objective is logging in, but the operational symptom is an unhandled webpage JavaScript reload.
4. Record `TS-0570`:
   - Text: `"The courier tried to deliver while I was at work and I can't find a new time to pick it up."`
   - True: Delivery | Predicted: Account Access | Confidence: 77.75% | Margin: 0.6769
   - Root Cause: Ambiguity. The phrase "can't find a new time to pick it up" lacked specific parcel keywords and was attributed to calendar/portal appointment scheduling.
5. Record `TS-0590`:
   - Text: `"Great job on the latest release, search is so much faster now."`
   - True: General Inquiry | Predicted: Technical Issue | Confidence: 76.46% | Margin: 0.5957
   - Root Cause: Positive user feedback containing software release vocabulary ("latest release", "search", "faster").

### 13.2 System Limitations
1. Closed-World Assumption: The classifier forces out-of-domain inputs into one of the 5 categories. Out-of-domain trivia questions map to General Inquiry.
2. Single-Label Constraint: Multi-intent tickets are assigned to whichever complaint has higher cumulative keyword weight.
3. Priority Policy: Statistical priority modeling remains paused under Gate G1; operational guidance relies on non-blocking heuristic rules.

---

## 14. Application Features and Testing

The Streamlit web application (`app.py`) provides four functional operational tabs:

1. Single Ticket Classification:
   - Users paste or type support messages.
   - Displays Predicted Category, Calibrated Confidence (%), Triage Status (Autonomous vs Review Required), and Heuristic Priority.
   - Provides an expandable bar chart displaying the calibrated probability distribution across all five categories.
   - Includes a human-in-the-loop override form allowing agents to reclassify tickets and save feedback directly to `runtime_corrections.csv`.
2. Batch CSV Processing:
   - Ingests multi-row CSV files containing ticket text.
   - Validates every row: empty rows or rows shorter than 5 characters are rejected with row numbers and rejection reasons displayed in a table.
   - Valid tickets are processed in batch mode, generating an enriched DataFrame with predicted categories, confidence scores, and uncertainty reasons.
   - Provides one-click CSV export of enriched predictions.
3. Performance and Governance Dashboard:
   - Displays real-time model metadata: training sample count ($N=600$), validation accuracy ($93.0\%$), and macro F1 ($0.9297$).
   - Renders the validation confusion matrix plot.
   - Clearly displays the Gate G1 policy banner explaining that priority ML training is paused.
4. Corrections Audit Log:
   - Displays a table of all runtime human corrections logged to `data/corrections/runtime_corrections.csv`.

Application testing verified clean startup on port 8501 and complete end-to-end functionality across all tabs.

---

## 15. Technology Stack

The project relies on a verified, minimal Python scientific software stack:
- Python (v3.13.15): Base runtime environment.
- pandas (v2.2.3): Dataset manipulation, split management, and CSV ingestion.
- numpy (v2.2.3): Vector calculations, probability normalization, and Brier score evaluation.
- scikit-learn (v1.6.1): Preprocessing transformers, `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `LinearSVC`, `SGDClassifier`, `CalibratedClassifierCV`, and evaluation metrics.
- joblib (v1.4.2): Model and metadata serialization and deserialization.
- Streamlit (v1.42.1): Interactive web application interface.
- pytest (v9.2.0): Automated test execution framework.

---

## 16. Project Deliverables

All deliverables reside within the project root repository:
- Source Code:
  - `ticketsense/config.py`: Central configuration and taxonomy paths.
  - `ticketsense/nlp/preprocessor.py`: Text cleaning and normalizer transformer.
  - `ticketsense/data/loader.py`: Dataset split loading utilities.
  - `ticketsense/data/validator.py`: Input boundary and batch validation logic.
  - `ticketsense/data/corrections.py`: Human override audit logging module.
  - `ticketsense/models/category_pipeline.py`: Pipeline builder and training implementation.
  - `ticketsense/models/evaluator.py`: Classification metrics and confusion matrix plotter.
  - `ticketsense/models/persistence.py`: Model bundle persistence utilities.
  - `ticketsense/inference/engine.py`: High-level inference engine.
  - `ticketsense/inference/uncertainty.py`: Uncertainty and triage diagnostic calculator.
  - `ticketsense/inference/priority_stub.py`: Rule-based priority heuristic module.
  - `ticketsense/visualization.py`: Production data visualization module (6 project-relevant figures).
  - `app.py`: Streamlit multi-tab production dashboard.
- Scripts:
  - `scripts/train.py`: CLI training runner.
  - `scripts/generate_synthetic_expansion.py`: Synthetic dataset generator.
  - `scripts/validate_synthetic_expansion.py`: Synthetic data validation runner.
  - `scripts/run_optimization_experiments.py`: 8-experiment optimization matrix runner.
  - `scripts/evaluate_threshold_tradeoffs.py`: Threshold trade-off evaluation script.
  - `scripts/evaluate_final_test_set.py`: Test dataset evaluation script.
  - `scripts/run_qa_manual_checklist.py`: Comprehensive manual QA verification script.
- Datasets:
  - `data/processed/ticketsense_dataset.csv`: 500 records (400 train, 100 validation).
  - `data/processed/ticketsense_test.csv`: 100 records (locked test set).
  - `data/processed/ticketsense_synthetic_expansion_v1.csv`: 200 synthetic training records.
  - `data/corrections/runtime_corrections.csv`: Audit log of agent overrides.
- Model Artifacts:
  - `models/category_model.joblib`: Serialized production pipeline artifact.
  - `models/category_model_metadata.json`: Model architecture and training metadata.
- Reports and Canonical Artifacts:
  - `reports/FINAL_PROJECT_REPORT.md`: This canonical report.
  - `reports/EXPERIMENT_RESULTS_LOG.json`: Machine-readable experiment log.
  - `reports/CATEGORY_MODEL_EVALUATION.md`: Training evaluation markdown summary.
  - `reports/category_confusion_matrix.png`: Confusion matrix plot.
  - `reports/category_validation_errors.csv`: Detailed validation misclassifications log.
  - `reports/visualizations/`: Dedicated folder containing publication-grade 300 DPI chart images (`dataset_class_distribution.png`, `test_confusion_matrix.png`, `category_performance.png`, `model_experiment_comparison.png`, `confidence_distribution.png`, `automatic_vs_manual_review.png`).
- Historical Archives:
  - `docs/archive/final_independent_audit/`: Dedicated audit folder containing `FINAL_AUDIT_REPORT.md`, `AUDIT_METRICS.json`, `DATASET_INTEGRITY_REPORT.md`, `INFERENCE_AUDIT_REPORT.md`, `ISSUES_AND_RECOMMENDATIONS.md`, and `audit_runner.py`.
  - `docs/archive/MODEL_OPTIMIZATION_EXPERIMENTS.md`: Historical optimization experiment report.
  - `docs/archive/HANDOVER.md`: Engineering handover documentation.
  - `docs/archive/HANDOVER_2026-11-23_LEGACY.md`: Baseline legacy handover notes.

---

## 17. Current Project Status

| Component | Status | Details |
| :--- | :--- | :--- |
| Category Classification Model | Completed | MultinomialNB + TF-IDF with Sigmoid calibration fit on 600 records; verified 92% test accuracy and 90% autonomous throughput. |
| Confidence & Triage Engine | Completed | Dual-factor uncertainty logic (threshold 0.65, margin 0.15) verified; reduces human review to 10%. |
| Data Expansion & Quality Audit | Completed | 200 synthetic records validated with zero duplicates and zero leakage. Original datasets locked and immutable. |
| Streamlit Application | Completed | Multi-tab UI verified functional; single ticket, batch processing, governance, and audit logging working cleanly. |
| Automated Test Suite | Completed | 45 of 45 pytest unit and integration tests passing. |
| Priority Classification Model | Provisional | Machine learning training paused pending Gate G1 supervisor approval. Provisional rule-based heuristic indicator active. |
| Open-Domain Intent Gatekeeper | Future Improvement | Proposed enhancement to reject non-support trivia queries before category classification. |
| Multi-Label Intent Parsing | Future Improvement | Proposed architectural expansion for compound multi-problem tickets. |

---

## 18. Conclusion

The TicketSense-AI project has successfully resolved the low-confidence triage bottleneck that previously caused 93% of support tickets to be halted for manual review. By augmenting the training corpus with 200 high-quality, targeted synthetic records and implementing Platt Sigmoid probability calibration on a sublinear TF-IDF Multinomial Naive Bayes pipeline, the system achieves an optimal balance between high operational automation and reliable accuracy.

On the isolated test set, autonomous throughput increased from 7% to 90%, while maintaining a 94.44% autonomous classification accuracy. The uncertainty triage engine demonstrated effective risk mitigation by successfully flagging doubtful and ambiguous predictions for human review.

The complete software stack, data partitions, model artifacts, and test suites are verified and reproducible. The project is fully prepared for supervisor review and demonstration.

---

## 19. References and Evidence

1. Project Internal Source Code:
   - Configuration: `ticketsense/config.py`
   - Inference Engine: `ticketsense/inference/engine.py`
   - Training Pipeline: `ticketsense/models/category_pipeline.py`
2. Project Empirical Evidence:
   - Audit Metrics Payload: `docs/archive/final_independent_audit/AUDIT_METRICS.json`
   - Comprehensive Experiment Log: `reports/EXPERIMENT_RESULTS_LOG.json`
   - Independent Audit Report: `docs/archive/final_independent_audit/FINAL_AUDIT_REPORT.md`
   - Dataset Integrity Report: `docs/archive/final_independent_audit/DATASET_INTEGRITY_REPORT.md`
   - Inference Audit Report: `docs/archive/final_independent_audit/INFERENCE_AUDIT_REPORT.md`
   - Optimization Experiments Guide: `docs/archive/MODEL_OPTIMIZATION_EXPERIMENTS.md`
   - Engineering Handover Guide: `docs/archive/HANDOVER.md`
3. Primary Dataset Partitions:
   - Training and Validation Split: `data/processed/ticketsense_dataset.csv`
   - Isolated Test Split: `data/processed/ticketsense_test.csv`
   - Synthetic Expansion Partition: `data/processed/ticketsense_synthetic_expansion_v1.csv`
