# TicketSense-AI: Final Project Report

Repository: TicketSense-AI
Status: Complete and ready for supervisor demonstration

---

## 1. Summary

TicketSense-AI classifies customer support tickets into five categories: Account Access, Billing, Delivery, General Inquiry and Technical Issue. It also reports how confident it is. Confident predictions are accepted automatically. Low-confidence and ambiguous ones are sent to a human for review.

The first version of the model had a confidence problem. Its test accuracy was 88.00%, but its mean confidence was only 41.25%. With the review rules in place (confidence of at least 0.65 and a margin of at least 0.15), 93% of tickets were flagged for review and only 7% were handled automatically. The model was often right but never sure.

I fixed this in three steps:

1. Added 200 synthetic training tickets to cover vocabulary the original data was missing.
2. Added sigmoid probability calibration (5-fold cross-validation).
3. Compared nine model and calibration setups and picked the best one, which was Multinomial Naive Bayes.

Results on the untouched test set (100 tickets):

| Metric | Baseline | Final model |
| :--- | :--- | :--- |
| Accuracy | 88.00% | 92.00% |
| Macro F1 | 0.8802 | 0.9200 |
| Brier score (lower is better) | 0.4883 | 0.1441 |
| Tickets handled automatically | 7% | 90% |
| Tickets sent to review | 93% | 10% |
| Accuracy on automatic tickets | n/a | 94.44% (85 of 90) |

Mean confidence on the test set is 84.22%, which is below the 92% accuracy, so the model is slightly under-confident.

The code, data splits, model and app were checked end to end, and all 45 automated tests pass.

---

## 2. Problem

Support teams receive large numbers of tickets by email, chat and web forms. Someone has to read each one, decide what it is about and send it to the right team. Done by hand, this is slow, inconsistent between people, and gets more expensive as volume grows.

A classifier can do the routing instantly, but a standard classifier always returns an answer, even when the message is vague, unusual or covers several topics. TicketSense-AI uses selective classification: the model acts on predictions it is confident about and passes the rest to a person with a short explanation of why. Human corrections are saved to `runtime_corrections.csv` so they can be reviewed or used later.

---

## 3. Scope

**Implemented**

- A scikit-learn pipeline: text cleaning, TF-IDF, Multinomial Naive Bayes, sigmoid calibration.
- Human review flags based on a confidence threshold (0.65) and a margin between the top two classes (0.15).
- A Streamlit app with single-ticket triage, batch CSV processing, a model dashboard and a corrections log.
- A keyword-based priority indicator (Low, Medium, High) shown as a hint only.
- 45 automated tests.

**Not implemented**

- **Trained priority model.** Training is paused (`PRIORITY_TRAINING_PAUSED = True`) pending supervisor approval at Gate G1. Only 6.5% of the priority labels are High, so a model trained on them would likely miss urgent tickets until the labels are rebalanced.
- **Multi-label classification.** Each ticket gets one category. If a ticket has several issues, the model picks the dominant one.
- **An "out of scope" class.** The model works with five categories only. Off-topic messages are assigned to the closest one.

---

## 4. How it works

```
Ticket text
  -> Validation (5-5000 characters, not empty)
  -> Text cleaning
  -> TF-IDF (unigrams + bigrams)
  -> Multinomial Naive Bayes (alpha = 0.1)
  -> Sigmoid calibration (5-fold CV)
  -> Uncertainty check (confidence < 0.65 or margin < 0.15 -> human review)
  -> Result: category, probabilities, review flag
```

| Component | Location | What it does |
| :--- | :--- | :--- |
| Validation | `ticketsense/data/validator.py` | Rejects empty, non-text, too short (under 5) or too long (over 5000 characters) input. |
| Text cleaning | `ticketsense/nlp/preprocessor.py` | Expands contractions ("can't" becomes "cannot"), replaces URLs and emails with placeholders, removes punctuation, lowercases, and collapses whitespace. |
| TF-IDF | scikit-learn | Converts text into word and word-pair features. |
| Classifier | scikit-learn | Multinomial Naive Bayes with smoothing `alpha = 0.1`. |
| Calibration | scikit-learn | `CalibratedClassifierCV` with sigmoid scaling, so the probabilities better match how often the model is right. |
| Uncertainty check | `ticketsense/inference/uncertainty.py` | Flags a ticket for review and explains why. |
| Priority heuristic | `ticketsense/inference/priority_stub.py` | Looks for urgent words such as "urgent", "immediately", "outage" and "unauthorized". |
| App | `app.py` | Streamlit interface. |

The same cleaning step is used in training, batch prediction and single-ticket prediction.

---

## 5. Data

### 5.1 Categories, Provenance & Licensing

The dataset consists of 400 records sourced from the `devtry3d/support-ticket-classification` public corpus (Kaggle MIT License) across four categories, combined with 200 reviewed synthetic records to cover General Inquiry and test partitions. All texts underwent automated regex scans (for emails, phone numbers, and payment cards) to eliminate personally identifiable information (PII). It covers five operational areas:

- **Account Access:** password resets, lockouts, 2FA/MFA problems, session expiry.
- **Billing:** failed payments, subscription charges, invoice errors, refunds.
- **Delivery:** delayed or missing parcels, tracking problems, damaged packages.
- **General Inquiry:** product questions, opening hours, system requirements, pricing.
- **Technical Issue:** crashes, HTTP errors (500, 502, 429), blank pages, broken exports, API failures.

### 5.2 Splits

The data is split three ways. SHA-256 hashes were recorded to confirm that the original files were never changed.

| File | Rows | Use | SHA-256 |
| :--- | :--- | :--- | :--- |
| `data/processed/ticketsense_dataset.csv` | 500 | 400 train, 100 validation | `ab28ce2070f91246cf7416b99ed732b3eac261643a61c27f9444a85b8e2e9893` |
| `data/processed/ticketsense_test.csv` | 100 | Test only | `356bcd20c70d96daf07e3aa0a3378cebbf7a1b129b263cf08c687dc6d1ee8615` |
| `data/processed/ticketsense_synthetic_expansion_v1.csv` | 200 | Extra training data | `9be4a2cd58ae78c989a46832d3cf84f0809e9d45ab67feae9d560f89abc1e5cb` |

The final training set has 600 tickets: Account Access 118, Billing 120, Delivery 120, General Inquiry 120 and Technical Issue 122.

### 5.3 Problems found in the original data

1. **Many unseen words.** The 400 training tickets contained only 1,593 distinct words. In the validation split, 27.10% of words (190) never appeared in training, so the model had no information about them.
2. **Missing topics.** Several common scenarios were absent: magic links, authenticator (2FA) resets, SAML token expiry, overage spending caps, VAT exemption, HTTP 429 rate limits and customs holds.
3. **Uneven ticket length.** General Inquiry tickets averaged 11.4 words, compared with 25.5 for Billing and 24.9 for Technical Issue.
4. **Spread-out probabilities.** When a ticket contained unfamiliar words, the model's probabilities were nearly equal across the five classes. The top probability was typically 0.25 to 0.42, below the 0.65 threshold. As a result, 92 to 93% of tickets were flagged even when the predicted category was correct.

### 5.4 Synthetic expansion

To address problems 1 and 2, I generated 200 synthetic tickets (40 per category) with `scripts/generate_synthetic_expansion.py` and kept them in a separate file. `scripts/validate_synthetic_expansion.py` checks them:

- No missing values, and all lengths are between 54 and 146 characters.
- No duplicates within the synthetic set or against the train, validation or test data.
- No synthetic ticket has a cosine similarity above 0.85 with any original training ticket (the highest was 0.74).
- The validation and test sets were not used to generate any synthetic ticket.
- Vocabulary grew from 1,593 to 2,224 unique words (+39.6%).

---

## 6. Features

TF-IDF was chosen over embeddings such as Word2Vec or BERT for three reasons: the features are easy to inspect, it runs fast on a CPU, and it is less likely to overfit on a small dataset of 600 tickets.

Vectorizer settings:

| Setting | Value | Reason |
| :--- | :--- | :--- |
| `ngram_range` | (1, 2) | Captures single words ("refund") and short phrases ("charged twice", "not arrived"). |
| `sublinear_tf` | True | Stops repeated words from dominating a ticket. |
| `min_df` | 1 | Keeps rare technical terms. |
| `max_df` | 0.85 | Drops words that appear in more than 85% of tickets. |
| `token_pattern` | `(?u)\b\w+\b` | Keeps single-character tokens and codes such as "500". |

---

## 7. Experiments

I ran nine configurations (EXP-00 to EXP-08), comparing algorithms, calibration methods and training size. Each was cross-validated (stratified 5-fold) on the training data and then scored on the validation split. The test set was not used. All runs are logged in `reports/EXPERIMENT_RESULTS_LOG.json`.

"Coverage" is the share of validation tickets accepted automatically. "Auto accuracy" is the accuracy on those tickets only.

| ID | Train rows | Model | CV acc. | Val. acc. | Val. macro F1 | Brier | Coverage | Auto accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-00 | 400 | Logistic Regression (C=1), uncalibrated | 0.8900 | 0.8000 | 0.7997 | 0.4645 | 8% | 100.0% |
| EXP-01 | 400 | Logistic Regression (C=5), sigmoid | 0.9025 | 0.8300 | 0.8286 | 0.2522 | 66% | 93.9% |
| EXP-02 | 600 | Logistic Regression (C=1), uncalibrated | 0.9050 | 0.8800 | 0.8801 | 0.4144 | 10% | 100.0% |
| EXP-03 | 600 | Logistic Regression (C=5), sigmoid | 0.9217 | 0.9000 | 0.9014 | 0.1732 | 75% | 98.7% |
| EXP-04 | 600 | Linear SVM (C=1), sigmoid | 0.9200 | 0.9200 | 0.9202 | 0.1513 | 76% | 98.7% |
| **EXP-05** | 600 | **Multinomial Naive Bayes (alpha=0.1), sigmoid** | 0.9217 | **0.9300** | **0.9297** | **0.1059** | **87%** | **98.9%** |
| EXP-06 | 600 | SGD (log loss), sigmoid | 0.9200 | 0.9000 | 0.9014 | 0.1699 | 73% | 97.3% |
| EXP-07 | 600 | Logistic Regression (C=5), isotonic | 0.9117 | 0.9000 | 0.8994 | 0.1530 | 86% | 95.3% |
| EXP-08 | 600 | Logistic Regression (C=8), sigmoid | 0.9200 | 0.9100 | 0.9109 | 0.1798 | 72% | 98.6% |

What the results show:

- **More data helped accuracy.** EXP-00 vs EXP-02: adding the synthetic tickets raised validation accuracy from 80% to 88%. Confidence was still low, so about 90% of tickets were still sent to review.
- **Calibration fixed the confidence problem.** EXP-02 vs EXP-03: sigmoid calibration cut the Brier score from 0.4144 to 0.1732 and raised automatic coverage from 10% to 75%.
- **Naive Bayes was best on every headline metric.** It had the highest accuracy (93%) and macro F1 (0.9297), the lowest Brier score (0.1059) and the highest coverage among the sigmoid-calibrated models (87%).
- **Isotonic calibration was worse than sigmoid here.** Its automatic accuracy was lower (95.3% vs 98.7%). This is probably because isotonic calibration needs more data than 600 tickets.

---

## 8. Final model

The model is saved to `models/category_model.joblib`, with metadata in `models/category_model_metadata.json`.

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
# wrapped in CalibratedClassifierCV(method="sigmoid", cv=5)
```

- Training data: 600 tickets (400 original, 200 synthetic)
- Classes: Account Access, Billing, Delivery, General Inquiry, Technical Issue
- Review rules: `DEFAULT_UNCERTAINTY_THRESHOLD = 0.65`, `DEFAULT_UNCERTAINTY_MARGIN = 0.15`

---

## 9. Results

### 9.1 Validation set (100 tickets)

| Metric | Value |
| :--- | :--- |
| Accuracy | 93.00% (7 errors) |
| Macro precision / recall / F1 | 0.9303 / 0.9300 / 0.9297 |
| Brier score / log loss | 0.1059 / 0.2551 |
| Mean confidence | 82.70% |
| Handled automatically | 87 tickets, 86 correct (98.85%) |
| Sent to review | 13 tickets |

| Category | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| Account Access | 0.857 | 0.900 | 0.878 |
| Billing | 0.947 | 0.900 | 0.923 |
| Delivery | 0.952 | 1.000 | 0.976 |
| General Inquiry | 1.000 | 1.000 | 1.000 |
| Technical Issue | 0.895 | 0.850 | 0.872 |

### 9.2 Test set (100 tickets)

The test set was not used for training, tuning or choosing thresholds.

| Metric | Value |
| :--- | :--- |
| Accuracy | 92.00% (8 errors) |
| Macro precision / recall / F1 | 0.9200 / 0.9200 / 0.9200 |
| Brier score / log loss | 0.1441 / 0.3187 |
| Mean confidence | 84.22% |
| Handled automatically | 90 tickets, 85 correct (94.44%) |
| Sent to review | 10 tickets |

Confusion matrix (rows are the true category, columns are the prediction):

| True \ Predicted | Account Access | Billing | Delivery | General Inquiry | Technical Issue |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Account Access | **18** | 0 | 0 | 0 | 2 |
| Billing | 0 | **19** | 1 | 0 | 0 |
| Delivery | 1 | 0 | **19** | 0 | 0 |
| General Inquiry | 0 | 0 | 0 | **19** | 1 |
| Technical Issue | 1 | 1 | 0 | 1 | **17** |

| Category | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| Account Access | 0.900 | 0.900 | 0.900 |
| Billing | 0.950 | 0.950 | 0.950 |
| Delivery | 0.950 | 0.950 | 0.950 |
| General Inquiry | 0.950 | 0.950 | 0.950 |
| Technical Issue | 0.850 | 0.850 | 0.850 |

Technical Issue is the weakest class. The most common mix-up is between Account Access and Technical Issue (3 of the 8 errors), which share vocabulary such as login errors and verification codes.

### 9.3 Metric definitions

- **Accuracy:** share of tickets where the top predicted category is correct.
- **Macro F1:** average of the five per-class F1 scores, so every class counts equally.
- **Brier score:** mean squared difference between the predicted probabilities and the true label (as a 0/1 vector). Lower means better-calibrated probabilities.
- **Coverage:** share of tickets accepted automatically.
- **Automatic accuracy:** accuracy on the automatically accepted tickets only. This shows how reliable the system is when no one checks its output.

The app's Performance tab shows six charts generated by `ticketsense/visualization.py`: class distribution, test confusion matrix, per-category performance, experiment comparison, confidence distribution and the automatic vs review split.

---

## 10. Human review rules

A ticket is sent to review if either condition is true:

1. **Low confidence:** the top probability is below 0.65.
2. **Ambiguous:** the gap between the top two probabilities is below 0.15.

The app shows the reason, for example `Low confidence (58.4% < 65.0%)` or `Ambiguous prediction between Technical Issue and Account Access (margin 0.08 < 0.15)`.

### 10.1 Choosing the threshold

I compared thresholds on the validation set, with the margin fixed at 0.15:

| Threshold | Coverage | Sent to review | Auto accuracy | Wrong automatic tickets |
| :--- | :--- | :--- | :--- | :--- |
| 0.50 | 96% | 4% | 95.8% | 4 |
| 0.55 | 94% | 6% | 96.8% | 3 |
| 0.60 | 91% | 9% | 98.9% | 1 |
| **0.65** | **87%** | **13%** | **98.9%** | **1** |
| 0.70 | 79% | 21% | 98.7% | 1 |
| 0.75 | 74% | 26% | 100.0% | 0 |
| 0.80 | 69% | 31% | 100.0% | 0 |

A higher threshold sends more tickets to people but lets fewer mistakes through. I chose 0.65 as a compromise. On the validation set it handles 87% of tickets automatically with one error, and on the test set it handles 90% at 94.44% accuracy. Teams that want fewer wrong automatic decisions can raise the threshold.

### 10.2 Confidence is not a guarantee

The model can be confidently wrong. A ticket that reads like a delivery problem but is really a refund request can get over 85% confidence for the wrong category (see the first error below). Confidence reflects how similar a ticket is to the training data, not whether the prediction is correct. That is why a person remains responsible for the final decision.

---

## 11. Error analysis

### 11.1 Wrong predictions that were accepted automatically

Five test tickets passed the review rules but were misclassified:

| Record | Ticket | True | Predicted | Confidence | Likely cause |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TS-0512 | "Refund for order 8802 still hasnt arrived after 3 weeks." | Billing | Delivery | 88.83% | Phrases like "order", "hasnt arrived" and "3 weeks" are strong delivery signals in the training data and outweighed "refund". |
| TS-0543 | "Never got the verification code by text." | Account Access | Technical Issue | 75.94% | The ticket describes an SMS delivery failure, which the model reads as a technical fault rather than an account problem. |
| TS-0548 | "Login page just refreshes when I press the button." | Account Access | Technical Issue | 88.88% | The user wants to log in, but the symptom (a page reload) looks like a software bug. |
| TS-0570 | "The courier tried to deliver while I was at work and I can't find a new time to pick it up." | Delivery | Account Access | 77.75% | There are no clear parcel keywords, and "find a new time" was read as appointment scheduling. |
| TS-0590 | "Great job on the latest release, search is so much faster now." | General Inquiry | Technical Issue | 76.46% | Positive feedback that uses software words ("release", "search", "faster"). |

Three themes appear:

- **Overlapping vocabulary:** words from one category appear in another category's tickets (TS-0512, TS-0543, TS-0548).
- **Missing keywords:** the ticket lacks the words the model relies on (TS-0570).
- **Not really a support request:** the message is feedback, not an issue (TS-0590).

Three more test errors were already flagged for review, so the review step caught them before they were accepted.

### 11.2 Limitations

1. **Closed set of categories.** Off-topic messages, such as general trivia, are forced into the closest category (usually General Inquiry).
2. **One label per ticket.** Multi-issue tickets get whichever issue has the stronger keywords.
3. **Priority is a heuristic.** It does not come from a trained model (see Section 3).
4. **Small test set.** With 100 test tickets, one ticket is one percentage point of accuracy. The results are a good estimate but not a precise one.

---

## 12. Verification

### 12.1 Data integrity

The SHA-256 hashes of `ticketsense_dataset.csv` and `ticketsense_test.csv` were recomputed and match the values in Section 5.2.

### 12.2 Metric discrepancies

Two differences between earlier drafts and the final audit were checked:

- **Test log loss (0.3524 vs 0.3187).** `scripts/evaluate_final_test_set.py` retrained the Naive Bayes model without a fixed cross-validation split, which gave 0.3524. The audit loaded the saved model (`models/category_model.joblib`) and got 0.3187. The saved model is what the app uses, so 0.3187 is the reported figure.
- **Test macro precision (0.9213 vs 0.9200).** The 0.9213 value was copied from a different experiment's table by mistake. The per-class test precisions are 0.90, 0.95, 0.95, 0.95 and 0.85, which average to 0.9200.

### 12.3 Automated and Special-Case Tests

- **Automated tests:** 45 of 45 pass across unit, integration, visualization, and headless UI suites.
- **Manual QA checklist** (`scripts/run_qa_manual_checklist.py`): all checks passed, with a final verdict of "READY FOR SUPERVISOR DEMO".
- **Special-case scenarios:**
  - Short inputs (< 5 characters) and empty strings are rejected with clear validation error messages.
  - Misspelled tickets (e.g. "cant log in to my accout bcuz password is incorect") are handled autonomously by the sublinear TF-IDF pipeline.
  - Ambiguous tickets crossing category boundaries (e.g. billing charges mixed with portal access issues) trigger the review-required threshold.
  - Out-of-category queries map to the closest semantic category under the closed-world constraint.
- **Demonstration scenarios:** The app includes pre-populated sample tickets covering normal autonomous prediction, borderline uncertain review routing, and human correction logging to `data/corrections/runtime_corrections.csv`.

The audit scripts and reports are kept locally in `docs/archive/final_independent_audit/` and are not tracked in git.

---

## 13. The application

`app.py` is a Streamlit app with four tabs:

1. **Single ticket:** paste a message to see the category, calibrated confidence, review status and heuristic priority. An expandable chart shows the probability for each category. An override form lets an agent correct the category and save it to `runtime_corrections.csv`.
2. **Batch CSV:** upload a CSV of tickets. Empty rows and rows shorter than 5 characters are rejected, and the table shows the row number and reason. Valid rows get a category, confidence and review reason, and the results can be downloaded as CSV.
3. **Performance and governance:** shows training size (600), validation accuracy (93.0%), macro F1 (0.9297), the confusion matrix, and a note that priority training is paused.
4. **Corrections log:** shows every correction saved in `data/corrections/runtime_corrections.csv`.

---

## 14. Tools and repository

| Tool | Version | Use |
| :--- | :--- | :--- |
| Python | 3.13 | Runtime |
| pandas | 2.2.3 | Data loading and CSV handling |
| numpy | 2.2.3 | Numerical work and Brier score |
| scikit-learn | 1.6.1 | Pipeline, TF-IDF, models, calibration, metrics |
| joblib | 1.4.2 | Saving and loading the model |
| Streamlit | 1.42.1 | Web app |
| pytest | 9.2.0 | Tests |

See the README for the repository layout and setup instructions. Reports and charts are in `reports/`, including `EXPERIMENT_RESULTS_LOG.json`, `CATEGORY_MODEL_EVALUATION.md`, `category_validation_errors.csv` and the six chart images in `reports/visualizations/`.

---

## 15. Status and next steps

| Component | Status |
| :--- | :--- |
| Category classifier (TF-IDF + calibrated Naive Bayes) | Complete: 92% test accuracy, 90% handled automatically |
| Human review rules | Complete |
| Synthetic data and quality checks | Complete: no duplicates, no leakage |
| Streamlit app | Complete |
| Automated tests | Complete: 45 of 45 pass |
| Priority model | On hold: waiting for Gate G1 approval. A keyword heuristic is shown in the meantime. |
| Filter for off-topic messages | Future work: reject non-support messages before classification |
| Multi-label tickets | Future work: handle tickets that contain more than one issue |

---

## 16. Conclusion

The original model was accurate but unsure of itself, which sent 93% of tickets to human review. Adding 200 targeted synthetic tickets and calibrating the probabilities of a TF-IDF Naive Bayes model fixed this. On the held-out test set the system reaches 92% accuracy, handles 90% of tickets automatically at 94.44% accuracy, and sends the other 10% to a person.

The remaining errors come mostly from tickets that mix vocabulary from two categories. Confidence is a useful signal but not a guarantee, so a person should still review flagged tickets and spot-check the automatic ones. The code, data splits, model and tests are reproducible, and the project is ready for supervisor review.
