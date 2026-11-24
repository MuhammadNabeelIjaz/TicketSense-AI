"""Systematic ML Optimization Experiment Suite for TicketSense-AI.

Runs distinct iterative experiments across algorithms, TF-IDF representations,
calibration techniques, and dataset expansions using Stratified 5-Fold CV
and out-of-sample validation evaluation.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from sklearn.preprocessing import label_binarize

from ticketsense.config import (
    APPROVED_CATEGORIES,
    TRAIN_VAL_DATASET_PATH,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
)
from ticketsense.nlp.preprocessor import TextCleaner
from ticketsense.data.loader import load_dataset_splits

SYNTHETIC_DATA_PATH = PROCESSED_DATA_DIR / "ticketsense_synthetic_expansion_v1.csv"
EXPERIMENT_LOG_PATH = REPORTS_DIR / "EXPERIMENT_RESULTS_LOG.json"


def multiclass_brier_score(y_true, y_prob, classes):
    Y = label_binarize(y_true, classes=classes)
    return float(np.mean(np.sum((y_prob - Y) ** 2, axis=1)))


def evaluate_model_pipeline(model, X_train, y_train, X_val, y_val, threshold=0.65, margin=0.15):
    # 5-Fold Stratified CV on training split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    # Fit model on training split
    model.fit(X_train, y_train)

    # Validation evaluation
    y_pred = model.predict(X_val)
    probas = model.predict_proba(X_val)
    classes = list(model.classes_)

    acc = float(accuracy_score(y_val, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="macro", zero_division=0)
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(y_val, y_pred, average=None, labels=APPROVED_CATEGORIES, zero_division=0)
    
    brier = multiclass_brier_score(y_val, probas, classes)
    try:
        ll = float(log_loss(y_val, probas, labels=classes))
    except Exception:
        ll = None

    confidences = np.max(probas, axis=1)
    mean_conf = float(np.mean(confidences))

    uncertain_count = 0
    autonomous_correct = 0
    autonomous_incorrect = 0

    for i in range(len(y_val)):
        sorted_p = sorted(probas[i], reverse=True)
        top_p = sorted_p[0]
        second_p = sorted_p[1]
        m = top_p - second_p

        is_u = (top_p < threshold) or (m < margin)
        if is_u:
            uncertain_count += 1
        else:
            if y_pred[i] == y_val[i]:
                autonomous_correct += 1
            else:
                autonomous_incorrect += 1

    review_rate = uncertain_count / len(y_val)
    auto_coverage = (len(y_val) - uncertain_count) / len(y_val)
    auto_count = len(y_val) - uncertain_count
    auto_acc = (autonomous_correct / auto_count) if auto_count > 0 else 0.0

    return {
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "val_accuracy": acc,
        "val_macro_precision": float(p),
        "val_macro_recall": float(r),
        "val_macro_f1": float(f1),
        "per_category_f1": {cat: float(f) for cat, f in zip(APPROVED_CATEGORIES, per_f1)},
        "brier_score": brier,
        "log_loss": ll,
        "mean_confidence": mean_conf,
        "review_rate": review_rate,
        "autonomous_coverage": auto_coverage,
        "autonomous_accuracy": auto_acc,
        "incorrect_autonomous_count": autonomous_incorrect,
    }


def run_all_experiments():
    orig_train_df, val_df = load_dataset_splits()
    syn_df = pd.read_csv(SYNTHETIC_DATA_PATH)

    # Combine original train + synthetic train
    expanded_train_df = pd.concat([orig_train_df, syn_df], ignore_index=True)

    X_train_orig = orig_train_df["ticket_text"].tolist()
    y_train_orig = orig_train_df["category"].tolist()

    X_train_exp = expanded_train_df["ticket_text"].tolist()
    y_train_exp = expanded_train_df["category"].tolist()

    X_val = val_df["ticket_text"].tolist()
    y_val = val_df["category"].tolist()

    print("=" * 80)
    print("EXECUTING SYSTEMATIC ML EXPERIMENT MATRIX")
    print(f"Original Train: {len(X_train_orig)} records | Expanded Train: {len(X_train_exp)} records | Val: {len(X_val)} records")
    print("=" * 80)

    experiments = []

    # -------------------------------------------------------------------------
    # EXP-00: Pure Baseline (Original Data 400, Uncalibrated Logistic Regression C=1.0)
    # -------------------------------------------------------------------------
    print("\nRunning EXP-00: Baseline (Original 400, Uncalibrated LR C=1.0)...")
    pipe_00 = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")),
    ])
    res_00 = evaluate_model_pipeline(pipe_00, X_train_orig, y_train_orig, X_val, y_val)
    experiments.append({
        "id": "EXP-00",
        "description": "Baseline: Original 400 records, uncalibrated Logistic Regression (C=1.0)",
        "train_size": len(X_train_orig),
        "data_version": "v1_original",
        "algorithm": "LogisticRegression(C=1.0)",
        "calibration": "none",
        "results": res_00,
    })

    # -------------------------------------------------------------------------
    # EXP-01: Baseline + Sigmoid Calibration (Original 400)
    # -------------------------------------------------------------------------
    print("Running EXP-01: Baseline + Sigmoid Calibration (Original 400)...")
    base_01 = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")),
    ])
    cal_01 = CalibratedClassifierCV(estimator=base_01, method="sigmoid", cv=5)
    res_01 = evaluate_model_pipeline(cal_01, X_train_orig, y_train_orig, X_val, y_val)
    experiments.append({
        "id": "EXP-01",
        "description": "Original 400 records, Sigmoid Calibrated LR (C=5.0, min_df=1, max_df=0.85)",
        "train_size": len(X_train_orig),
        "data_version": "v1_original",
        "algorithm": "LogisticRegression(C=5.0)",
        "calibration": "sigmoid_5cv",
        "results": res_01,
    })

    # -------------------------------------------------------------------------
    # EXP-02: Expanded Data (600) + Uncalibrated Logistic Regression
    # -------------------------------------------------------------------------
    print("Running EXP-02: Expanded Data (600) + Uncalibrated LR (C=1.0)...")
    pipe_02 = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")),
    ])
    res_02 = evaluate_model_pipeline(pipe_02, X_train_exp, y_train_exp, X_val, y_val)
    experiments.append({
        "id": "EXP-02",
        "description": "Expanded 600 records, Uncalibrated LR (C=1.0)",
        "train_size": len(X_train_exp),
        "data_version": "v2_expanded",
        "algorithm": "LogisticRegression(C=1.0)",
        "calibration": "none",
        "results": res_02,
    })

    # -------------------------------------------------------------------------
    # EXP-03: Expanded Data (600) + Tuned TF-IDF + Sigmoid Calibrated LR (C=5.0)
    # -------------------------------------------------------------------------
    print("Running EXP-03: Expanded Data (600) + Tuned TF-IDF + Sigmoid LR (C=5.0)...")
    base_03 = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")),
    ])
    cal_03 = CalibratedClassifierCV(estimator=base_03, method="sigmoid", cv=5)
    res_03 = evaluate_model_pipeline(cal_03, X_train_exp, y_train_exp, X_val, y_val)
    experiments.append({
        "id": "EXP-03",
        "description": "Expanded 600 records, Tuned TF-IDF + Sigmoid LR (C=5.0)",
        "train_size": len(X_train_exp),
        "data_version": "v2_expanded",
        "algorithm": "LogisticRegression(C=5.0)",
        "calibration": "sigmoid_5cv",
        "results": res_03,
    })

    # -------------------------------------------------------------------------
    # EXP-04: Expanded Data (600) + Linear SVM (LinearSVC) + Sigmoid Calibration
    # -------------------------------------------------------------------------
    print("Running EXP-04: Expanded Data (600) + Linear SVM (LinearSVC) + Sigmoid Calibration...")
    base_04 = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    cal_04 = CalibratedClassifierCV(estimator=base_04, method="sigmoid", cv=5)
    res_04 = evaluate_model_pipeline(cal_04, X_train_exp, y_train_exp, X_val, y_val)
    experiments.append({
        "id": "EXP-04",
        "description": "Expanded 600 records, LinearSVC (C=1.0) + Sigmoid Calibration",
        "train_size": len(X_train_exp),
        "data_version": "v2_expanded",
        "algorithm": "LinearSVC(C=1.0)",
        "calibration": "sigmoid_5cv",
        "results": res_04,
    })

    # -------------------------------------------------------------------------
    # EXP-05: Expanded Data (600) + Multinomial Naive Bayes + Sigmoid Calibration
    # -------------------------------------------------------------------------
    print("Running EXP-05: Expanded Data (600) + Multinomial Naive Bayes + Sigmoid Calibration...")
    base_05 = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", MultinomialNB(alpha=0.1)),
    ])
    cal_05 = CalibratedClassifierCV(estimator=base_05, method="sigmoid", cv=5)
    res_05 = evaluate_model_pipeline(cal_05, X_train_exp, y_train_exp, X_val, y_val)
    experiments.append({
        "id": "EXP-05",
        "description": "Expanded 600 records, MultinomialNB (alpha=0.1) + Sigmoid Calibration",
        "train_size": len(X_train_exp),
        "data_version": "v2_expanded",
        "algorithm": "MultinomialNB(alpha=0.1)",
        "calibration": "sigmoid_5cv",
        "results": res_05,
    })

    # -------------------------------------------------------------------------
    # EXP-06: Expanded Data (600) + SGDClassifier (log_loss, elasticnet) + Sigmoid Calibration
    # -------------------------------------------------------------------------
    print("Running EXP-06: Expanded Data (600) + SGDClassifier (log_loss) + Sigmoid Calibration...")
    base_06 = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=1e-4, random_state=RANDOM_STATE, max_iter=2000)),
    ])
    cal_06 = CalibratedClassifierCV(estimator=base_06, method="sigmoid", cv=5)
    res_06 = evaluate_model_pipeline(cal_06, X_train_exp, y_train_exp, X_val, y_val)
    experiments.append({
        "id": "EXP-06",
        "description": "Expanded 600 records, SGDClassifier(loss=log_loss) + Sigmoid Calibration",
        "train_size": len(X_train_exp),
        "data_version": "v2_expanded",
        "algorithm": "SGDClassifier(log_loss)",
        "calibration": "sigmoid_5cv",
        "results": res_06,
    })

    # -------------------------------------------------------------------------
    # EXP-07: Expanded Data (600) + Logistic Regression with Isotonic Calibration
    # -------------------------------------------------------------------------
    print("Running EXP-07: Expanded Data (600) + Logistic Regression with Isotonic Calibration...")
    base_07 = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")),
    ])
    cal_07 = CalibratedClassifierCV(estimator=base_07, method="isotonic", cv=5)
    res_07 = evaluate_model_pipeline(cal_07, X_train_exp, y_train_exp, X_val, y_val)
    experiments.append({
        "id": "EXP-07",
        "description": "Expanded 600 records, Tuned TF-IDF + Isotonic Calibrated LR (C=5.0)",
        "train_size": len(X_train_exp),
        "data_version": "v2_expanded",
        "algorithm": "LogisticRegression(C=5.0)",
        "calibration": "isotonic_5cv",
        "results": res_07,
    })

    # -------------------------------------------------------------------------
    # EXP-08: Expanded Data (600) + Optimized LR (C=8.0, sublinear_tf, ngram (1,2), max_features=5000)
    # -------------------------------------------------------------------------
    print("Running EXP-08: Expanded Data (600) + Optimized LR (C=8.0, max_features=5000) + Sigmoid...")
    base_08 = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, max_features=5000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", LogisticRegression(C=8.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")),
    ])
    cal_08 = CalibratedClassifierCV(estimator=base_08, method="sigmoid", cv=5)
    res_08 = evaluate_model_pipeline(cal_08, X_train_exp, y_train_exp, X_val, y_val)
    experiments.append({
        "id": "EXP-08",
        "description": "Expanded 600 records, Tuned TF-IDF (max_features=5000) + Sigmoid LR (C=8.0)",
        "train_size": len(X_train_exp),
        "data_version": "v2_expanded",
        "algorithm": "LogisticRegression(C=8.0)",
        "calibration": "sigmoid_5cv",
        "results": res_08,
    })

    # Save complete JSON log
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'ID':<7} | {'Train':<5} | {'Algorithm & Calibration':<36} | {'CV Acc':<7} | {'Val Acc':<7} | {'Macro F1':<8} | {'Brier':<7} | {'Auto Cov':<8} | {'Auto Acc':<8}")
    print("-" * 105)
    for exp in experiments:
        r = exp["results"]
        desc = f"{exp['algorithm']} ({exp['calibration']})"[:36]
        print(f"{exp['id']:<7} | {exp['train_size']:<5} | {desc:<36} | {r['cv_accuracy_mean']:.4f}  | {r['val_accuracy']:.4f}  | {r['val_macro_f1']:.4f}   | {r['brier_score']:.4f}  | {r['autonomous_coverage']*100:.1f}%    | {r['autonomous_accuracy']*100:.1f}%")
    print("=" * 105)


if __name__ == "__main__":
    run_all_experiments()
