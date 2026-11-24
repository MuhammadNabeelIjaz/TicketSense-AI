"""Final evaluation of baseline vs improved models on the untouched test dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
from sklearn.preprocessing import label_binarize

from ticketsense.config import (
    APPROVED_CATEGORIES,
    TEST_DATASET_PATH,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
)
from ticketsense.nlp.preprocessor import TextCleaner
from ticketsense.data.loader import load_dataset_splits

SYNTHETIC_DATA_PATH = PROCESSED_DATA_DIR / "ticketsense_synthetic_expansion_v1.csv"


def multiclass_brier_score(y_true, y_prob, classes):
    Y = label_binarize(y_true, classes=classes)
    return float(np.mean(np.sum((y_prob - Y) ** 2, axis=1)))


def evaluate_on_test(model, X_test, y_test, threshold=0.65, margin=0.15):
    y_pred = model.predict(X_test)
    probas = model.predict_proba(X_test)
    classes = list(model.classes_)

    acc = float(accuracy_score(y_test, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(y_test, y_pred, average=None, labels=APPROVED_CATEGORIES, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=APPROVED_CATEGORIES)

    brier = multiclass_brier_score(y_test, probas, classes)
    try:
        ll = float(log_loss(y_test, probas, labels=classes))
    except Exception:
        ll = None

    confidences = np.max(probas, axis=1)
    mean_conf = float(np.mean(confidences))

    uncertain_count = 0
    auto_correct = 0
    auto_incorrect = 0

    for i in range(len(y_test)):
        sorted_p = sorted(probas[i], reverse=True)
        top_p = sorted_p[0]
        second_p = sorted_p[1]
        m = top_p - second_p

        is_u = (top_p < threshold) or (m < margin)
        if is_u:
            uncertain_count += 1
        else:
            if y_pred[i] == y_test[i]:
                auto_correct += 1
            else:
                auto_incorrect += 1

    review_rate = uncertain_count / len(y_test)
    auto_cov = (len(y_test) - uncertain_count) / len(y_test)
    auto_count = len(y_test) - uncertain_count
    auto_acc = (auto_correct / auto_count) if auto_count > 0 else 0.0
    incorrect_auto_rate = auto_incorrect / len(y_test)

    return {
        "accuracy": acc,
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "per_category_f1": {cat: float(f) for cat, f in zip(APPROVED_CATEGORIES, per_f1)},
        "per_category_precision": {cat: float(pr) for cat, pr in zip(APPROVED_CATEGORIES, per_p)},
        "per_category_recall": {cat: float(rc) for cat, rc in zip(APPROVED_CATEGORIES, per_r)},
        "confusion_matrix": cm.tolist(),
        "brier_score": brier,
        "log_loss": ll,
        "mean_confidence": mean_conf,
        "review_rate": review_rate,
        "autonomous_coverage": auto_cov,
        "autonomous_count": auto_count,
        "autonomous_accuracy": auto_acc,
        "incorrect_autonomous_count": auto_incorrect,
        "incorrect_autonomous_rate": incorrect_auto_rate,
    }


def main():
    orig_train_df, _ = load_dataset_splits()
    syn_df = pd.read_csv(SYNTHETIC_DATA_PATH)
    exp_train_df = pd.concat([orig_train_df, syn_df], ignore_index=True)
    test_df = pd.read_csv(TEST_DATASET_PATH)

    X_train_orig = orig_train_df["ticket_text"].tolist()
    y_train_orig = orig_train_df["category"].tolist()

    X_train_exp = exp_train_df["ticket_text"].tolist()
    y_train_exp = exp_train_df["category"].tolist()

    X_test = test_df["ticket_text"].tolist()
    y_test = test_df["category"].tolist()

    print("=" * 80)
    print("FINAL EVALUATION ON UNTOUCHED TEST DATASET (N=100)")
    print("=" * 80)

    # 1. Baseline Model (Original 400, Uncalibrated LR C=1.0)
    m_base = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")),
    ])
    m_base.fit(X_train_orig, y_train_orig)
    res_base = evaluate_on_test(m_base, X_test, y_test, threshold=0.65)

    # 2. Optimized Expanded Model: Calibrated MultinomialNB (alpha=0.1) on Expanded 600 records
    m_nb = CalibratedClassifierCV(
        estimator=Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        method="sigmoid",
        cv=5,
    )
    m_nb.fit(X_train_exp, y_train_exp)
    res_nb = evaluate_on_test(m_nb, X_test, y_test, threshold=0.65)

    # 3. Optimized Expanded Model: Calibrated LinearSVC (C=1.0) on Expanded 600 records
    m_svc = CalibratedClassifierCV(
        estimator=Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
            ("clf", LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        method="sigmoid",
        cv=5,
    )
    m_svc.fit(X_train_exp, y_train_exp)
    res_svc = evaluate_on_test(m_svc, X_test, y_test, threshold=0.65)

    # 4. Optimized Expanded Model: Calibrated LogisticRegression (C=5.0) on Expanded 600 records
    m_lr = CalibratedClassifierCV(
        estimator=Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
            ("clf", LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")),
        ]),
        method="sigmoid",
        cv=5,
    )
    m_lr.fit(X_train_exp, y_train_exp)
    res_lr = evaluate_on_test(m_lr, X_test, y_test, threshold=0.65)

    models_eval = [
        ("Baseline (Uncalibrated LR, N=400)", res_base),
        ("Expanded Calibrated LR (C=5.0, N=600)", res_lr),
        ("Expanded Calibrated LinearSVC (C=1.0, N=600)", res_svc),
        ("Expanded Calibrated MultinomialNB (alpha=0.1, N=600)", res_nb),
    ]

    print(f"{'Model':<42} | {'Test Acc':<8} | {'Macro F1':<8} | {'Brier':<7} | {'Mean Conf':<9} | {'Auto Cov':<8} | {'Review %':<8} | {'Auto Acc':<8}")
    print("-" * 115)
    for name, r in models_eval:
        print(f"{name:<42} | {r['accuracy']*100:>7.1f}% | {r['macro_f1']:>8.4f} | {r['brier_score']:>7.4f} | {r['mean_confidence']*100:>8.1f}% | {r['autonomous_coverage']*100:>7.1f}% | {r['review_rate']*100:>7.1f}% | {r['autonomous_accuracy']*100:>7.1f}%")
    print("=" * 115)

    print("\n=== DETAILED PER-CATEGORY F1 ON UNTOUCHED TEST SET ===")
    for cat in APPROVED_CATEGORIES:
        f_base = res_base["per_category_f1"][cat]
        f_lr = res_lr["per_category_f1"][cat]
        f_svc = res_svc["per_category_f1"][cat]
        f_nb = res_nb["per_category_f1"][cat]
        print(f"{cat:<18}: Baseline={f_base:.3f} | LR(600)={f_lr:.3f} | SVC(600)={f_svc:.3f} | NB(600)={f_nb:.3f}")

    print("\n=== CONFUSION MATRIX (Expanded Calibrated MultinomialNB on Test Set) ===")
    cm_df = pd.DataFrame(res_nb["confusion_matrix"], index=APPROVED_CATEGORIES, columns=APPROVED_CATEGORIES)
    print(cm_df)

    print("\n=== CONFUSION MATRIX (Expanded Calibrated LinearSVC on Test Set) ===")
    cm_svc_df = pd.DataFrame(res_svc["confusion_matrix"], index=APPROVED_CATEGORIES, columns=APPROVED_CATEGORIES)
    print(cm_svc_df)


if __name__ == "__main__":
    main()
