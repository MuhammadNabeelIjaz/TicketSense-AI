"""Evaluate threshold trade-offs and automation coverage vs reliability on validation data."""

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

from ticketsense.config import PROCESSED_DATA_DIR, RANDOM_STATE
from ticketsense.nlp.preprocessor import TextCleaner
from ticketsense.data.loader import load_dataset_splits

SYNTHETIC_DATA_PATH = PROCESSED_DATA_DIR / "ticketsense_synthetic_expansion_v1.csv"


def sweep_thresholds(model, X_val, y_val, model_name, margin=0.15):
    y_pred = model.predict(X_val)
    probas = model.predict_proba(X_val)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    records = []

    for th in thresholds:
        uncertain_count = 0
        auto_correct = 0
        auto_incorrect = 0

        for i in range(len(y_val)):
            sorted_p = sorted(probas[i], reverse=True)
            top_p = sorted_p[0]
            second_p = sorted_p[1]
            m = top_p - second_p

            is_u = (top_p < th) or (m < margin)
            if is_u:
                uncertain_count += 1
            else:
                if y_pred[i] == y_val[i]:
                    auto_correct += 1
                else:
                    auto_incorrect += 1

        auto_count = len(y_val) - uncertain_count
        review_rate = uncertain_count / len(y_val)
        auto_cov = auto_count / len(y_val)
        auto_acc = (auto_correct / auto_count) if auto_count > 0 else 0.0
        incorrect_auto_rate = auto_incorrect / len(y_val)

        records.append({
            "model": model_name,
            "threshold": th,
            "margin": margin,
            "auto_coverage": auto_cov,
            "auto_count": auto_count,
            "review_rate": review_rate,
            "autonomous_accuracy": auto_acc,
            "incorrect_auto_count": auto_incorrect,
            "incorrect_auto_rate": incorrect_auto_rate,
        })

    return pd.DataFrame(records)


def main():
    orig_train, val_df = load_dataset_splits()
    syn_df = pd.read_csv(SYNTHETIC_DATA_PATH)
    exp_train = pd.concat([orig_train, syn_df], ignore_index=True)

    X_train = exp_train["ticket_text"].tolist()
    y_train = exp_train["category"].tolist()
    X_val = val_df["ticket_text"].tolist()
    y_val = val_df["category"].tolist()

    # Model 1: Calibrated Logistic Regression (C=5.0)
    m1 = CalibratedClassifierCV(
        estimator=Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
            ("clf", LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")),
        ]),
        method="sigmoid",
        cv=5,
    )
    m1.fit(X_train, y_train)

    # Model 2: Calibrated LinearSVC (C=1.0)
    m2 = CalibratedClassifierCV(
        estimator=Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
            ("clf", LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        method="sigmoid",
        cv=5,
    )
    m2.fit(X_train, y_train)

    # Model 3: Calibrated MultinomialNB (alpha=0.1)
    m3 = CalibratedClassifierCV(
        estimator=Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        method="sigmoid",
        cv=5,
    )
    m3.fit(X_train, y_train)

    df1 = sweep_thresholds(m1, X_val, y_val, "Calibrated-LR(C=5.0)")
    df2 = sweep_thresholds(m2, X_val, y_val, "Calibrated-LinearSVC(C=1.0)")
    df3 = sweep_thresholds(m3, X_val, y_val, "Calibrated-MultinomialNB(alpha=0.1)")

    comb = pd.concat([df1, df2, df3], ignore_index=True)
    print("=" * 95)
    print("THRESHOLD SWEEP & COVERAGE-RELIABILITY TRADE-OFF MATRIX (VALIDATION DATA)")
    print("=" * 95)
    print(f"{'Model':<30} | {'Thresh':<6} | {'Coverage':<8} | {'Review Rate':<11} | {'Auto Acc':<8} | {'Incorrect Auto':<14}")
    print("-" * 95)
    for _, r in comb.iterrows():
        print(f"{r['model']:<30} | {r['threshold']:<6.2f} | {r['auto_coverage']*100:>6.1f}%  | {r['review_rate']*100:>9.1f}%  | {r['autonomous_accuracy']*100:>6.1f}%  | {int(r['incorrect_auto_count']):>2} tickets ({r['incorrect_auto_rate']*100:.1f}%)")
    print("=" * 95)


if __name__ == "__main__":
    main()
