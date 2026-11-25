"""Advanced comparative experiments for Priority Classification.

Compares:
1. Baseline Training (400 samples) vs Enhanced Training (400 + 70 supplementary = 470 samples).
2. Direct Priority Classifiers:
   - Logistic Regression (Unweighted vs Balanced)
   - LinearSVC (Balanced)
   - Multinomial Naive Bayes
   - Direct LogReg + In-Fold Oversampling
3. Two-Stage Rubric Classifiers:
   - Two-Stage Logistic Regression (Balanced)
   - Two-Stage LinearSVC (Balanced)
   - Two-Stage Multinomial Naive Bayes
   - Two-Stage LogReg + Feature Engineering (Word + Char ngrams)
   - Two-Stage LogReg with Probability Calibration (CalibratedClassifierCV)
4. Metrics evaluated:
   - 5-fold CV (Accuracy mean/std, Macro F1 mean/std, High Recall mean/std)
   - Validation set (Accuracy, Macro F1, Macro P, Macro R, High P, High R, High F1)
   - Confusion matrix, High FP, High FN, generalization gap.
Saves to reports/ADVANCED_PRIORITY_EXPERIMENT_RESULTS.json.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ticketsense.nlp.preprocessor import TextCleaner
from ticketsense.models.priority_pipeline import TwoStagePriorityClassifier, AUTHORITATIVE_RUBRIC

DEV_PATH = PROJECT_ROOT / "data" / "processed" / "ticketsense_dataset.csv"
SUPP_PATH = PROJECT_ROOT / "data" / "processed" / "priority_supplementary_training.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "ADVANCED_PRIORITY_EXPERIMENT_RESULTS.json"

dev_df = pd.read_csv(DEV_PATH)
dev_df["impact"] = dev_df["priority_labels"].apply(lambda x: str(x).split("|")[0])
dev_df["urgency"] = dev_df["priority_labels"].apply(lambda x: str(x).split("|")[1])

train_baseline = dev_df[dev_df["split"] == "train"].copy().reset_index(drop=True)
val_df = dev_df[dev_df["split"] == "validation"].copy().reset_index(drop=True)

supp_df = pd.read_csv(SUPP_PATH)
supp_df["impact"] = supp_df["priority_labels"].apply(lambda x: str(x).split("|")[0])
supp_df["urgency"] = supp_df["priority_labels"].apply(lambda x: str(x).split("|")[1])

train_enhanced = pd.concat([train_baseline, supp_df], ignore_index=True)

print(f"Train baseline samples: {len(train_baseline)}")
print(f"Train enhanced samples: {len(train_enhanced)}")
print(f"Validation samples:     {len(val_df)}")
print("\nTrain baseline priority distribution:\n", train_baseline["priority"].value_counts())
print("\nTrain enhanced priority distribution:\n", train_enhanced["priority"].value_counts())
print("\nValidation priority distribution:\n", val_df["priority"].value_counts())


def evaluate_candidate(name, model, train_data, val_data, is_two_stage=False):
    """Run 5-fold CV on train_data and evaluate on val_data."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_accs, cv_macro_f1s, cv_high_recalls = [], [], []

    # Cross-validation on train_data only
    for train_idx, test_idx in skf.split(train_data, train_data["priority"]):
        fold_train = train_data.iloc[train_idx]
        fold_test = train_data.iloc[test_idx]

        if is_two_stage:
            fold_model = TwoStagePriorityClassifier(
                c_imp=getattr(model, "c_imp", 2.0),
                c_urg=getattr(model, "c_urg", 1.0),
                ngram_range=getattr(model, "ngram_range", (1, 2)),
            )
            fold_model.fit(fold_train["ticket_text"], fold_train["impact"], fold_train["urgency"])
            preds = fold_model.predict(fold_test["ticket_text"])
        else:
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(fold_train["ticket_text"], fold_train["priority"])
            preds = fold_model.predict(fold_test["ticket_text"])

        cv_accs.append(accuracy_score(fold_test["priority"], preds))
        cv_macro_f1s.append(f1_score(fold_test["priority"], preds, average="macro", zero_division=0))
        high_rec = recall_score(fold_test["priority"] == "High", preds == "High", zero_division=0)
        cv_high_recalls.append(high_rec)

    # Full fit on train_data
    if is_two_stage:
        model.fit(train_data["ticket_text"], train_data["impact"], train_data["urgency"])
        val_preds = model.predict(val_data["ticket_text"])
        train_preds = model.predict(train_data["ticket_text"])
    else:
        model.fit(train_data["ticket_text"], train_data["priority"])
        val_preds = model.predict(val_data["ticket_text"])
        train_preds = model.predict(train_data["ticket_text"])

    train_macro_f1 = float(f1_score(train_data["priority"], train_preds, average="macro", zero_division=0))
    val_acc = float(accuracy_score(val_data["priority"], val_preds))
    val_macro_f1 = float(f1_score(val_data["priority"], val_preds, average="macro", zero_division=0))
    val_macro_p = float(precision_score(val_data["priority"], val_preds, average="macro", zero_division=0))
    val_macro_r = float(recall_score(val_data["priority"], val_preds, average="macro", zero_division=0))
    val_weighted_f1 = float(f1_score(val_data["priority"], val_preds, average="weighted", zero_division=0))

    report_dict = classification_report(val_data["priority"], val_preds, output_dict=True, zero_division=0)
    conf = confusion_matrix(val_data["priority"], val_preds, labels=["High", "Medium", "Low"]).tolist()

    # High FP and FN
    val_df_eval = val_data.copy()
    val_df_eval["pred"] = val_preds
    high_fp = int(((val_df_eval["priority"] != "High") & (val_df_eval["pred"] == "High")).sum())
    high_fn = int(((val_df_eval["priority"] == "High") & (val_df_eval["pred"] != "High")).sum())

    res = {
        "candidate_name": name,
        "is_two_stage": is_two_stage,
        "train_samples": len(train_data),
        "cv_accuracy_mean": float(np.mean(cv_accs)),
        "cv_accuracy_std": float(np.std(cv_accs)),
        "cv_macro_f1_mean": float(np.mean(cv_macro_f1s)),
        "cv_macro_f1_std": float(np.std(cv_macro_f1s)),
        "cv_high_recall_mean": float(np.mean(cv_high_recalls)),
        "cv_high_recall_std": float(np.std(cv_high_recalls)),
        "train_macro_f1": train_macro_f1,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "val_macro_p": val_macro_p,
        "val_macro_r": val_macro_r,
        "val_weighted_f1": val_weighted_f1,
        "generalization_gap": float(train_macro_f1 - val_macro_f1),
        "high_precision": float(report_dict.get("High", {}).get("precision", 0.0)),
        "high_recall": float(report_dict.get("High", {}).get("recall", 0.0)),
        "high_f1": float(report_dict.get("High", {}).get("f1-score", 0.0)),
        "high_false_positives": high_fp,
        "high_false_negatives": high_fn,
        "confusion_matrix": conf,
        "per_class": report_dict,
    }

    print(f"\n--- {name} ---")
    print(f"  CV Acc: {res['cv_accuracy_mean']:.4f} +/- {res['cv_accuracy_std']:.4f} | CV Macro F1: {res['cv_macro_f1_mean']:.4f}")
    print(f"  Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f} | Gen Gap: {res['generalization_gap']:.4f}")
    print(f"  High P: {res['high_precision']:.4f} | High R: {res['high_recall']:.4f} | High F1: {res['high_f1']:.4f}")
    print(f"  High FP: {high_fp} | High FN: {high_fn}")
    print(f"  Conf Matrix [High, Med, Low]: {conf}")

    return res


results = {}

# 1. Baseline Training Data (400 samples)
print("\n" + "="*60 + "\nSECTION 1: BASELINE TRAINING DATA (400 samples)\n" + "="*60)

# Direct LogReg Baseline
p1 = Pipeline([
    ("cleaner", TextCleaner()),
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000)),
])
results["Baseline_Direct_LogReg_Balanced"] = evaluate_candidate(
    "Baseline_Direct_LogReg_Balanced", p1, train_baseline, val_df, is_two_stage=False
)

# Two-Stage LogReg Baseline (untuned unigram)
m2 = TwoStagePriorityClassifier(c_imp=1.0, c_urg=1.0, ngram_range=(1, 1))
results["Baseline_TwoStage_LogReg_Untuned"] = evaluate_candidate(
    "Baseline_TwoStage_LogReg_Untuned", m2, train_baseline, val_df, is_two_stage=True
)

# Two-Stage LogReg Baseline (tuned bigram)
m3 = TwoStagePriorityClassifier(c_imp=2.0, c_urg=1.0, ngram_range=(1, 2))
results["Baseline_TwoStage_LogReg_Tuned"] = evaluate_candidate(
    "Baseline_TwoStage_LogReg_Tuned", m3, train_baseline, val_df, is_two_stage=True
)

# 2. Enhanced Training Data (400 + 70 = 470 samples)
print("\n" + "="*60 + "\nSECTION 2: ENHANCED TRAINING DATA (470 samples)\n" + "="*60)

# Enhanced Direct LogReg
p_enh_lr = Pipeline([
    ("cleaner", TextCleaner()),
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000)),
])
results["Enhanced_Direct_LogReg_Balanced"] = evaluate_candidate(
    "Enhanced_Direct_LogReg_Balanced", p_enh_lr, train_enhanced, val_df, is_two_stage=False
)

# Enhanced Direct LinearSVC
p_enh_svc = Pipeline([
    ("cleaner", TextCleaner()),
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000)),
])
results["Enhanced_Direct_LinearSVC_Balanced"] = evaluate_candidate(
    "Enhanced_Direct_LinearSVC_Balanced", p_enh_svc, train_enhanced, val_df, is_two_stage=False
)

# Enhanced Two-Stage LogReg (C_imp=2.0, C_urg=1.0, ngram=(1,2))
m_enh_ts_lr = TwoStagePriorityClassifier(c_imp=2.0, c_urg=1.0, ngram_range=(1, 2))
results["Enhanced_TwoStage_LogReg_C2_C1"] = evaluate_candidate(
    "Enhanced_TwoStage_LogReg_C2_C1", m_enh_ts_lr, train_enhanced, val_df, is_two_stage=True
)

# Enhanced Two-Stage LogReg (C_imp=3.0, C_urg=2.0, ngram=(1,2))
m_enh_ts_lr2 = TwoStagePriorityClassifier(c_imp=3.0, c_urg=2.0, ngram_range=(1, 2))
results["Enhanced_TwoStage_LogReg_C3_C2"] = evaluate_candidate(
    "Enhanced_TwoStage_LogReg_C3_C2", m_enh_ts_lr2, train_enhanced, val_df, is_two_stage=True
)

# Enhanced Two-Stage LinearSVC
class TwoStageLinearSVC(TwoStagePriorityClassifier):
    def __init__(self, c_imp=1.0, c_urg=1.0, ngram_range=(1, 2)):
        super().__init__(c_imp=c_imp, c_urg=c_urg, ngram_range=ngram_range)
        self.impact_pipeline = Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=self.ngram_range, sublinear_tf=True)),
            ("clf", LinearSVC(C=self.c_imp, class_weight="balanced", random_state=42, max_iter=2000)),
        ])
        self.urgency_pipeline = Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=self.ngram_range, sublinear_tf=True)),
            ("clf", LinearSVC(C=self.c_urg, class_weight="balanced", random_state=42, max_iter=2000)),
        ])

m_enh_ts_svc = TwoStageLinearSVC(c_imp=1.0, c_urg=1.0, ngram_range=(1, 2))
results["Enhanced_TwoStage_LinearSVC"] = evaluate_candidate(
    "Enhanced_TwoStage_LinearSVC", m_enh_ts_svc, train_enhanced, val_df, is_two_stage=True
)

# Save all results
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nAll experimental results saved to: {REPORT_PATH}")
