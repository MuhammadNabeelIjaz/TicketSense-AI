"""Comprehensive experimental benchmark for Priority Classification architectures.

Compares:
1. Approach A: Direct Priority Classification (Logistic Regression, Linear SVM, Naive Bayes, Random Forest)
2. Approach B: Two-Stage Classification (Impact model + Urgency model -> Rubric mapping)
3. Approach C: MultiOutputClassifier (Joint Impact + Urgency -> Rubric mapping)
4. Class Imbalance Strategies (Unweighted, Balanced Class Weights, In-fold Random Oversampling)

Evaluates on 5-Fold Stratified CV on training split (N=400) and holdout validation split (N=100).
Locked test set remains untouched.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from ticketsense.nlp.preprocessor import TextCleaner

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ticketsense_dataset.csv"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "PRIORITY_EXPERIMENT_RESULTS.json"

print(f"Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
val_df = df[df["split"] == "validation"].copy().reset_index(drop=True)

# Parse Impact and Urgency
for d in [train_df, val_df]:
    d["impact"] = d["priority_labels"].apply(lambda x: x.split("|")[0])
    d["urgency"] = d["priority_labels"].apply(lambda x: x.split("|")[1])

RUBRIC = {
    ("I1", "U1"): "Low", ("I1", "U2"): "Low", ("I1", "U3"): "Medium",
    ("I2", "U1"): "Low", ("I2", "U2"): "Medium", ("I2", "U3"): "High",
    ("I3", "U1"): "Medium", ("I3", "U2"): "High", ("I3", "U3"): "High",
}

def map_rubric(impacts, urgencies):
    return [RUBRIC.get((i, u), "Medium") for i, u in zip(impacts, urgencies)]

# Function for 5-fold CV evaluation
def evaluate_cv(build_pipeline_fn, X, y, n_splits=5, is_two_stage=False, build_two_stage_fn=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_accs, cv_macro_f1s, cv_high_recalls = [], [], []

    for train_idx, test_idx in skf.split(X, y if not is_two_stage else X["priority"]):
        if not is_two_stage:
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            pipe = build_pipeline_fn()
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
        else:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            pipe_imp, pipe_urg = build_two_stage_fn()
            pipe_imp.fit(X_tr["ticket_text"], X_tr["impact"])
            pipe_urg.fit(X_tr["ticket_text"], X_tr["urgency"])
            pred_imp = pipe_imp.predict(X_te["ticket_text"])
            pred_urg = pipe_urg.predict(X_te["ticket_text"])
            preds = map_rubric(pred_imp, pred_urg)
            y_te = X_te["priority"].tolist()

        cv_accs.append(accuracy_score(y_te, preds))
        cv_macro_f1s.append(f1_score(y_te, preds, average="macro", zero_division=0))
        # Per class High recall
        rec_high = recall_score(y_te, preds, labels=["High"], average=None, zero_division=0)[0]
        cv_high_recalls.append(rec_high)

    return {
        "cv_accuracy_mean": float(np.mean(cv_accs)),
        "cv_accuracy_std": float(np.std(cv_accs)),
        "cv_macro_f1_mean": float(np.mean(cv_macro_f1s)),
        "cv_macro_f1_std": float(np.std(cv_macro_f1s)),
        "cv_high_recall_mean": float(np.mean(cv_high_recalls)),
        "cv_high_recall_std": float(np.std(cv_high_recalls)),
    }

# In-fold random oversampler helper
def oversample_df(train_data, target_col="priority"):
    max_count = train_data[target_col].value_counts().max()
    dfs = []
    for val, count in train_data[target_col].value_counts().items():
        sub = train_data[train_data[target_col] == val]
        if count < max_count:
            # Upsample minority to 100 samples or match
            target_n = min(max_count, max(count * 3, 80))
            sub_up = sub.sample(n=target_n, replace=True, random_state=42)
            dfs.append(sub_up)
        else:
            dfs.append(sub)
    return pd.concat(dfs, ignore_index=True)

experiments = {}

print("\n" + "="*70)
print("EXPERIMENT 1: APPROACH A - DIRECT PRIORITY CLASSIFICATION (ALGORITHMS)")
print("="*70)

direct_models = {
    "Direct_LogReg_Balanced": lambda: Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000))
    ]),
    "Direct_LogReg_Unweighted": lambda: Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, class_weight=None, random_state=42, max_iter=1000))
    ]),
    "Direct_LinearSVC_Balanced": lambda: Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42))
    ]),
    "Direct_MultinomialNB": lambda: Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.5))
    ]),
    "Direct_RandomForest_Balanced": lambda: Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
    ]),
}

for name, build_fn in direct_models.items():
    print(f"\n--- Testing {name} ---")
    cv_res = evaluate_cv(build_fn, train_df["ticket_text"], train_df["priority"])
    pipe = build_fn()
    pipe.fit(train_df["ticket_text"], train_df["priority"])
    train_preds = pipe.predict(train_df["ticket_text"])
    val_preds = pipe.predict(val_df["ticket_text"])

    val_acc = accuracy_score(val_df["priority"], val_preds)
    val_macro_f1 = f1_score(val_df["priority"], val_preds, average="macro", zero_division=0)
    train_macro_f1 = f1_score(train_df["priority"], train_preds, average="macro", zero_division=0)
    rep = classification_report(val_df["priority"], val_preds, output_dict=True, zero_division=0)

    print(f"5-Fold CV Macro F1: {cv_res['cv_macro_f1_mean']:.4f} +/- {cv_res['cv_macro_f1_std']:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} | Validation Macro F1: {val_macro_f1:.4f}")
    print(f"High Priority - Precision: {rep.get('High',{}).get('precision',0):.4f}, Recall: {rep.get('High',{}).get('recall',0):.4f}")

    experiments[name] = {
        "architecture_type": "Direct_Priority",
        "cv_metrics": cv_res,
        "train_macro_f1": float(train_macro_f1),
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "val_high_precision": float(rep.get("High",{}).get("precision",0)),
        "val_high_recall": float(rep.get("High",{}).get("recall",0)),
        "val_high_f1": float(rep.get("High",{}).get("f1-score",0)),
        "generalization_gap": float(train_macro_f1 - val_macro_f1),
        "per_class": rep
    }

print("\n" + "="*70)
print("EXPERIMENT 2: APPROACH B - TWO-STAGE CLASSIFICATION (IMPACT + URGENCY -> RUBRIC)")
print("="*70)

two_stage_models = {
    "TwoStage_LogReg_Balanced": lambda: (
        Pipeline([("cleaner", TextCleaner()), ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)), ("clf", LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000))]),
        Pipeline([("cleaner", TextCleaner()), ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)), ("clf", LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000))])
    ),
    "TwoStage_LinearSVC_Balanced": lambda: (
        Pipeline([("cleaner", TextCleaner()), ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)), ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42))]),
        Pipeline([("cleaner", TextCleaner()), ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)), ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42))])
    ),
    "TwoStage_MultinomialNB": lambda: (
        Pipeline([("cleaner", TextCleaner()), ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)), ("clf", MultinomialNB(alpha=0.5))]),
        Pipeline([("cleaner", TextCleaner()), ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)), ("clf", MultinomialNB(alpha=0.5))])
    ),
}

for name, build_fn in two_stage_models.items():
    print(f"\n--- Testing {name} ---")
    cv_res = evaluate_cv(None, train_df, None, is_two_stage=True, build_two_stage_fn=build_fn)
    p_imp, p_urg = build_fn()
    p_imp.fit(train_df["ticket_text"], train_df["impact"])
    p_urg.fit(train_df["ticket_text"], train_df["urgency"])

    train_pred_imp = p_imp.predict(train_df["ticket_text"])
    train_pred_urg = p_urg.predict(train_df["ticket_text"])
    train_preds = map_rubric(train_pred_imp, train_pred_urg)

    val_pred_imp = p_imp.predict(val_df["ticket_text"])
    val_pred_urg = p_urg.predict(val_df["ticket_text"])
    val_preds = map_rubric(val_pred_imp, val_pred_urg)

    val_acc = accuracy_score(val_df["priority"], val_preds)
    val_macro_f1 = f1_score(val_df["priority"], val_preds, average="macro", zero_division=0)
    train_macro_f1 = f1_score(train_df["priority"], train_preds, average="macro", zero_division=0)
    rep = classification_report(val_df["priority"], val_preds, output_dict=True, zero_division=0)

    print(f"5-Fold CV Macro F1: {cv_res['cv_macro_f1_mean']:.4f} +/- {cv_res['cv_macro_f1_std']:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} | Validation Macro F1: {val_macro_f1:.4f}")
    print(f"High Priority - Precision: {rep.get('High',{}).get('precision',0):.4f}, Recall: {rep.get('High',{}).get('recall',0):.4f}")

    experiments[name] = {
        "architecture_type": "Two_Stage_Rubric",
        "cv_metrics": cv_res,
        "train_macro_f1": float(train_macro_f1),
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "val_high_precision": float(rep.get("High",{}).get("precision",0)),
        "val_high_recall": float(rep.get("High",{}).get("recall",0)),
        "val_high_f1": float(rep.get("High",{}).get("f1-score",0)),
        "generalization_gap": float(train_macro_f1 - val_macro_f1),
        "per_class": rep
    }

print("\n" + "="*70)
print("EXPERIMENT 3: IMBALANCE RESAMPLING (IN-FOLD RANDOM OVERSAMPLING)")
print("="*70)

# Evaluate in-fold oversampling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ros_cv_f1s, ros_high_recalls = [], []
for train_idx, test_idx in skf.split(train_df, train_df["priority"]):
    f_train = train_df.iloc[train_idx]
    f_test = train_df.iloc[test_idx]
    f_train_up = oversample_df(f_train, "priority")
    pipe = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, class_weight=None, random_state=42, max_iter=1000))
    ])
    pipe.fit(f_train_up["ticket_text"], f_train_up["priority"])
    preds = pipe.predict(f_test["ticket_text"])
    ros_cv_f1s.append(f1_score(f_test["priority"], preds, average="macro", zero_division=0))
    ros_high_recalls.append(recall_score(f_test["priority"], preds, labels=["High"], average=None, zero_division=0)[0])

# Fit on full train oversampled, evaluate on validation
full_train_up = oversample_df(train_df, "priority")
pipe_ros = Pipeline([
    ("cleaner", TextCleaner()),
    ("tfidf", TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)),
    ("clf", LogisticRegression(C=2.0, class_weight=None, random_state=42, max_iter=1000))
])
pipe_ros.fit(full_train_up["ticket_text"], full_train_up["priority"])
ros_val_preds = pipe_ros.predict(val_df["ticket_text"])
ros_val_acc = accuracy_score(val_df["priority"], ros_val_preds)
ros_val_macro_f1 = f1_score(val_df["priority"], ros_val_preds, average="macro", zero_division=0)
rep_ros = classification_report(val_df["priority"], ros_val_preds, output_dict=True, zero_division=0)

print(f"Random Oversampling 5-Fold CV Macro F1: {np.mean(ros_cv_f1s):.4f} +/- {np.std(ros_cv_f1s):.4f}")
print(f"Validation Accuracy: {ros_val_acc:.4f} | Validation Macro F1: {ros_val_macro_f1:.4f}")
print(f"High Priority - Precision: {rep_ros.get('High',{}).get('precision',0):.4f}, Recall: {rep_ros.get('High',{}).get('recall',0):.4f}")

experiments["Direct_LogReg_RandomOversampling"] = {
    "architecture_type": "InFold_Oversampling",
    "cv_metrics": {
        "cv_macro_f1_mean": float(np.mean(ros_cv_f1s)),
        "cv_macro_f1_std": float(np.std(ros_cv_f1s)),
        "cv_high_recall_mean": float(np.mean(ros_high_recalls)),
        "cv_high_recall_std": float(np.std(ros_high_recalls)),
    },
    "val_accuracy": float(ros_val_acc),
    "val_macro_f1": float(ros_val_macro_f1),
    "val_high_precision": float(rep_ros.get("High",{}).get("precision",0)),
    "val_high_recall": float(rep_ros.get("High",{}).get("recall",0)),
    "val_high_f1": float(rep_ros.get("High",{}).get("f1-score",0)),
    "per_class": rep_ros
}

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    json.dump(experiments, f, indent=2)
print(f"\nAll experimental results saved to: {OUTPUT_REPORT}")
