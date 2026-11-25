"""Comprehensive ML experiment suite for High-Priority Recall and Cost-Sensitive Optimization.

Investigates:
1. Baseline Two-Stage Regularized Logistic Regression (argmax)
2. Cost-Sensitive Class-Weighted Two-Stage Logistic Regression (upweighting I3 and U3)
3. Component Decision-Threshold Tuning (lowering I3/U3 thresholds on component probabilities)
4. Joint Expected Cost Minimization (decision theory using explicit cost matrix)
5. Direct Priority Classifiers (Logistic Regression, LinearSVC, ComplementNB with class weighting)
6. Supplementary Data Training (incorporating priority_supplementary_training.csv)

Strict Protocol:
- Evaluates strictly on validation split (N=100) of ticketsense_dataset.csv.
- Never touches the locked test set.
- Calculates Cost-Weighted Error, High Recall, High Precision, Macro F1, False-Lows.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ticketsense.nlp.preprocessor import clean_text
from ticketsense.models.priority_pipeline import AUTHORITATIVE_RUBRIC

# Cost Matrix: Rows = Actual, Columns = Predicted (Low, Medium, High)
# High->Low cost = 10, High->Med cost = 5, Med->Low cost = 4, Low->High cost = 2
COST_MATRIX = {
    ("High", "High"): 0, ("High", "Medium"): 5, ("High", "Low"): 10,
    ("Medium", "High"): 1, ("Medium", "Medium"): 0, ("Medium", "Low"): 4,
    ("Low", "High"): 2, ("Low", "Medium"): 1, ("Low", "Low"): 0,
}

def compute_cost(actuals, preds):
    total = sum(COST_MATRIX.get((a, p), 0) for a, p in zip(actuals, preds))
    return total / len(actuals)

def parse_labels(labels_str):
    if pd.isna(labels_str) or not str(labels_str).strip():
        return "I2", "U1"
    clean = str(labels_str).strip("()").replace("'", "").replace('"', "")
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    if "|" in str(labels_str):
        parts = [p.strip() for p in str(labels_str).split("|") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
    return "I2", "U1"

def evaluate_predictions(name, actual_priorities, pred_priorities):
    acc = accuracy_score(actual_priorities, pred_priorities)
    macro_f1 = f1_score(actual_priorities, pred_priorities, average="macro", zero_division=0)
    weighted_f1 = f1_score(actual_priorities, pred_priorities, average="weighted", zero_division=0)
    
    classes = ["High", "Medium", "Low"]
    precisions = precision_score(actual_priorities, pred_priorities, average=None, labels=classes, zero_division=0)
    recalls = recall_score(actual_priorities, pred_priorities, average=None, labels=classes, zero_division=0)
    f1s = f1_score(actual_priorities, pred_priorities, average=None, labels=classes, zero_division=0)
    
    cost = compute_cost(actual_priorities, pred_priorities)
    
    # False-lows: actual High or Medium predicted as Low
    fl_high = sum(1 for a, p in zip(actual_priorities, pred_priorities) if a == "High" and p == "Low")
    fl_med = sum(1 for a, p in zip(actual_priorities, pred_priorities) if a == "Medium" and p == "Low")
    total_fl = fl_high + fl_med
    
    cm = confusion_matrix(actual_priorities, pred_priorities, labels=classes)
    
    return {
        "candidate": name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "high_precision": float(precisions[0]),
        "high_recall": float(recalls[0]),
        "high_f1": float(f1s[0]),
        "medium_precision": float(precisions[1]),
        "medium_recall": float(recalls[1]),
        "medium_f1": float(f1s[1]),
        "low_precision": float(precisions[2]),
        "low_recall": float(recalls[2]),
        "low_f1": float(f1s[2]),
        "cost_weighted_error": float(cost),
        "false_low_high": int(fl_high),
        "false_low_medium": int(fl_med),
        "total_false_lows": int(total_fl),
        "confusion_matrix": cm.tolist(),
    }

def run_experiments():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "ticketsense_dataset.csv"
    df = pd.read_csv(dataset_path)
    
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    
    # Parse impact and urgency
    train_df[["impact", "urgency"]] = train_df["priority_labels"].apply(lambda x: pd.Series(parse_labels(x)))
    val_df[["impact", "urgency"]] = val_df["priority_labels"].apply(lambda x: pd.Series(parse_labels(x)))
    
    # Preprocess text
    X_train_clean = [clean_text(t) for t in train_df["ticket_text"]]
    X_val_clean = [clean_text(t) for t in val_df["ticket_text"]]
    
    y_train_prio = train_df["priority"].tolist()
    y_val_prio = val_df["priority"].tolist()
    
    y_train_imp = train_df["impact"].tolist()
    y_val_imp = val_df["impact"].tolist()
    
    y_train_urg = train_df["urgency"].tolist()
    y_val_urg = val_df["urgency"].tolist()
    
    # Base Vectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X_train_vec = vec.fit_transform(X_train_clean)
    X_val_vec = vec.transform(X_val_clean)
    
    results = []
    
    # --------------------------------------------------------------------------
    # Candidate 1: Current Baseline Two-Stage (Argmax, balanced)
    # --------------------------------------------------------------------------
    clf_imp_base = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000)
    clf_imp_base.fit(X_train_vec, y_train_imp)
    clf_urg_base = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000)
    clf_urg_base.fit(X_train_vec, y_train_urg)
    
    pred_imp_base = clf_imp_base.predict(X_val_vec)
    pred_urg_base = clf_urg_base.predict(X_val_vec)
    pred_prio_base = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(pred_imp_base, pred_urg_base)]
    results.append(evaluate_predictions("1. Baseline Two-Stage (Argmax, C_imp=2.0, C_urg=1.0)", y_val_prio, pred_prio_base))
    
    # --------------------------------------------------------------------------
    # Candidate 2: Cost-Sensitive Class Weights (Upweighting I3 and U3)
    # --------------------------------------------------------------------------
    imp_weights = {"I1": 1.0, "I2": 2.0, "I3": 5.0}
    urg_weights = {"U1": 1.0, "U2": 2.0, "U3": 5.0}
    clf_imp_cs = LogisticRegression(C=2.0, class_weight=imp_weights, random_state=42, max_iter=1000)
    clf_imp_cs.fit(X_train_vec, y_train_imp)
    clf_urg_cs = LogisticRegression(C=1.0, class_weight=urg_weights, random_state=42, max_iter=1000)
    clf_urg_cs.fit(X_train_vec, y_train_urg)
    
    pred_imp_cs = clf_imp_cs.predict(X_val_vec)
    pred_urg_cs = clf_urg_cs.predict(X_val_vec)
    pred_prio_cs = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(pred_imp_cs, pred_urg_cs)]
    results.append(evaluate_predictions("2. Cost-Sensitive Class Weights (I3x5, U3x5)", y_val_prio, pred_prio_cs))
    
    # --------------------------------------------------------------------------
    # Candidate 3: Component Decision-Threshold Moving on Baseline Two-Stage
    # If P(I3) >= theta_imp -> predict I3; If P(U3) >= theta_urg -> predict U3
    # --------------------------------------------------------------------------
    probas_imp = clf_imp_base.predict_proba(X_val_vec)
    probas_urg = clf_urg_base.predict_proba(X_val_vec)
    classes_imp = list(clf_imp_base.classes_)
    classes_urg = list(clf_urg_base.classes_)
    idx_i3 = classes_imp.index("I3")
    idx_u3 = classes_urg.index("U3")
    
    for theta in [0.25, 0.30, 0.35]:
        pred_prio_thresh = []
        for p_i, p_u in zip(probas_imp, probas_urg):
            # Impact decision
            imp = "I3" if p_i[idx_i3] >= theta else classes_imp[np.argmax(p_i)]
            # Urgency decision
            urg = "U3" if p_u[idx_u3] >= theta else classes_urg[np.argmax(p_u)]
            pred_prio_thresh.append(AUTHORITATIVE_RUBRIC.get((imp, urg), "Low"))
        results.append(evaluate_predictions(f"3. Threshold-Tuned Two-Stage (Theta_I3=U3={theta})", y_val_prio, pred_prio_thresh))

    # --------------------------------------------------------------------------
    # Candidate 4: Expected Cost Minimization via Joint Probabilities
    # Evaluate expected cost of each priority class and select argmin
    # --------------------------------------------------------------------------
    pred_prio_bayes = []
    # Joint probabilities: P(prio) = sum_{(i,u) -> prio} P(i) * P(u)
    for p_i, p_u in zip(probas_imp, probas_urg):
        joint = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
        for i_idx, i_cls in enumerate(classes_imp):
            for u_idx, u_cls in enumerate(classes_urg):
                p_pair = p_i[i_idx] * p_u[u_idx]
                mapped = AUTHORITATIVE_RUBRIC.get((i_cls, u_cls), "Low")
                joint[mapped] += p_pair
        
        # Expected costs for choosing action a in {Low, Medium, High}:
        # Expected Cost(a) = sum_{y} P(y) * COST(y, a)
        cost_low = joint["High"] * 10 + joint["Medium"] * 4 + joint["Low"] * 0
        cost_med = joint["High"] * 5  + joint["Medium"] * 0 + joint["Low"] * 1
        cost_high = joint["High"] * 0 + joint["Medium"] * 1 + joint["Low"] * 2
        
        best_prio = min([("Low", cost_low), ("Medium", cost_med), ("High", cost_high)], key=lambda x: x[1])[0]
        pred_prio_bayes.append(best_prio)
    results.append(evaluate_predictions("4. Joint Expected Cost Minimization (Bayesian Decision)", y_val_prio, pred_prio_bayes))

    # --------------------------------------------------------------------------
    # Candidate 5: Direct Priority Classifier (Balanced LogReg & LinearSVC)
    # --------------------------------------------------------------------------
    clf_direct_lr = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000)
    clf_direct_lr.fit(X_train_vec, y_train_prio)
    pred_prio_direct_lr = clf_direct_lr.predict(X_val_vec)
    results.append(evaluate_predictions("5a. Direct Priority Logistic Regression (Balanced)", y_val_prio, pred_prio_direct_lr))

    clf_direct_svc = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000)
    clf_direct_svc.fit(X_train_vec, y_train_prio)
    pred_prio_direct_svc = clf_direct_svc.predict(X_val_vec)
    results.append(evaluate_predictions("5b. Direct Priority LinearSVC (Balanced)", y_val_prio, pred_prio_direct_svc))

    # --------------------------------------------------------------------------
    # Candidate 6: Training with Supplementary Data
    # --------------------------------------------------------------------------
    supp_path = PROJECT_ROOT / "data" / "processed" / "priority_supplementary_training.csv"
    if supp_path.exists():
        supp_df = pd.read_csv(supp_path)
        supp_df[["impact", "urgency"]] = supp_df["priority_labels"].apply(lambda x: pd.Series(parse_labels(x)))
        combined_train = pd.concat([train_df, supp_df], ignore_index=True)
        
        X_comb_clean = [clean_text(t) for t in combined_train["ticket_text"]]
        vec_comb = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        X_comb_vec = vec_comb.fit_transform(X_comb_clean)
        X_val_comb_vec = vec_comb.transform(X_val_clean)
        
        clf_imp_supp = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000)
        clf_imp_supp.fit(X_comb_vec, combined_train["impact"])
        clf_urg_supp = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000)
        clf_urg_supp.fit(X_comb_vec, combined_train["urgency"])
        
        pred_imp_supp = clf_imp_supp.predict(X_val_comb_vec)
        pred_urg_supp = clf_urg_supp.predict(X_val_comb_vec)
        pred_prio_supp = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(pred_imp_supp, pred_urg_supp)]
        results.append(evaluate_predictions("6. Two-Stage + Supplementary Incidents (N=470)", y_val_prio, pred_prio_supp))

    # Output comparative results table
    print("="*115)
    print(f"{'Experiment Candidate':48} | {'Val Acc':7} | {'Macro F1':8} | {'High Rec':8} | {'High Prec':9} | {'Cost Err':8} | {'False-Lows':10}")
    print("="*115)
    for r in results:
        print(f"{r['candidate']:48} | {r['accuracy']:.4f}  | {r['macro_f1']:.4f}   | {r['high_recall']:.4f}   | {r['high_precision']:.4f}    | {r['cost_weighted_error']:.4f}   | {r['total_false_lows']:2d} (H:{r['false_low_high']})")

    # Save to JSON
    out_path = PROJECT_ROOT / "reports" / "HIGH_RECALL_EXPERIMENT_RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed experiment metrics to: {out_path}")

if __name__ == "__main__":
    run_experiments()
