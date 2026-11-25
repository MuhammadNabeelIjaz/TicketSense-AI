"""Comprehensive benchmark of Priority models comparing:
- Baseline training data (N=400) vs. Enhanced training data (N=450 with v2 expansion)
- Two-Stage regularized LogReg vs. FE-3 Lexical Augmented vs. Direct Priority vs. LinearSVC
- Evaluated strictly on the untouched validation split (N=100).
- Reports Accuracy, Macro F1, High Precision, High Recall, Cost-Weighted Error, False-Lows.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ticketsense.nlp.preprocessor import clean_text
from ticketsense.models.priority_pipeline import AUTHORITATIVE_RUBRIC
from scripts.run_feature_engineering_experiments import KeywordCountExtractor, parse_labels, evaluate_predictions

def run_benchmark():
    base_path = PROJECT_ROOT / "data" / "processed" / "ticketsense_dataset.csv"
    exp2_path = PROJECT_ROOT / "data" / "processed" / "priority_synthetic_expansion_v2.csv"
    
    df = pd.read_csv(base_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    
    train_df[["impact", "urgency"]] = train_df["priority_labels"].apply(lambda x: pd.Series(parse_labels(x)))
    val_df[["impact", "urgency"]] = val_df["priority_labels"].apply(lambda x: pd.Series(parse_labels(x)))
    
    exp2_df = pd.read_csv(exp2_path)
    combined_train_df = pd.concat([train_df, exp2_df], ignore_index=True)
    
    X_val_clean = [clean_text(t) for t in val_df["ticket_text"]]
    y_val_prio = val_df["priority"].tolist()
    
    candidates = []
    
    # Setup training configurations
    configs = [
        ("Base_Train (N=400)", train_df),
        ("Expanded_Train_V2 (N=450)", combined_train_df),
    ]
    
    urg_kw = ["immediately", "urgent", "asap", "emergency", "hurry", "quickly", "critical", "soon", "now"]
    imp_kw = ["crashed", "outage", "down", "unauthorized", "fraud", "hacked", "stolen", "breach", "locked out"]
    kw_ext = KeywordCountExtractor(urg_kw + imp_kw)
    
    for train_name, tr_df in configs:
        X_tr_clean = [clean_text(t) for t in tr_df["ticket_text"]]
        y_tr_imp = tr_df["impact"].tolist()
        y_tr_urg = tr_df["urgency"].tolist()
        y_tr_prio = tr_df["priority"].tolist()
        
        # 1. Standard TF-IDF Two-Stage
        vec_std = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        X_tr_vec = vec_std.fit_transform(X_tr_clean)
        X_va_vec = vec_std.transform(X_val_clean)
        
        m_i = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr_vec, y_tr_imp)
        m_u = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr_vec, y_tr_urg)
        
        p_i = m_i.predict(X_va_vec)
        p_u = m_u.predict(X_va_vec)
        p_prio = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(p_i, p_u)]
        candidates.append(evaluate_predictions(f"{train_name}: Two-Stage LogReg (Standard)", y_val_prio, p_prio))
        
        # 2. Lexical Augmented Two-Stage (FE-3)
        X_tr_kw = kw_ext.transform(X_tr_clean)
        X_va_kw = kw_ext.transform(X_val_clean)
        X_tr_aug = hstack([X_tr_vec, X_tr_kw])
        X_va_aug = hstack([X_va_vec, X_va_kw])
        
        m_i_aug = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr_aug, y_tr_imp)
        m_u_aug = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr_aug, y_tr_urg)
        
        p_i_aug = m_i_aug.predict(X_va_aug)
        p_u_aug = m_u_aug.predict(X_va_aug)
        p_prio_aug = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(p_i_aug, p_u_aug)]
        candidates.append(evaluate_predictions(f"{train_name}: Two-Stage LogReg (Lexical Augmented)", y_val_prio, p_prio_aug))
        
        # 3. Direct Priority LogReg
        m_dir = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr_vec, y_tr_prio)
        p_prio_dir = m_dir.predict(X_va_vec)
        candidates.append(evaluate_predictions(f"{train_name}: Direct LogReg (Balanced)", y_val_prio, p_prio_dir))
        
        # 4. Direct Priority LinearSVC
        m_svc = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000).fit(X_tr_vec, y_tr_prio)
        p_prio_svc = m_svc.predict(X_va_vec)
        candidates.append(evaluate_predictions(f"{train_name}: Direct LinearSVC (Balanced)", y_val_prio, p_prio_svc))

    print("="*115)
    print(f"{'Model & Training Configuration':50} | {'Val Acc':7} | {'Macro F1':8} | {'High Rec':8} | {'High Prec':9} | {'False-Lows':10}")
    print("="*115)
    for c in candidates:
        print(f"{c['candidate']:50} | {c['accuracy']:.4f}  | {c['macro_f1']:.4f}   | {c['high_recall']:.4f}   | {c['high_precision']:.4f}    | {c['total_false_lows']:2d} (H:{c['false_low_high']})")

    out_json = PROJECT_ROOT / "reports" / "PRIORITY_EXPANSION_V2_BENCHMARK.json"
    with open(out_json, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"\nSaved benchmark metrics to: {out_json}")

if __name__ == "__main__":
    run_benchmark()
