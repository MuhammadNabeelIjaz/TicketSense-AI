"""Side-by-side comparison of Baseline Two-Stage vs. FE-3 Lexical Augmented Two-Stage.

Evaluates:
- Standalone metrics (Accuracy, Macro F1, High Precision, High Recall, False-Lows)
- Uncertainty gate performance (Autonomous coverage, Autonomous accuracy, Effective High Recall, Review rate)
- Cost-weighted error
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ticketsense.nlp.preprocessor import clean_text
from ticketsense.models.priority_pipeline import AUTHORITATIVE_RUBRIC
from ticketsense.inference.uncertainty import compute_priority_uncertainty_score
from scripts.run_feature_engineering_experiments import KeywordCountExtractor, parse_labels, evaluate_predictions

def compare_models():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "ticketsense_dataset.csv"
    df = pd.read_csv(dataset_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    
    train_df[["impact", "urgency"]] = train_df["priority_labels"].apply(lambda x: pd.Series(parse_labels(x)))
    val_df[["impact", "urgency"]] = val_df["priority_labels"].apply(lambda x: pd.Series(parse_labels(x)))
    
    X_train_raw = train_df["ticket_text"].tolist()
    X_val_raw = val_df["ticket_text"].tolist()
    
    X_train_clean = [clean_text(t) for t in X_train_raw]
    X_val_clean = [clean_text(t) for t in X_val_raw]
    
    y_train_imp, y_val_imp = train_df["impact"].tolist(), val_df["impact"].tolist()
    y_train_urg, y_val_urg = train_df["urgency"].tolist(), val_df["urgency"].tolist()
    y_val_prio = val_df["priority"].tolist()
    
    # Model 1: Baseline
    vec1 = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X_tr1 = vec1.fit_transform(X_train_clean)
    X_va1 = vec1.transform(X_val_clean)
    m_i1 = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr1, y_train_imp)
    m_u1 = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr1, y_train_urg)
    
    # Model 2: FE-3 Lexical Augmented
    urg_kw = ["immediately", "urgent", "asap", "emergency", "hurry", "quickly", "critical", "soon", "now"]
    imp_kw = ["crashed", "outage", "down", "unauthorized", "fraud", "hacked", "stolen", "breach", "locked out"]
    kw_ext = KeywordCountExtractor(urg_kw + imp_kw)
    X_tr_kw = kw_ext.transform(X_train_clean)
    X_va_kw = kw_ext.transform(X_val_clean)
    X_tr2 = hstack([X_tr1, X_tr_kw])
    X_va2 = hstack([X_va1, X_va_kw])
    m_i2 = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr2, y_train_imp)
    m_u2 = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr2, y_train_urg)
    
    for name, (m_i, m_u, X_va) in [("Baseline Two-Stage", (m_i1, m_u1, X_va1)), ("FE-3 Lexical Two-Stage", (m_i2, m_u2, X_va2))]:
        pred_i = m_i.predict(X_va)
        pred_u = m_u.predict(X_va)
        prob_i = m_i.predict_proba(X_va)
        prob_u = m_u.predict_proba(X_va)
        classes_i = list(m_i.classes_)
        classes_u = list(m_u.classes_)
        
        pred_prio = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(pred_i, pred_u)]
        eval_base = evaluate_predictions(name, y_val_prio, pred_prio)
        
        # Uncertainty Gate evaluation
        gate_flags = []
        for text, pi, pu, prio_hat, imp_hat, urg_hat in zip(X_val_raw, prob_i, prob_u, pred_prio, pred_i, pred_u):
            # Compute joint probabilities
            joint = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
            for i_idx, i_c in enumerate(classes_i):
                for u_idx, u_c in enumerate(classes_u):
                    joint[AUTHORITATIVE_RUBRIC.get((i_c, u_c), "Low")] += pi[i_idx] * pu[u_idx]
            
            imp_conf = float(np.max(pi))
            urg_conf = float(np.max(pu))
            
            diag = compute_priority_uncertainty_score(
                priority=prio_hat,
                priority_probabilities=joint,
                predicted_impact=imp_hat,
                impact_confidence=imp_conf,
                predicted_urgency=urg_hat,
                urgency_confidence=urg_conf,
                ticket_text=text,
                threshold=0.60,
                margin=0.15,
                component_threshold=0.50,
                high_risk_threshold=0.15,
            )
            gate_flags.append(diag["priority_review_required"])
            
        gate_flags = np.array(gate_flags)
        auto_mask = ~gate_flags
        review_mask = gate_flags
        
        auto_acc = np.mean([p == a for p, a in zip(np.array(pred_prio)[auto_mask], np.array(y_val_prio)[auto_mask])]) if sum(auto_mask) > 0 else 0
        
        # Effective High recall under gate (routed to review or predicted high)
        high_mask = (np.array(y_val_prio) == "High")
        high_caught = sum(1 for is_h, p, rev in zip(high_mask, pred_prio, gate_flags) if is_h and (p == "High" or rev))
        effective_high_rec = high_caught / sum(high_mask)
        
        print(f"\n========================================================")
        print(f"MODEL: {name}")
        print(f"========================================================")
        print(f"Standalone Accuracy:      {eval_base['accuracy']:.4f}")
        print(f"Standalone Macro F1:      {eval_base['macro_f1']:.4f}")
        print(f"Standalone High Recall:   {eval_base['high_recall']:.4f}")
        print(f"Standalone High Precision: {eval_base['high_precision']:.4f}")
        print(f"Total False-Lows:         {eval_base['total_false_lows']} (High: {eval_base['false_low_high']})")
        print(f"Autonomous Coverage:      {np.mean(auto_mask):.1%} ({sum(auto_mask)}/100)")
        print(f"Autonomous Accuracy:      {auto_acc:.4f} ({sum([p == a for p, a in zip(np.array(pred_prio)[auto_mask], np.array(y_val_prio)[auto_mask])])}/{sum(auto_mask)})")
        print(f"Human Review Rate:        {np.mean(review_mask):.1%} ({sum(review_mask)}/100)")
        print(f"Effective High Recall:    {effective_high_rec:.4f} ({high_caught}/{sum(high_mask)})")

if __name__ == "__main__":
    compare_models()
