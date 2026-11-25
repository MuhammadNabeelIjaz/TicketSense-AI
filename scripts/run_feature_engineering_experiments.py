"""Feature Engineering and Preprocessing Optimization for Priority Model.

Investigates:
1. Word + Char N-Grams (word ngram 1-2 + char ngram 3-5)
2. Sublinear TF vs Raw TF
3. Custom Lexical Feature Augmentation (explicit counts of Urgency and Impact keywords)
4. Vocabulary Filtering (min_df, max_df)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ticketsense.nlp.preprocessor import clean_text
from ticketsense.models.priority_pipeline import AUTHORITATIVE_RUBRIC

class KeywordCountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, keywords):
        self.keywords = [k.lower() for k in keywords]
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features = []
        for text in X:
            t = str(text).lower()
            counts = [t.count(k) for k in self.keywords]
            total = sum(counts)
            features.append(counts + [total])
        return np.array(features)

def parse_labels(labels_str):
    if pd.isna(labels_str) or not str(labels_str).strip():
        return "I2", "U1"
    val = str(labels_str).strip()
    if "|" in val:
        parts = [p.strip() for p in val.split("|")]
        if len(parts) >= 2:
            return parts[0], parts[1]
    clean = val.strip("()").replace("'", "").replace('"', "")
    if "," in clean:
        parts = [p.strip() for p in clean.split(",") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
    return "I2", "U1"

def evaluate_predictions(name, actual_priorities, pred_priorities):
    acc = accuracy_score(actual_priorities, pred_priorities)
    macro_f1 = f1_score(actual_priorities, pred_priorities, average="macro", zero_division=0)
    classes = ["High", "Medium", "Low"]
    precisions = precision_score(actual_priorities, pred_priorities, average=None, labels=classes, zero_division=0)
    recalls = recall_score(actual_priorities, pred_priorities, average=None, labels=classes, zero_division=0)
    
    fl_high = sum(1 for a, p in zip(actual_priorities, pred_priorities) if a == "High" and p == "Low")
    fl_med = sum(1 for a, p in zip(actual_priorities, pred_priorities) if a == "Medium" and p == "Low")
    
    return {
        "candidate": name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "high_precision": float(precisions[0]),
        "high_recall": float(recalls[0]),
        "false_low_high": int(fl_high),
        "false_low_med": int(fl_med),
        "total_false_lows": int(fl_high + fl_med)
    }

def run_fe_experiments():
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
    
    results = []
    
    # 1. Baseline: Word (1,2) sublinear
    vec1 = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X_tr1 = vec1.fit_transform(X_train_clean)
    X_va1 = vec1.transform(X_val_clean)
    m_i1 = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr1, y_train_imp)
    m_u1 = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr1, y_train_urg)
    p_i1, p_u1 = m_i1.predict(X_va1), m_u1.predict(X_va1)
    p_p1 = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(p_i1, p_u1)]
    results.append(evaluate_predictions("FE-1: Baseline Word (1,2) Sublinear TF", y_val_prio, p_p1))
    
    # 2. Word (1,3)
    vec2 = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
    X_tr2 = vec2.fit_transform(X_train_clean)
    X_va2 = vec2.transform(X_val_clean)
    m_i2 = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr2, y_train_imp)
    m_u2 = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr2, y_train_urg)
    p_i2, p_u2 = m_i2.predict(X_va2), m_u2.predict(X_va2)
    p_p2 = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(p_i2, p_u2)]
    results.append(evaluate_predictions("FE-2: Word (1,3) Sublinear TF", y_val_prio, p_p2))
    
    # 3. Word (1,2) + Lexical Keyword Features
    urg_kw = ["immediately", "urgent", "asap", "emergency", "hurry", "quickly", "critical", "soon", "now"]
    imp_kw = ["crashed", "outage", "down", "unauthorized", "fraud", "hacked", "stolen", "breach", "locked out"]
    from scipy.sparse import hstack
    kw_ext = KeywordCountExtractor(urg_kw + imp_kw)
    X_tr_kw = kw_ext.transform(X_train_clean)
    X_va_kw = kw_ext.transform(X_val_clean)
    X_tr3 = hstack([X_tr1, X_tr_kw])
    X_va3 = hstack([X_va1, X_va_kw])
    m_i3 = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr3, y_train_imp)
    m_u3 = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr3, y_train_urg)
    p_i3, p_u3 = m_i3.predict(X_va3), m_u3.predict(X_va3)
    p_p3 = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(p_i3, p_u3)]
    results.append(evaluate_predictions("FE-3: Word (1,2) + Explicit Lexical Indicators", y_val_prio, p_p3))
    
    # 4. Word (1,2) + Char (3,5)
    union = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char", sublinear_tf=True))
    ])
    X_tr4 = union.fit_transform(X_train_clean)
    X_va4 = union.transform(X_val_clean)
    m_i4 = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr4, y_train_imp)
    m_u4 = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr4, y_train_urg)
    p_i4, p_u4 = m_i4.predict(X_va4), m_u4.predict(X_va4)
    p_p4 = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(p_i4, p_u4)]
    results.append(evaluate_predictions("FE-4: Word (1,2) + Char (3,5) Feature Union", y_val_prio, p_p4))

    print("="*105)
    print(f"{'Feature Engineering Experiment':45} | {'Val Acc':7} | {'Macro F1':8} | {'High Rec':8} | {'High Prec':9} | {'False-Lows':10}")
    print("="*105)
    for r in results:
        print(f"{r['candidate']:45} | {r['accuracy']:.4f}  | {r['macro_f1']:.4f}   | {r['high_recall']:.4f}   | {r['high_precision']:.4f}    | {r['total_false_lows']:2d} (H:{r['false_low_high']})")

if __name__ == "__main__":
    run_fe_experiments()
