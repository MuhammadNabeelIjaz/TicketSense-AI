"""Train and serialize FE-3 Lexical Augmented Priority Model candidate.

Preserves the original models/priority_model.joblib untouched.
Saves candidate to: models/priority_model_fe3_candidate.joblib
"""

import sys
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ticketsense.nlp.preprocessor import clean_text
from ticketsense.models.priority_pipeline import AUTHORITATIVE_RUBRIC
from scripts.run_feature_engineering_experiments import KeywordCountExtractor, parse_labels, evaluate_predictions

def train_fe3():
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
    
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X_tr_tfidf = vec.fit_transform(X_train_clean)
    X_va_tfidf = vec.transform(X_val_clean)
    
    urg_kw = ["immediately", "urgent", "asap", "emergency", "hurry", "quickly", "critical", "soon", "now"]
    imp_kw = ["crashed", "outage", "down", "unauthorized", "fraud", "hacked", "stolen", "breach", "locked out"]
    kw_ext = KeywordCountExtractor(urg_kw + imp_kw)
    X_tr_kw = kw_ext.transform(X_train_clean)
    X_va_kw = kw_ext.transform(X_val_clean)
    
    X_tr = hstack([X_tr_tfidf, X_tr_kw])
    X_va = hstack([X_va_tfidf, X_va_kw])
    
    m_i = LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr, y_train_imp)
    m_u = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000).fit(X_tr, y_train_urg)
    
    pred_i = m_i.predict(X_va)
    pred_u = m_u.predict(X_va)
    pred_prio = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(pred_i, pred_u)]
    
    eval_metrics = evaluate_predictions("FE-3 Lexical Two-Stage", y_val_prio, pred_prio)
    
    bundle = {
        "model_type": "TwoStage_Lexical_Augmented",
        "vectorizer": vec,
        "keyword_extractor": kw_ext,
        "impact_model": m_i,
        "urgency_model": m_u,
        "rubric": AUTHORITATIVE_RUBRIC,
        "validation_metrics": eval_metrics
    }
    
    out_model = PROJECT_ROOT / "models" / "priority_model_fe3_candidate.joblib"
    out_meta = PROJECT_ROOT / "models" / "priority_model_fe3_candidate_metadata.json"
    
    joblib.dump(bundle, out_model)
    with open(out_meta, "w") as f:
        json.dump(eval_metrics, f, indent=2)
        
    print(f"Candidate model saved to: {out_model}")
    print(f"Candidate metadata saved to: {out_meta}")
    print(f"Validation Metrics: Acc={eval_metrics['accuracy']:.4f}, Macro F1={eval_metrics['macro_f1']:.4f}, High Recall={eval_metrics['high_recall']:.4f}, High Prec={eval_metrics['high_precision']:.4f}")

if __name__ == "__main__":
    train_fe3()
