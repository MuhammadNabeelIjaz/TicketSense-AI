import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import joblib

from ticketsense.nlp.preprocessor import clean_text
from ticketsense.models.priority_pipeline import AUTHORITATIVE_RUBRIC
from scripts.run_feature_engineering_experiments import KeywordCountExtractor, parse_labels

dataset_path = Path("data/processed/ticketsense_dataset.csv")
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

pi_all = m_i.predict_proba(X_va)
pu_all = m_u.predict_proba(X_va)
pred_i = m_i.predict(X_va)
pred_u = m_u.predict(X_va)
pred_prio = [AUTHORITATIVE_RUBRIC.get((i, u), "Low") for i, u in zip(pred_i, pred_u)]

classes_i = list(m_i.classes_)
classes_u = list(m_u.classes_)

records = []
for text, pi, pu, p_hat, i_hat, u_hat, true_p in zip(X_val_raw, pi_all, pu_all, pred_prio, pred_i, pred_u, y_val_prio):
    joint = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    for i_idx, i_c in enumerate(classes_i):
        for u_idx, u_c in enumerate(classes_u):
            joint[AUTHORITATIVE_RUBRIC.get((i_c, u_c), "Low")] += pi[i_idx] * pu[u_idx]
    
    sorted_p = sorted(joint.values(), reverse=True)
    p_conf = sorted_p[0]
    p_margin = sorted_p[0] - sorted_p[1]
    
    records.append({
        "text": text,
        "true_priority": true_p,
        "pred_priority": p_hat,
        "is_correct": (true_p == p_hat),
        "p_conf": p_conf,
        "p_margin": p_margin,
        "p_high": joint.get("High", 0.0),
        "pred_impact": i_hat,
        "i_conf": float(np.max(pi)),
        "pred_urgency": u_hat,
        "u_conf": float(np.max(pu)),
    })

eval_df = pd.DataFrame(records)

EMERGENCY_WORDS = re.compile(r"\b(emergency|outage|crashed?|critical|down|escalat\w+|immediate\w*)\b", re.IGNORECASE)

print(f"{'Conf':<6} | {'Margin':<6} | {'Comp':<6} | {'Risk':<6} | {'KW':<5} | {'Review%':<8} | {'Auto%':<7} | {'Auto Acc':<9} | {'High Recall':<11} | {'False Low Flagged'}")
print("-" * 105)

grid = []
for t_conf in [0.50, 0.55, 0.58, 0.60]:
    for t_margin in [0.10, 0.15]:
        for t_comp in [0.0, 0.45, 0.50]:
            for risk_high in [1.0, 0.20, 0.15]:
                for kw in [False, True]:
                    def flag_fn(r):
                        if r["p_conf"] < t_conf:
                            return True
                        if r["p_margin"] < t_margin:
                            return True
                        if t_comp > 0 and ((r["i_conf"] < t_comp) or (r["u_conf"] < t_comp)):
                            return True
                        if r["pred_priority"] == "Low" and r["p_high"] >= risk_high:
                            return True
                        if kw and r["pred_priority"] == "Low" and bool(EMERGENCY_WORDS.search(r["text"])):
                            return True
                        return False

                    rev_flag = eval_df.apply(flag_fn, axis=1)
                    rev_rate = rev_flag.mean() * 100
                    auto_df = eval_df[~rev_flag]
                    auto_acc = auto_df["is_correct"].mean() * 100 if len(auto_df) > 0 else 0.0
                    
                    # High recall: true High is either autonomously High or flagged for review
                    true_high = eval_df[eval_df["true_priority"] == "High"]
                    high_caught = ((true_high["pred_priority"] == "High") & (~rev_flag.loc[true_high.index])) | rev_flag.loc[true_high.index]
                    high_recall = high_caught.mean() * 100
                    
                    false_low = eval_df[(eval_df["pred_priority"] == "Low") & (eval_df["true_priority"].isin(["High", "Medium"]))]
                    fl_flagged = false_low.apply(flag_fn, axis=1).sum() if len(false_low) > 0 else 0
                    
                    grid.append({
                        "t_conf": t_conf,
                        "t_margin": t_margin,
                        "t_comp": t_comp,
                        "risk_high": risk_high,
                        "kw": kw,
                        "rev_rate": rev_rate,
                        "auto_cov": 100 - rev_rate,
                        "auto_acc": auto_acc,
                        "high_recall": high_recall,
                        "fl_flagged": fl_flagged,
                        "fl_total": len(false_low),
                    })

res_df = pd.DataFrame(grid)
print("Top configurations sorted by review rate and auto accuracy:")
promising = res_df.sort_values(by=["rev_rate", "auto_acc"], ascending=[True, False])

for idx, r in promising.head(15).iterrows():
    print(f"{r['t_conf']:<6.2f} | {r['t_margin']:<6.2f} | {r['t_comp']:<6.2f} | {r['risk_high']:<6.2f} | {str(r['kw']):<5} | {r['rev_rate']:>6.1f}% | {r['auto_cov']:>5.1f}% | {r['auto_acc']:>7.2f}% | {r['high_recall']:>9.1f}% | {int(r['fl_flagged']):>2}/{int(r['fl_total']):<2}")
