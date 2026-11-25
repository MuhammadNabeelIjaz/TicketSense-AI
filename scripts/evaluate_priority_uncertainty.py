import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import re
from ticketsense.data.loader import load_dataset_splits
from ticketsense.inference.priority_predictor import PriorityPredictor

predictor = PriorityPredictor()
_, val_df = load_dataset_splits(include_synthetic=False)

# Evaluate on all 100 validation samples
records = []
for idx, row in val_df.iterrows():
    text = row["ticket_text"]
    true_prio = row["priority"]
    pred = predictor.predict(text)
    
    p_probs = pred["priority_probabilities"]
    sorted_p = sorted(p_probs.values(), reverse=True)
    p_conf = sorted_p[0]
    p_margin = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else 1.0
    
    i_probs = pred["impact_probabilities"]
    sorted_i = sorted(i_probs.values(), reverse=True)
    i_conf = sorted_i[0]
    i_margin = sorted_i[0] - sorted_i[1] if len(sorted_i) > 1 else 1.0
    
    u_probs = pred["urgency_probabilities"]
    sorted_u = sorted(u_probs.values(), reverse=True)
    u_conf = sorted_u[0]
    u_margin = sorted_u[0] - sorted_u[1] if len(sorted_u) > 1 else 1.0
    
    records.append({
        "id": row["ticket_id"],
        "text": text,
        "true_priority": true_prio,
        "pred_priority": pred["priority"],
        "is_correct": (true_prio == pred["priority"]),
        "p_conf": p_conf,
        "p_margin": p_margin,
        "p_high": p_probs.get("High", 0.0),
        "pred_impact": pred["predicted_impact"],
        "i_conf": i_conf,
        "i_margin": i_margin,
        "pred_urgency": pred["predicted_urgency"],
        "u_conf": u_conf,
        "u_margin": u_margin,
    })

df = pd.DataFrame(records)

# Define criteria:
# 1. Low Confidence: p_conf < T_conf
# 2. Small Margin: p_margin < T_margin
# 3. Component Uncertainty: i_conf < T_comp or u_conf < T_comp
# 4. Ambiguity / Risk: pred == 'Low' but (p_high >= T_risk or contains emergency/outage keywords)

EMERGENCY_WORDS = re.compile(r"\b(emergency|outage|crashed?|critical|down|escalat\w+|immediate\w*)\b", re.IGNORECASE)

configs = [
    # Baseline
    {"name": "C0: Baseline (No Review)", "t_conf": 0.0, "t_margin": 0.0, "t_comp": 0.0, "risk_p_high": 1.0, "check_kw": False},
    # Simple thresholds
    {"name": "C1: Prio Conf < 0.65", "t_conf": 0.65, "t_margin": 0.0, "t_comp": 0.0, "risk_p_high": 1.0, "check_kw": False},
    {"name": "C2: Prio Conf < 0.65 or Margin < 0.15", "t_conf": 0.65, "t_margin": 0.15, "t_comp": 0.0, "risk_p_high": 1.0, "check_kw": False},
    {"name": "C3: Prio Conf < 0.65 or Margin < 0.15 or Comp < 0.55", "t_conf": 0.65, "t_margin": 0.15, "t_comp": 0.55, "risk_p_high": 1.0, "check_kw": False},
    {"name": "C4: C3 + High Risk (Pred Low & P(High) >= 0.12)", "t_conf": 0.65, "t_margin": 0.15, "t_comp": 0.55, "risk_p_high": 0.12, "check_kw": False},
    {"name": "C5: C3 + Suspicious Keyword Alert", "t_conf": 0.65, "t_margin": 0.15, "t_comp": 0.55, "risk_p_high": 0.20, "check_kw": True},
    {"name": "C6: Balanced Safe Routing (Conf < 0.65, Margin < 0.15, Comp < 0.55, Pred Low & High>=0.15 or Kw)", "t_conf": 0.65, "t_margin": 0.15, "t_comp": 0.55, "risk_p_high": 0.15, "check_kw": True},
    {"name": "C7: Calibrated Component Gate (Conf < 0.65, Margin < 0.15, Comp < 0.50, Pred Low & High>=0.15 or Kw)", "t_conf": 0.65, "t_margin": 0.15, "t_comp": 0.50, "risk_p_high": 0.15, "check_kw": True},
]

print("=" * 115)
print(f"{'Config Name':<42} | {'Review%':<8} | {'Auto%':<7} | {'Auto Acc':<9} | {'High Recall':<11} | {'False Low Flagged':<18} | {'Outage Flagged':<14}")
print("=" * 115)

outage_text = "Emergency: our primary database cluster crashed and customer checkout is completely down nationwide. Immediate escalation needed."
outage_pred = predictor.predict(outage_text)
sorted_p = sorted(outage_pred["priority_probabilities"].values(), reverse=True)
outage_data = {
    "text": outage_text,
    "p_conf": sorted_p[0],
    "p_margin": sorted_p[0] - sorted_p[1],
    "p_high": outage_pred["priority_probabilities"].get("High", 0.0),
    "pred_priority": outage_pred["priority"],
    "i_conf": outage_pred["impact_confidence"],
    "u_conf": outage_pred["urgency_confidence"],
}

for c in configs:
    def evaluate_row(r):
        is_low_conf = r["p_conf"] < c["t_conf"]
        is_small_margin = r["p_margin"] < c["t_margin"]
        is_comp_uncertain = (r["i_conf"] < c["t_comp"]) or (r["u_conf"] < c["t_comp"])
        is_risk = (r["pred_priority"] == "Low") and (r["p_high"] >= c["risk_p_high"])
        is_kw = False
        if c["check_kw"] and r["pred_priority"] == "Low":
            is_kw = bool(EMERGENCY_WORDS.search(r["text"]))
        return is_low_conf or is_small_margin or is_comp_uncertain or is_risk or is_kw

    df["flag"] = df.apply(evaluate_row, axis=1)
    rev_rate = df["flag"].mean() * 100
    auto_df = df[~df["flag"]]
    auto_cov = 100.0 - rev_rate
    auto_acc = auto_df["is_correct"].mean() * 100 if len(auto_df) > 0 else 0.0
    
    # High recall among all samples if review is treated as catching errors
    # Specifically, check how many true High samples are either correctly predicted autonomously or sent to human review
    true_high = df[df["true_priority"] == "High"]
    # correctly autonomous High OR flagged for human review
    high_caught = ((true_high["pred_priority"] == "High") & (~true_high["flag"])) | true_high["flag"]
    high_recall = high_caught.mean() * 100
    
    # False low: true is High/Medium, but predicted Low
    false_low = df[(df["pred_priority"] == "Low") & (df["true_priority"].isin(["High", "Medium"]))]
    fl_flagged = false_low["flag"].sum()
    fl_total = len(false_low)
    
    # Outage test
    outage_flag = evaluate_row(outage_data)
    
    print(f"{c['name']:<42} | {rev_rate:>6.1f}% | {auto_cov:>5.1f}% | {auto_acc:>7.2f}% | {high_recall:>9.1f}% | {fl_flagged:>2}/{fl_total:<2} ({fl_flagged/fl_total*100:.0f}%) | {str(outage_flag):<14}")

print("=" * 115)
