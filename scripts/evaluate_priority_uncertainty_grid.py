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

EMERGENCY_WORDS = re.compile(r"\b(emergency|outage|crashed?|critical|down|escalat\w+|immediate\w*)\b", re.IGNORECASE)

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

print(f"{'Conf':<6} | {'Margin':<6} | {'Comp':<6} | {'Risk':<6} | {'KW':<5} | {'Review%':<8} | {'Auto%':<7} | {'Auto Acc':<9} | {'False Low Flagged':<18} | {'Outage Flag'}")
print("-" * 95)

grid = []
for t_conf in [0.50, 0.55, 0.58, 0.60, 0.65]:
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

                    rev_flag = df.apply(flag_fn, axis=1)
                    rev_rate = rev_flag.mean() * 100
                    auto_df = df[~rev_flag]
                    auto_acc = auto_df["is_correct"].mean() * 100 if len(auto_df) > 0 else 0.0
                    false_low = df[(df["pred_priority"] == "Low") & (df["true_priority"].isin(["High", "Medium"]))]
                    fl_flagged = false_low.apply(flag_fn, axis=1).sum()
                    outage_f = flag_fn(outage_data)
                    grid.append({
                        "t_conf": t_conf,
                        "t_margin": t_margin,
                        "t_comp": t_comp,
                        "risk_high": risk_high,
                        "kw": kw,
                        "rev_rate": rev_rate,
                        "auto_cov": 100 - rev_rate,
                        "auto_acc": auto_acc,
                        "fl_flagged": fl_flagged,
                        "fl_total": len(false_low),
                        "outage_flag": outage_f
                    })

res_df = pd.DataFrame(grid)
# Filter for configurations that catch the Outage ticket AND catch >= 7/8 false lows
promising = res_df[(res_df["outage_flag"] == True) & (res_df["fl_flagged"] >= 7)].sort_values(by=["rev_rate", "auto_acc"], ascending=[True, False])

for idx, r in promising.head(15).iterrows():
    print(f"{r['t_conf']:<6.2f} | {r['t_margin']:<6.2f} | {r['t_comp']:<6.2f} | {r['risk_high']:<6.2f} | {str(r['kw']):<5} | {r['rev_rate']:>6.1f}% | {r['auto_cov']:>5.1f}% | {r['auto_acc']:>7.2f}% | {int(r['fl_flagged']):>2}/{int(r['fl_total']):<2} ({r['fl_flagged']/r['fl_total']*100:.0f}%) | {r['outage_flag']}")
