import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from ticketsense.data.loader import load_dataset_splits
from ticketsense.inference.priority_predictor import PriorityPredictor
from ticketsense.inference.uncertainty import compute_priority_uncertainty_score

predictor = PriorityPredictor()
_, val_df = load_dataset_splits(include_synthetic=False)

records = []
for idx, row in val_df.iterrows():
    text = row["ticket_text"]
    true_prio = row["priority"]
    pred = predictor.predict(text)
    
    records.append({
        "ticket_id": row["ticket_id"],
        "ticket_text": text,
        "true_priority": true_prio,
        "pred_priority": pred["priority"],
        "predicted_impact": pred["predicted_impact"],
        "impact_confidence": pred["impact_confidence"],
        "predicted_urgency": pred["predicted_urgency"],
        "urgency_confidence": pred["urgency_confidence"],
        "priority_confidence": pred["priority_confidence"],
        "priority_margin": pred["priority_margin"],
        "priority_probabilities": pred["priority_probabilities"],
        "priority_review_required": pred["priority_review_required"],
        "priority_uncertainty_reason": pred["priority_uncertainty_reason"],
        "priority_review_details": pred.get("priority_review_details", {}),
    })

eval_df = pd.DataFrame(records)

# Filter for the 8 false-Low cases: actual in [High, Medium], predicted Low
false_low_df = eval_df[(eval_df["pred_priority"] == "Low") & (eval_df["true_priority"].isin(["High", "Medium"]))].reset_index(drop=True)

print(f"Total false-Low validation cases found: {len(false_low_df)}")
print("=" * 80)

for i, r in false_low_df.iterrows():
    print(f"CASE {i+1}:")
    print(f"  Ticket ID:           {r['ticket_id']}")
    print(f"  Actual Priority:     {r['true_priority']}")
    print(f"  Predicted Priority:  {r['pred_priority']}")
    print(f"  Predicted Impact:    {r['predicted_impact']} (conf: {r['impact_confidence']})")
    print(f"  Predicted Urgency:   {r['predicted_urgency']} (conf: {r['urgency_confidence']})")
    print(f"  Priority Conf:       {r['priority_confidence']}")
    print(f"  Priority Margin:     {r['priority_margin']}")
    print(f"  Joint Probabilities: {r['priority_probabilities']}")
    print(f"  Review Triggered:    {r['priority_review_required']}")
    print(f"  Review Reason:       {r['priority_uncertainty_reason']}")
    print(f"  Ticket Text:         {r['ticket_text'][:120]}...")
    print("-" * 80)

escaped = false_low_df[~false_low_df["priority_review_required"]]
print(f"\nTotal Escaped False-Low Cases: {len(escaped)}")
if len(escaped) > 0:
    for i, r in escaped.iterrows():
        print(f"ESCAPED CASE DETAILS:")
        print(f"  ID: {r['ticket_id']}")
        print(f"  Text: {r['ticket_text']}")
        print(f"  Actual: {r['true_priority']}, Predicted: {r['pred_priority']}")
        print(f"  Impact: {r['predicted_impact']} ({r['impact_confidence']}), Urgency: {r['predicted_urgency']} ({r['urgency_confidence']})")
        print(f"  Prio Conf: {r['priority_confidence']}, Margin: {r['priority_margin']}")
        print(f"  Joint Probs: {r['priority_probabilities']}")
        print(f"  Why it escaped the gate criteria:")
        print(f"    1. Priority confidence ({r['priority_confidence']}) >= 0.60: {r['priority_confidence'] >= 0.60}")
        print(f"    2. Priority margin ({r['priority_margin']}) >= 0.15: {r['priority_margin'] >= 0.15}")
        print(f"    3. Impact conf ({r['impact_confidence']}) >= 0.50: {r['impact_confidence'] >= 0.50}")
        print(f"    4. Urgency conf ({r['urgency_confidence']}) >= 0.50: {r['urgency_confidence'] >= 0.50}")
        print(f"    5. P(High) ({r['priority_probabilities'].get('High', 0)}) < 0.15: {r['priority_probabilities'].get('High', 0) < 0.15}")
        print(f"    6. Incident words present: False")
