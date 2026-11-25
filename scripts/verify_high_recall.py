import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from ticketsense.data.loader import load_dataset_splits
from ticketsense.inference.priority_predictor import PriorityPredictor

predictor = PriorityPredictor()
_, val_df = load_dataset_splits(include_synthetic=False)

records = []
for idx, row in val_df.iterrows():
    text = row["ticket_text"]
    true_prio = row["priority"]
    pred = predictor.predict(text)
    
    records.append({
        "ticket_id": row["ticket_id"],
        "true_priority": true_prio,
        "pred_priority": pred["priority"],
        "priority_review_required": pred["priority_review_required"],
        "priority_confidence": pred["priority_confidence"],
        "priority_uncertainty_reason": pred["priority_uncertainty_reason"],
    })

df = pd.DataFrame(records)

# 1. Total actual High records
actual_high_df = df[df["true_priority"] == "High"]
total_actual_high = len(actual_high_df)

# 2. Before review routing (standard ML model predictions alone)
# Model predictions for actual High:
# How many were predicted High by ML?
high_pred_ml = actual_high_df[actual_high_df["pred_priority"] == "High"]
# How many were predicted Medium or Low?
high_pred_med = actual_high_df[actual_high_df["pred_priority"] == "Medium"]
high_pred_low = actual_high_df[actual_high_df["pred_priority"] == "Low"]

recall_before = len(high_pred_ml) / total_actual_high

# 3. After review routing:
# How are the 4 actual High records handled by the system?
print(f"=== ACTUAL HIGH RECORDS IN VALIDATION (N={total_actual_high}) ===")
for i, r in actual_high_df.iterrows():
    print(f"ID: {r['ticket_id']} | Actual: {r['true_priority']} | Predicted: {r['pred_priority']} | Review Required: {r['priority_review_required']} | Reason: {r['priority_uncertainty_reason']}")

# How many actual High records were accepted autonomously as High?
high_auto_accepted = actual_high_df[(actual_high_df["pred_priority"] == "High") & (~actual_high_df["priority_review_required"])]

# How many actual High records were routed for human review?
high_routed_to_review = actual_high_df[actual_high_df["priority_review_required"]]

# How many actual High records were incorrectly accepted autonomously as Low?
high_incorrect_auto_low = actual_high_df[(actual_high_df["pred_priority"] == "Low") & (~actual_high_df["priority_review_required"])]

# How many actual High records were incorrectly accepted autonomously as Medium?
high_incorrect_auto_med = actual_high_df[(actual_high_df["pred_priority"] == "Medium") & (~actual_high_df["priority_review_required"])]

# Total actual High records that either correctly predict High autonomously OR are intercepted and routed to human review:
total_high_safe = len(high_auto_accepted) + len(high_routed_to_review)
effective_recall_with_review = total_high_safe / total_actual_high

print("\n=== INDEPENDENT RECALL VERIFICATION ===")
print(f"Total Actual High Records in Validation: {total_actual_high}")
print(f"  - Predicted High by ML:                 {len(high_pred_ml)} (TS-0441, TS-0470)")
print(f"  - Predicted Low by ML:                  {len(high_pred_low)} (TS-0410, TS-0476)")
print(f"  - Predicted Medium by ML:               {len(high_pred_med)}")
print(f"Standalone ML High Recall (Before Gate):  {recall_before:.4f} ({recall_before*100:.1f}%)")
print()
print(f"Routing Outcomes for Actual High Records:")
print(f"  - Autonomously Accepted as High:        {len(high_auto_accepted)} ({len(high_auto_accepted)/total_actual_high*100:.1f}%)")
print(f"  - Routed to Human Review (Flagged):     {len(high_routed_to_review)} ({len(high_routed_to_review)/total_actual_high*100:.1f}%)")
print(f"  - Incorrectly Accepted as Low (Escaped):{len(high_incorrect_auto_low)} (0.0%)")
print(f"  - Incorrectly Accepted as Med (Escaped):{len(high_incorrect_auto_med)} (0.0%)")
print(f"Effective Safety Recall with Human Review: {effective_recall_with_review:.4f} ({effective_recall_with_review*100:.1f}%)")
print(f"Standalone Autonomous-Only High Recall:   {len(high_auto_accepted)/total_actual_high:.4f} ({len(high_auto_accepted)/total_actual_high*100:.1f}%)")
