import sys
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ticketsense.inference.priority_predictor import PriorityPredictor
from ticketsense.models.priority_pipeline import AUTHORITATIVE_RUBRIC

def audit_validation_priority():
    data_path = PROJECT_ROOT / "data" / "processed" / "ticketsense_dataset.csv"
    df = pd.read_csv(data_path)
    val_df = df[df["split"] == "validation"].copy()
    
    predictor = PriorityPredictor()
    
    results = []
    for idx, row in val_df.iterrows():
        text = row["ticket_text"]
        actual_prio = row["priority"]
        # In ticketsense_dataset.csv, priority_labels holds e.g. "('I3', 'U3')" or similar if present
        p_label = row.get("priority_labels", "")
        p_reason = row.get("priority_reason", "")
        
        pred = predictor.predict(text)
        
        results.append({
            "ticket_id": row.get("ticket_id", idx),
            "text": text,
            "actual_priority": actual_prio,
            "priority_labels": p_label,
            "priority_reason": p_reason,
            "predicted_priority": pred["priority"],
            "predicted_impact": pred["predicted_impact"],
            "impact_conf": pred["impact_confidence"],
            "predicted_urgency": pred["predicted_urgency"],
            "urgency_conf": pred["urgency_confidence"],
            "prio_conf": pred["priority_confidence"],
            "prio_probas": pred["priority_probabilities"],
            "impact_probas": pred.get("impact_probabilities", {}),
            "urgency_probas": pred.get("urgency_probabilities", {}),
            "review_required": pred["priority_review_required"],
            "review_reason": pred["priority_uncertainty_reason"],
        })
        
    res_df = pd.DataFrame(results)
    
    print("="*80)
    print("VALIDATION SET PRIORITY CONFUSION MATRIX & RECALL AUDIT")
    print("="*80)
    cm = pd.crosstab(res_df["actual_priority"], res_df["predicted_priority"], margins=True)
    print(cm)
    
    print("\n" + "="*80)
    print("INSPECTING ALL ACTUAL HIGH-PRIORITY VALIDATION TICKETS (N=4)")
    print("="*80)
    high_df = res_df[res_df["actual_priority"] == "High"]
    for i, r in high_df.iterrows():
        print(f"\nTicket ID: {r['ticket_id']}")
        print(f"Text: {r['text']}")
        print(f"Actual Priority: {r['actual_priority']} | Ground Truth Labels: {r['priority_labels']}")
        print(f"Predicted Priority: {r['predicted_priority']}")
        print(f"Predicted Impact: {r['predicted_impact']} ({r['impact_conf']:.3f}) | Probas: {r['impact_probas']}")
        print(f"Predicted Urgency: {r['predicted_urgency']} ({r['urgency_conf']:.3f}) | Probas: {r['urgency_probas']}")
        print(f"Joint Priority Probas: {r['prio_probas']}")
        print(f"Review Required: {r['review_required']} | Reason: {r['review_reason']}")

    print("\n" + "="*80)
    print("INSPECTING ALL FALSE-LOW PREDICTIONS (ACTUAL MEDIUM/HIGH PREDICTED LOW)")
    print("="*80)
    false_low = res_df[(res_df["actual_priority"].isin(["Medium", "High"])) & (res_df["predicted_priority"] == "Low")]
    print(f"Total False-Low records: {len(false_low)}")
    for i, r in false_low.iterrows():
        print(f"\nTicket ID: {r['ticket_id']}")
        print(f"Text: {r['text']}")
        print(f"Actual Priority: {r['actual_priority']} | Ground Truth Labels: {r['priority_labels']}")
        print(f"Predicted Priority: {r['predicted_priority']}")
        print(f"Predicted Impact: {r['predicted_impact']} ({r['impact_conf']:.3f}) | Probas: {r['impact_probas']}")
        print(f"Predicted Urgency: {r['predicted_urgency']} ({r['urgency_conf']:.3f}) | Probas: {r['urgency_probas']}")
        print(f"Joint Priority Probas: {r['prio_probas']}")
        print(f"Review Required: {r['review_required']} | Reason: {r['review_reason']}")

if __name__ == "__main__":
    audit_validation_priority()
