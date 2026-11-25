"""Independent evaluation of Two-Stage Priority ML Model on the locked test set.

Evaluates models/priority_model.joblib on data/processed/ticketsense_test.csv.
Note: Test set is 100% synthetic (Claude 3.5 Sonnet) and priority labels
are rubric-derived (Impact x Urgency matrix).
This benchmark is conducted strictly for historical evaluation, not tuning.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

MODEL_PATH = PROJECT_ROOT / "models" / "priority_model.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ticketsense_test.csv"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "reports" / "PRIORITY_LOCKED_TEST_RESULTS.json"

print(f"Loading Priority model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"Loading locked test dataset from: {TEST_DATA_PATH}")
df_test = pd.read_csv(TEST_DATA_PATH)

if "impact" not in df_test.columns and "priority_labels" in df_test.columns:
    df_test["impact"] = df_test["priority_labels"].apply(lambda x: str(x).split("|")[0])
    df_test["urgency"] = df_test["priority_labels"].apply(lambda x: str(x).split("|")[1])

print(f"Total test samples: {len(df_test)}")
print("Test priority distribution:\n", df_test["priority"].value_counts())
print("Test impact distribution:\n", df_test["impact"].value_counts())
print("Test urgency distribution:\n", df_test["urgency"].value_counts())

preds = model.predict(df_test["ticket_text"])

acc = accuracy_score(df_test["priority"], preds)
macro_p = precision_score(df_test["priority"], preds, average="macro", zero_division=0)
macro_r = recall_score(df_test["priority"], preds, average="macro", zero_division=0)
macro_f1 = f1_score(df_test["priority"], preds, average="macro", zero_division=0)
weighted_f1 = f1_score(df_test["priority"], preds, average="weighted", zero_division=0)

report_dict = classification_report(df_test["priority"], preds, output_dict=True, zero_division=0)
report_text = classification_report(df_test["priority"], preds, digits=4, zero_division=0)
conf_matrix = confusion_matrix(df_test["priority"], preds, labels=["High", "Medium", "Low"]).tolist()

# Component accuracy
if hasattr(model, "predict_components"):
    imp_preds, urg_preds, _ = model.predict_components(df_test["ticket_text"])
    imp_acc = float(accuracy_score(df_test["impact"], imp_preds))
    urg_acc = float(accuracy_score(df_test["urgency"], urg_preds))
else:
    imp_acc = None
    urg_acc = None

print("\n" + "="*60)
print("TWO-STAGE PRIORITY MODEL - FINAL LOCKED TEST SET EVALUATION")
print("="*60)
print(f"Test Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
print(f"Macro F1-Score:        {macro_f1:.4f}")
print(f"Macro Precision:       {macro_p:.4f}")
print(f"Macro Recall:          {macro_r:.4f}")
print(f"Weighted F1-Score:     {weighted_f1:.4f}")
if imp_acc is not None:
    print(f"Impact Accuracy:       {imp_acc:.4f}")
    print(f"Urgency Accuracy:      {urg_acc:.4f}")
print("\nClassification Report:\n", report_text)
print("Confusion Matrix [High, Medium, Low]:\n", conf_matrix)
print("="*60)

test_results = {
    "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "model_path": str(MODEL_PATH),
    "test_dataset": str(TEST_DATA_PATH),
    "test_samples": len(df_test),
    "architecture": "Two-Stage Rubric Model (Impact LR + Urgency LR -> 3x3 Rubric)",
    "metrics": {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "impact_accuracy": imp_acc,
        "urgency_accuracy": urg_acc,
    },
    "per_class": {
        "High": report_dict.get("High", {}),
        "Medium": report_dict.get("Medium", {}),
        "Low": report_dict.get("Low", {}),
    },
    "confusion_matrix_labels": ["High", "Medium", "Low"],
    "confusion_matrix": conf_matrix,
    "historical_context": (
        "Locked test set consists of 100 synthetic tickets generated via Claude 3.5 Sonnet. "
        "Labels are rubric-derived (Impact x Urgency matrix). Evaluated post-training without data leakage."
    )
}

OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(test_results, f, indent=2)

print(f"Detailed locked test benchmark saved to: {OUTPUT_REPORT_PATH}")
