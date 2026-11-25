"""Training script for Two-Stage Rubric Priority Classification Model.

Trains two balanced regularized Logistic Regression classifiers:
Stage 1: Predict Impact (I1, I2, I3)
Stage 2: Predict Urgency (U1, U2, U3)
Stage 3: Map through authoritative 3x3 Impact x Urgency rubric to determine Priority.

Validates on the development validation partition.
Does NOT touch the locked test set.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from ticketsense.models.priority_pipeline import train_two_stage_priority_model

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ticketsense_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "priority_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "priority_model_metadata.json"

print(f"Loading development dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print("Train priority distribution:\n", train_df["priority"].value_counts())
print("Validation priority distribution:\n", val_df["priority"].value_counts())

print("\nFitting Two-Stage Priority Model (Impact: C=2.0, Urgency: C=1.0, ngram=(1,2))...")
model, metadata = train_two_stage_priority_model(
    train_df=train_df,
    val_df=val_df,
    model_path=MODEL_PATH,
    metadata_path=METADATA_PATH,
    c_imp=2.0,
    c_urg=1.0,
    ngram_range=(1, 2),
)

metrics = metadata["validation_metrics"]
print("\n--- VALIDATION RESULTS (TWO-STAGE RUBRIC) ---")
print(f"Validation Priority Accuracy:  {metrics['priority_accuracy']:.4f}")
print(f"Validation Priority Macro F1:  {metrics['priority_macro_f1']:.4f}")
print(f"Validation Priority Macro P:   {metrics['priority_macro_precision']:.4f}")
print(f"Validation Priority Macro R:   {metrics['priority_macro_recall']:.4f}")
print(f"Validation Impact Accuracy:    {metrics['impact_accuracy']:.4f}")
print(f"Validation Urgency Accuracy:   {metrics['urgency_accuracy']:.4f}")
print("\nHigh-Priority Validation Metrics:")
print(f"  Precision: {metadata['per_class_validation']['High'].get('precision', 0):.4f}")
print(f"  Recall:    {metadata['per_class_validation']['High'].get('recall', 0):.4f}")
print(f"  F1:        {metadata['per_class_validation']['High'].get('f1-score', 0):.4f}")
print("\nConfusion Matrix [High, Medium, Low]:\n", metadata["confusion_matrix"])
print(f"\nModel artifact saved to: {MODEL_PATH}")
print(f"Metadata saved to: {METADATA_PATH}")
