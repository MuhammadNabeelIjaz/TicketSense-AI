import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, log_loss

from ticketsense.config import (
    TRAIN_VAL_DATASET_PATH,
    TEST_DATASET_PATH,
    APPROVED_CATEGORIES,
)
from ticketsense.data.loader import load_dataset_splits, load_test_dataset

model = joblib.load("models/category_model.joblib")

_, val_df = load_dataset_splits(include_synthetic=False)
test_df = load_test_dataset()

def evaluate(name, df, text_col="ticket_text", label_col="category"):
    y_true = df[label_col]
    y_pred = model.predict(df[text_col])
    y_proba = model.predict_proba(df[text_col])
    classes = list(model.classes_)
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    ll = log_loss(y_true, y_proba, labels=classes)
    
    print(f"============================================================")
    print(f"{name} (N={len(df)})")
    print(f"============================================================")
    print(f"Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro Precision:  {prec:.4f}")
    print(f"Macro Recall:     {rec:.4f}")
    print(f"Macro F1:         {f1:.4f}")
    print(f"Weighted F1:      {w_f1:.4f}")
    print(f"Log Loss:         {ll:.4f}")
    print("\nClassification Report (Macro / Per-Class):")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0, labels=APPROVED_CATEGORIES))
    print("Confusion Matrix (Labels: Account Access, Billing, Delivery, General Inquiry, Technical Issue):")
    print(confusion_matrix(y_true, y_pred, labels=APPROVED_CATEGORIES))
    print()

evaluate("VALIDATION SPLIT (data/processed/ticketsense_dataset.csv)", val_df)
evaluate("LOCKED TEST SET (data/processed/ticketsense_test.csv)", test_df)
