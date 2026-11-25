import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def audit_datasets():
    datasets = {
        "ticketsense_dataset.csv": PROJECT_ROOT / "data" / "processed" / "ticketsense_dataset.csv",
        "ticketsense_test.csv": PROJECT_ROOT / "data" / "processed" / "ticketsense_test.csv",
        "dataset_review_required.csv": PROJECT_ROOT / "data" / "processed" / "dataset_review_required.csv",
        "ticketsense_synthetic_expansion_v1.csv": PROJECT_ROOT / "data" / "processed" / "ticketsense_synthetic_expansion_v1.csv",
        "priority_supplementary_training.csv": PROJECT_ROOT / "data" / "processed" / "priority_supplementary_training.csv",
    }

    print("="*80)
    print("DATA SCIENCE AUDIT REPORT: DATASETS STRUCTURE & QUALITY")
    print("="*80)

    loaded_dfs = {}
    for name, path in datasets.items():
        if not path.exists():
            print(f"[MISSING] {name}")
            continue
        df = pd.read_csv(path)
        loaded_dfs[name] = df
        print(f"\n--- {name} ---")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Missing values:\n{df.isnull().sum().to_dict()}")
        
        # Text column identification
        text_col = "ticket_text" if "ticket_text" in df.columns else ("text" if "text" in df.columns else None)
        if text_col:
            dup_texts = df[text_col].duplicated().sum()
            print(f"Duplicate texts: {dup_texts}")
            lengths = df[text_col].dropna().apply(lambda x: len(str(x)))
            print(f"Text length: min={lengths.min()}, mean={lengths.mean():.1f}, median={lengths.median():.1f}, max={lengths.max()}")
        
        # Splits
        if "split" in df.columns:
            print(f"Split distribution:\n{df['split'].value_counts().to_dict()}")
        
        # Category distribution
        if "category" in df.columns:
            print(f"Category distribution:\n{df['category'].value_counts().to_dict()}")
        
        # Impact, Urgency, Priority distribution
        for col in ["impact", "urgency", "priority"]:
            if col in df.columns:
                print(f"{col.capitalize()} distribution:\n{df[col].value_counts().to_dict()}")

    print("\n" + "="*80)
    print("LEAKAGE ANALYSIS ACROSS SPLITS")
    print("="*80)
    
    if "ticketsense_dataset.csv" in loaded_dfs:
        full_df = loaded_dfs["ticketsense_dataset.csv"]
        train_texts = set(full_df[full_df["split"] == "train"]["ticket_text"].dropna().str.strip().str.lower())
        val_texts = set(full_df[full_df["split"] == "validation"]["ticket_text"].dropna().str.strip().str.lower())
        overlap_train_val = train_texts.intersection(val_texts)
        print(f"Overlap between Train (N={len(train_texts)}) and Validation (N={len(val_texts)}): {len(overlap_train_val)} tickets")

        if "ticketsense_test.csv" in loaded_dfs:
            test_texts = set(loaded_dfs["ticketsense_test.csv"]["ticket_text"].dropna().str.strip().str.lower())
            overlap_train_test = train_texts.intersection(test_texts)
            overlap_val_test = val_texts.intersection(test_texts)
            print(f"Overlap between Train and Locked Test Set: {len(overlap_train_test)} tickets")
            print(f"Overlap between Val and Locked Test Set: {len(overlap_val_test)} tickets")
            
        if "ticketsense_synthetic_expansion_v1.csv" in loaded_dfs:
            syn_texts = set(loaded_dfs["ticketsense_synthetic_expansion_v1.csv"]["ticket_text"].dropna().str.strip().str.lower())
            overlap_syn_test = syn_texts.intersection(test_texts)
            print(f"Overlap between Synthetic Expansion and Locked Test Set: {len(overlap_syn_test)} tickets")

    print("\n" + "="*80)
    print("PRIORITY LABELS AUDIT: TRAIN VS VALIDATION VS TEST")
    print("="*80)
    if "ticketsense_dataset.csv" in loaded_dfs:
        full_df = loaded_dfs["ticketsense_dataset.csv"]
        train_df = full_df[full_df["split"] == "train"]
        val_df = full_df[full_df["split"] == "validation"]
        print("Train Priority counts:\n", train_df["priority"].value_counts(normalize=True).apply(lambda x: f"{x:.1%}").to_dict())
        print("Val Priority counts:\n", val_df["priority"].value_counts(normalize=True).apply(lambda x: f"{x:.1%}").to_dict())
    if "ticketsense_test.csv" in loaded_dfs:
        test_df = loaded_dfs["ticketsense_test.csv"]
        print("Test Priority counts:\n", test_df["priority"].value_counts(normalize=True).apply(lambda x: f"{x:.1%}").to_dict())

if __name__ == "__main__":
    audit_datasets()
