"""Rigorous validation of synthetic dataset quality, duplicates, leakage, and category integrity."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ticketsense.config import (
    APPROVED_CATEGORIES,
    TRAIN_VAL_DATASET_PATH,
    TEST_DATASET_PATH,
    PROCESSED_DATA_DIR,
    MIN_TICKET_CHAR_LENGTH,
    MAX_TICKET_CHAR_LENGTH,
)
from ticketsense.nlp.preprocessor import clean_text

SYNTHETIC_PATH = PROCESSED_DATA_DIR / "ticketsense_synthetic_expansion_v1.csv"


def run_validation():
    print("=" * 70)
    print("TicketSense-AI Synthetic Data Quality Validation")
    print("=" * 70)

    syn_df = pd.read_csv(SYNTHETIC_PATH)
    orig_train_val = pd.read_csv(TRAIN_VAL_DATASET_PATH)
    orig_train = orig_train_val[orig_train_val["split"] == "train"].copy()
    orig_val = orig_train_val[orig_train_val["split"] == "validation"].copy()
    test_df = pd.read_csv(TEST_DATASET_PATH)

    print(f"Loaded synthetic dataset: {len(syn_df)} records")

    # 1. Missing Values
    print("\n[CHECK 1] Null & Missing Value Check...")
    null_counts = syn_df.isnull().sum()
    has_nulls = null_counts.sum() > 0
    if has_nulls:
        print("  FAIL: Found missing values:\n", null_counts[null_counts > 0])
        sys.exit(1)
    print("  PASS: Zero null or missing values across all columns.")

    # 2. Character Length Bounds
    print("\n[CHECK 2] Character Length Constraints...")
    lengths = syn_df["ticket_text"].apply(lambda x: len(str(x).strip()))
    too_short = syn_df[lengths < MIN_TICKET_CHAR_LENGTH]
    too_long = syn_df[lengths > MAX_TICKET_CHAR_LENGTH]
    if len(too_short) > 0 or len(too_long) > 0:
        print(f"  FAIL: Found {len(too_short)} too short and {len(too_long)} too long.")
        sys.exit(1)
    print(f"  PASS: All texts within valid length [{MIN_TICKET_CHAR_LENGTH}, {MAX_TICKET_CHAR_LENGTH}]. Range: [{lengths.min()}, {lengths.max()}].")

    # 3. Category Validation & Balance
    print("\n[CHECK 3] Category Taxonomy & Distribution...")
    invalid_cats = set(syn_df["category"]) - set(APPROVED_CATEGORIES)
    if invalid_cats:
        print("  FAIL: Invalid categories found:", invalid_cats)
        sys.exit(1)
    print("  PASS: All categories in approved taxonomy.")
    cat_counts = syn_df["category"].value_counts()
    print("  Category breakdown:")
    for c, count in cat_counts.items():
        print(f"    - {c:<18}: {count}")

    # 4. Exact Duplicates
    print("\n[CHECK 4] Exact Duplicate Checks...")
    syn_cleaned = [clean_text(t) for t in syn_df["ticket_text"]]
    syn_dupes = len(syn_cleaned) - len(set(syn_cleaned))
    if syn_dupes > 0:
        print(f"  FAIL: {syn_dupes} internal exact duplicate texts in synthetic data.")
        sys.exit(1)
    print("  PASS: Zero internal exact duplicates in synthetic data.")

    orig_train_cleaned = set([clean_text(t) for t in orig_train["ticket_text"]])
    leak_train = [t for t in syn_cleaned if t in orig_train_cleaned]
    if leak_train:
        print(f"  FAIL: {len(leak_train)} synthetic records are identical to original training records.")
        sys.exit(1)
    print("  PASS: Zero exact duplicates between synthetic and original training data.")

    # 5. Zero Leakage Against Validation and Test Sets
    print("\n[CHECK 5] Data Leakage Guardrail Against Validation & Test Sets...")
    val_cleaned = set([clean_text(t) for t in orig_val["ticket_text"]])
    test_cleaned = set([clean_text(t) for t in test_df["ticket_text"]])

    leak_val = [t for t in syn_cleaned if t in val_cleaned]
    leak_test = [t for t in syn_cleaned if t in test_cleaned]
    if leak_val:
        print(f"  FAIL: {len(leak_val)} synthetic records leaked from validation set!")
        sys.exit(1)
    if leak_test:
        print(f"  FAIL: {len(leak_test)} synthetic records leaked from test set!")
        sys.exit(1)
    print("  PASS: Zero leakage against validation set and locked test set.")

    # 6. Semantic Near-Duplicate / Overlap Check
    print("\n[CHECK 6] Semantic Near-Duplicate Analysis (Cosine Similarity > 0.85)...")
    vec = TfidfVectorizer(ngram_range=(1, 2))
    all_texts = list(orig_train["ticket_text"]) + list(syn_df["ticket_text"])
    all_vecs = vec.fit_transform(all_texts)

    train_vecs = all_vecs[:len(orig_train)]
    syn_vecs = all_vecs[len(orig_train):]

    # Check internal similarity in synthetic data
    syn_sim = cosine_similarity(syn_vecs)
    np.fill_diagonal(syn_sim, 0)
    high_internal = np.where(syn_sim > 0.85)
    internal_pairs = len(high_internal[0]) // 2
    print(f"  Internal synthetic pairs with sim > 0.85: {internal_pairs}")

    # Check cross similarity between synthetic and training data
    cross_sim = cosine_similarity(syn_vecs, train_vecs)
    high_cross = np.where(cross_sim > 0.85)
    cross_pairs = len(high_cross[0])
    print(f"  Cross pairs (synthetic vs original train) with sim > 0.85: {cross_pairs}")

    # 7. Vocabulary Expansion
    print("\n[CHECK 7] Vocabulary Expansion Impact...")
    orig_words = set(" ".join(orig_train_cleaned).split())
    syn_words = set(" ".join(syn_cleaned).split())
    new_words = syn_words - orig_words
    combined_words = orig_words.union(syn_words)
    print(f"  Original train unique words: {len(orig_words)}")
    print(f"  Synthetic unique words:      {len(syn_words)}")
    print(f"  New unique words added:      {len(new_words)} (+{len(new_words)/len(orig_words)*100:.1f}%)")
    print(f"  Total expanded vocabulary:   {len(combined_words)}")

    print("\n" + "=" * 70)
    print("VERDICT: SYNTHETIC DATA QUALITY VALIDATION PASSED (READY FOR ML EXPERIMENTS)")
    print("=" * 70)


if __name__ == "__main__":
    run_validation()
