"""Deep data quality investigation for Priority Classification.

Analyzes:
1. Impact and Urgency distributions across Train, Validation, and Test.
2. Label consistency and rubric adherence.
3. High priority sample inspect (16 Train, 4 Val, 19 Test).
4. Error analysis of validation false negatives.
5. Vocabulary overlap and distribution shift.
6. Duplicate and near-duplicate analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

dev_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "ticketsense_dataset.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "ticketsense_test.csv")

for df in [dev_df, test_df]:
    if "impact" not in df.columns:
        df["impact"] = df["priority_labels"].apply(lambda x: str(x).split("|")[0])
        df["urgency"] = df["priority_labels"].apply(lambda x: str(x).split("|")[1])

train_df = dev_df[dev_df["split"] == "train"].copy().reset_index(drop=True)
val_df = dev_df[dev_df["split"] == "validation"].copy().reset_index(drop=True)

print("="*70)
print("PHASE 2: DATASET QUALITY INVESTIGATION")
print("="*70)

print("\n1. IMPACT & URGENCY JOINT CELL DISTRIBUTIONS:")
print("\nTrain Split (400 samples):")
print(pd.crosstab(train_df["impact"], train_df["urgency"], margins=True))
print("\nValidation Split (100 samples):")
print(pd.crosstab(val_df["impact"], val_df["urgency"], margins=True))
print("\nLocked Test Split (100 samples):")
print(pd.crosstab(test_df["impact"], test_df["urgency"], margins=True))

print("\n2. PRIORITY DISTRIBUTION SUMMARY:")
dist_summary = pd.DataFrame({
    "Train Count": train_df["priority"].value_counts(),
    "Train %": (train_df["priority"].value_counts(normalize=True) * 100).round(2),
    "Val Count": val_df["priority"].value_counts(),
    "Val %": (val_df["priority"].value_counts(normalize=True) * 100).round(2),
    "Test Count": test_df["priority"].value_counts(),
    "Test %": (test_df["priority"].value_counts(normalize=True) * 100).round(2),
})
print(dist_summary)

print("\n3. HIGH PRIORITY SAMPLES IN TRAINING DATA (16 total):")
for i, r in train_df[train_df["priority"] == "High"].iterrows():
    print(f"  [{r['ticket_id']}] ({r['priority_labels']} -> {r['category']}): {r['ticket_text']}")

print("\n4. HIGH PRIORITY SAMPLES IN VALIDATION DATA (4 total):")
for i, r in val_df[val_df["priority"] == "High"].iterrows():
    print(f"  [{r['ticket_id']}] ({r['priority_labels']} -> {r['category']}): {r['ticket_text']}")

# Load current model to diagnose false negatives
model_path = PROJECT_ROOT / "models" / "priority_model.joblib"
if model_path.exists():
    model = joblib.load(model_path)
    val_preds = model.predict(val_df["ticket_text"])
    imp_preds, urg_preds, _ = model.predict_components(val_df["ticket_text"])
    val_df["pred_priority"] = val_preds
    val_df["pred_impact"] = imp_preds
    val_df["pred_urgency"] = urg_preds

    print("\n5. VALIDATION FALSE NEGATIVE ANALYSIS (Missed High Tickets):")
    missed_high = val_df[(val_df["priority"] == "High") & (val_df["pred_priority"] != "High")]
    for i, r in missed_high.iterrows():
        print(f"  [{r['ticket_id']}] TRUE: {r['priority_labels']} ({r['priority']}) | PRED: {r['pred_impact']}|{r['pred_urgency']} ({r['pred_priority']})")
        print(f"    Text: {r['ticket_text']}")

    detected_high = val_df[(val_df["priority"] == "High") & (val_df["pred_priority"] == "High")]
    print("\n6. VALIDATION DETECTED HIGH TICKETS (True Positives):")
    for i, r in detected_high.iterrows():
        print(f"  [{r['ticket_id']}] TRUE: {r['priority_labels']} ({r['priority']}) | PRED: {r['pred_impact']}|{r['pred_urgency']} ({r['pred_priority']})")
        print(f"    Text: {r['ticket_text']}")

print("\n7. VOCABULARY OVERLAP ANALYSIS:")
vec = TfidfVectorizer().fit(train_df["ticket_text"])
train_vocab = set(vec.vocabulary_.keys())

val_high_words = set(" ".join(val_df[val_df["priority"] == "High"]["ticket_text"]).lower().split())
test_high_words = set(" ".join(test_df[test_df["priority"] == "High"]["ticket_text"]).lower().split())

val_oov = val_high_words - train_vocab
test_oov = test_high_words - train_vocab
print(f"Total training vocabulary words: {len(train_vocab)}")
print(f"Validation High words not in training vocabulary: {len(val_oov)} ({len(val_oov)/max(1, len(val_high_words))*100:.1f}%)")
print(f"Test High words not in training vocabulary: {len(test_oov)} ({len(test_oov)/max(1, len(test_high_words))*100:.1f}%)")
print("Sample Test High OOV words:", list(test_oov)[:15])

print("\n8. DUPLICATE AND NEAR-DUPLICATE AUDIT:")
all_texts = dev_df["ticket_text"].tolist()
vec_all = TfidfVectorizer().fit_transform(all_texts)
sim = cosine_similarity(vec_all)
np.fill_diagonal(sim, 0)
high_sim_pairs = np.where(sim > 0.85)
print(f"Pairs with cosine similarity > 0.85 in dev dataset: {len(high_sim_pairs[0]) // 2}")
