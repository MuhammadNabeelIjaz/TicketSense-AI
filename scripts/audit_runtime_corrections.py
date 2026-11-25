import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from ticketsense.config import RUNTIME_CORRECTIONS_PATH

df = pd.read_csv(RUNTIME_CORRECTIONS_PATH, encoding="utf-8")

print(f"Total rows in {RUNTIME_CORRECTIONS_PATH.name}: {len(df)}")
print(f"Columns in CSV: {list(df.columns)}")
print("=" * 100)

qa_rows = []
genuine_rows = []

for idx, row in df.iterrows():
    row_num = idx + 2  # 1-indexed, accounting for header
    reviewer = str(row.get("corrected_by", ""))
    notes = str(row.get("review_notes", "")) if pd.notna(row.get("review_notes")) else ""
    ts = str(row.get("timestamp", ""))
    text_preview = str(row.get("ticket_text", ""))[:50].replace("\n", " ")
    pred_cat = str(row.get("predicted_category", ""))
    corr_cat = str(row.get("corrected_category", ""))
    pred_prio = str(row.get("predicted_priority", "None"))
    corr_prio = str(row.get("corrected_priority", "None"))
    
    is_qa = any(x in reviewer.lower() for x in ["qa", "test"]) or any(x in notes.lower() for x in ["test", "verification", "automated"])
    
    if is_qa:
        qa_rows.append(row_num)
        record_type = "QA / TEST GENERATED"
    else:
        genuine_rows.append(row_num)
        record_type = "GENUINE / OPERATIONAL"
    
    print(f"Row {row_num:>2} | {record_type:<21} | Reviewer: {reviewer:<20} | Cat: {pred_cat:<15} -> {corr_cat:<15} | Notes: {notes[:35]}")
    print(f"       Text: {text_preview}...")

print("=" * 100)
print(f"Summary of Records:")
print(f"  - Genuine / Operational Rows: {genuine_rows} (Total: {len(genuine_rows)})")
print(f"  - QA / Test-Generated Rows:   {qa_rows} (Total: {len(qa_rows)})")

# Check for duplicates by ticket_text and timestamp
print("\nDuplicate Check:")
text_counts = df["ticket_text"].value_counts()
repeated_texts = text_counts[text_counts > 1]
print(f"Unique ticket texts repeated across rows:")
for text, cnt in repeated_texts.items():
    matching_rows = df[df["ticket_text"] == text]
    print(f"  '{text[:50]}...' repeated {cnt} times across rows {list(matching_rows.index + 2)}")

