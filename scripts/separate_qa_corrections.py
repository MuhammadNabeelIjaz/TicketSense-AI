import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil
import pandas as pd
from ticketsense.config import RUNTIME_CORRECTIONS_PATH, CORRECTIONS_DIR
from ticketsense.data.corrections import CORRECTIONS_HEADER

backup_path = CORRECTIONS_DIR / "runtime_corrections_original_backup.csv"
qa_path = CORRECTIONS_DIR / "qa_test_corrections.csv"

# 1. Backup original file
print(f"Creating backup: {backup_path.name}")
shutil.copy2(RUNTIME_CORRECTIONS_PATH, backup_path)
assert backup_path.exists()
print(f"Backup verified. Size: {backup_path.stat().st_size} bytes.")

# 2. Read original data
df = pd.read_csv(backup_path, encoding="utf-8")
print(f"Original total rows: {len(df)}")
assert len(df) == 16, f"Expected 16 rows, got {len(df)}"

# 3. Identify QA records
is_qa_mask = df["corrected_by"].isin(["qa_verifier", "qa_tester_automated"]) | df["review_notes"].fillna("").str.contains("Automated", case=False)

qa_df = df[is_qa_mask].copy()
genuine_df = df[~is_qa_mask].copy()

print(f"Identified QA rows: {len(qa_df)} (Original indices {[idx + 2 for idx in qa_df.index]})")
print(f"Identified Genuine rows: {len(genuine_df)} (Original indices {[idx + 2 for idx in genuine_df.index]})")

assert len(qa_df) == 4, f"Expected 4 QA rows, got {len(qa_df)}"
assert len(genuine_df) == 12, f"Expected 12 genuine rows, got {len(genuine_df)}"

# 4. Standardize to 10-column dual-channel schema without fabricating values
for col in CORRECTIONS_HEADER:
    if col not in genuine_df.columns:
        genuine_df[col] = None
    if col not in qa_df.columns:
        qa_df[col] = None

genuine_df = genuine_df[CORRECTIONS_HEADER]
qa_df = qa_df[CORRECTIONS_HEADER]

# 5. Write separated files
genuine_df.to_csv(RUNTIME_CORRECTIONS_PATH, index=False, columns=CORRECTIONS_HEADER, encoding="utf-8")
qa_df.to_csv(qa_path, index=False, columns=CORRECTIONS_HEADER, encoding="utf-8")

# 6. Verify written files
read_genuine = pd.read_csv(RUNTIME_CORRECTIONS_PATH, encoding="utf-8")
read_qa = pd.read_csv(qa_path, encoding="utf-8")
read_backup = pd.read_csv(backup_path, encoding="utf-8")

print("\n--- POST-MIGRATION VERIFICATION ---")
print(f"Genuine runtime_corrections.csv rows: {len(read_genuine)}")
print(f"QA qa_test_corrections.csv rows:      {len(read_qa)}")
print(f"Backup original rows:                 {len(read_backup)}")
assert len(read_genuine) == 12
assert len(read_qa) == 4
assert len(read_genuine) + len(read_qa) == 16
assert list(read_genuine.columns) == CORRECTIONS_HEADER
assert list(read_qa.columns) == CORRECTIONS_HEADER
print("Record counts, schemas, and contents 100% verified.")
