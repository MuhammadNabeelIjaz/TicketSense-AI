import hashlib
from pathlib import Path

EXPECTED = {
    "models/category_model.joblib": "ab9de12f1969379c9a55bd7aaa7de5b032f7544205c3f19c4b549e92a8c367f0",
    "models/priority_model.joblib": "8e323552e866f92f9a3682efa0dc5fbd596035c5c6770df39df29301bbdafdd7",
    "data/processed/ticketsense_dataset.csv": "ab28ce2070f91246cf7416b99ed732b3eac261643a61c27f9444a85b8e2e9893",
    "data/processed/ticketsense_test.csv": "356bcd20c70d96daf07e3aa0a3378cebbf7a1b129b263cf08c687dc6d1ee8615",
    "data/processed/dataset_review_required.csv": "83804298bb978c45a3563df99513c4294251ba02d6760a1128600dcbe3ac3de9",
    "data/processed/ticketsense_synthetic_expansion_v1.csv": "9be4a2cd58ae78c989a46832d3cf84f0809e9d45ab67feae9d560f89abc1e5cb",
}

print("="*70)
print("CRYPTOGRAPHIC INTEGRITY VERIFICATION LEDGER")
print("="*70)

all_passed = True
for rel_path, expected_hash in EXPECTED.items():
    file_path = Path(rel_path)
    if not file_path.exists():
        print(f"[FAIL] Missing file: {rel_path}")
        all_passed = False
        continue
    actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if actual_hash == expected_hash:
        print(f"[PASS] {rel_path}")
        print(f"       Hash: {actual_hash}")
    else:
        print(f"[FAIL] {rel_path}")
        print(f"       Expected: {expected_hash}")
        print(f"       Actual:   {actual_hash}")
        all_passed = False

print("="*70)
if all_passed:
    print("ALL 6 PROTECTED ARTIFACTS AND DATASETS ARE 100% VERIFIED AND UNMODIFIED.")
else:
    print("INTEGRITY MISMATCH DETECTED.")
