"""Package final supervisor-ready submission ZIP and test clean extraction.

Includes all code, models, datasets, reports, tests, and documentation.
Excludes .git, .venv, __pycache__, and temporary files.
Verifies ZIP integrity with testzip().
Performs clean extraction verification test in a temporary folder.
"""

import sys
import zipfile
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "delivery"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = OUTPUT_DIR / "TicketSense-AI_Final_Submission.zip"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "delivery",
    "node_modules",
}

EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".tmp",
}

print("="*70)
print(f"PACKAGING FINAL SUBMISSION: {ZIP_PATH.name}")
print("="*70)

included_count = 0
total_uncompressed_bytes = 0

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in PROJECT_ROOT.rglob("*"):
        parts = path.relative_to(PROJECT_ROOT).parts
        if any(part in EXCLUDE_DIRS for part in parts):
            continue

        if path.is_file():
            if path.name in EXCLUDE_FILES or path.suffix in EXCLUDE_EXTENSIONS:
                continue

            rel_path = path.relative_to(PROJECT_ROOT)
            zf.write(path, arcname=str(rel_path))
            included_count += 1
            total_uncompressed_bytes += path.stat().st_size

print(f"Packaged {included_count} files ({total_uncompressed_bytes / (1024*1024):.2f} MB uncompressed).")
compressed_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
print(f"Final compressed size: {compressed_mb:.2f} MB.")

print("\n1. VERIFYING ZIP ARCHIVE INTEGRITY:")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    bad_file = zf.testzip()
    if bad_file:
        raise RuntimeError(f"Corrupt file detected in ZIP: {bad_file}")
    print("   [PASS] testzip() returned 0 corrupt files.")

print("\n2. CLEAN EXTRACTION VERIFICATION TEST:")
temp_extract_dir = Path(tempfile.mkdtemp(prefix="ticketsense_pkg_test_"))
try:
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(temp_extract_dir)

    print(f"   Extracted to: {temp_extract_dir}")
    required_paths = [
        "app.py",
        "README.md",
        "requirements.txt",
        "reports/FINAL_PROJECT_REPORT.md",
        "screenshots/Category_Model/01_category_confusion_matrix.png",
        "screenshots/Priority_Model/01_priority_confusion_matrix.png",
        "screenshots/Application_Overview/01_main_application_interface.png",
        "screenshots/Testing_Evidence/01_automated_test_suite_execution.png",
        "models/category_model.joblib",
        "models/category_model_metadata.json",
        "models/priority_model.joblib",
        "models/priority_model_metadata.json",
        "data/processed/ticketsense_dataset.csv",
        "data/processed/ticketsense_test.csv",
        "ticketsense/inference/engine.py",
        "tests/test_priority.py",
    ]

    for p in required_paths:
        target = temp_extract_dir / p
        assert target.exists(), f"Missing required file in extracted package: {p}"
    print(f"   [PASS] All {len(required_paths)} required artifacts verified in extracted package.")

    # Test execution in extracted package
    sys.path.insert(0, str(temp_extract_dir))
    from ticketsense.inference.engine import InferenceEngine
    test_engine = InferenceEngine(
        model_path=temp_extract_dir / "models" / "category_model.joblib",
        metadata_path=temp_extract_dir / "models" / "category_model_metadata.json",
    )
    res = test_engine.predict_ticket("My account was billed twice this morning.")
    assert res["success"] is True
    assert res["predicted_category"] == "Billing"
    assert res["priority"] in ["High", "Medium", "Low"]
    print(f"   [PASS] Isolated extracted package successfully executed inference: {res['predicted_category']} (Priority: {res['priority']}).")

finally:
    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    print("   [PASS] Temporary extraction directory cleaned up.")

print("\n" + "="*70)
print("SUBMISSION PACKAGE IS 100% VERIFIED AND READY FOR SUPERVISOR HANDOVER.")
print("="*70)
