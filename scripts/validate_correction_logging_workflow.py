import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
import pandas as pd
from ticketsense.data.corrections import log_manual_correction, load_corrections_log, CORRECTIONS_HEADER
from ticketsense.inference.engine import InferenceEngine

print("=" * 80)
print("TESTING DUAL-CHANNEL CORRECTION LOGGING WORKFLOW")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmp_dir:
    test_csv = Path(tmp_dir) / "test_corrections.csv"

    # Test 1: Category-only correction
    print("\n1. Testing Category-only correction...")
    res1 = log_manual_correction(
        ticket_text="My account password reset loop is broken.",
        predicted_category="Technical Issue",
        predicted_confidence=0.55,
        corrected_category="Account Access",
        corrected_by="agent_alpha",
        review_notes="User issue is password authentication, not general tech issue.",
        file_path=test_csv,
    )
    assert res1 is True
    df1 = load_corrections_log(test_csv)
    assert len(df1) == 1
    assert df1.iloc[0]["corrected_category"] == "Account Access"
    assert pd.isna(df1.iloc[0]["corrected_priority"]) or df1.iloc[0]["corrected_priority"] is None
    print("   [PASS] Category-only correction logged successfully.")

    # Test 2: Priority-only correction
    print("\n2. Testing Priority-only correction...")
    res2 = log_manual_correction(
        ticket_text="Slow loading times on transaction export screen.",
        predicted_category="Technical Issue",
        predicted_confidence=0.88,
        corrected_by="agent_beta",
        review_notes="Performance glitch is low urgency, non-blocking.",
        file_path=test_csv,
        predicted_priority="Medium",
        predicted_priority_confidence=0.58,
        corrected_priority="Low",
    )
    assert res2 is True
    df2 = load_corrections_log(test_csv)
    assert len(df2) == 2
    assert df2.iloc[1]["corrected_priority"] == "Low"
    assert df2.iloc[1]["predicted_priority"] == "Medium"
    assert df2.iloc[1]["corrected_category"] == "Technical Issue"
    print("   [PASS] Priority-only correction logged successfully.")

    # Test 3: Both Category and Priority corrections
    print("\n3. Testing Dual Category and Priority correction...")
    res3 = log_manual_correction(
        ticket_text="Emergency database cluster down customer checkout broken.",
        predicted_category="Technical Issue",
        predicted_confidence=0.665,
        corrected_category="Technical Issue",
        corrected_by="lead_supervisor",
        review_notes="Enterprise outage escalated to High Priority.",
        file_path=test_csv,
        predicted_priority="Low",
        predicted_priority_confidence=0.6395,
        corrected_priority="High",
    )
    assert res3 is True
    df3 = load_corrections_log(test_csv)
    assert len(df3) == 3
    assert df3.iloc[2]["corrected_category"] == "Technical Issue"
    assert df3.iloc[2]["corrected_priority"] == "High"
    assert df3.iloc[2]["predicted_priority"] == "Low"
    print("   [PASS] Dual Category and Priority correction logged successfully.")

    # Test 4: Historical records with missing Priority values
    print("\n4. Testing Historical records with missing Priority values...")
    # Simulate a historical CSV with only 7 columns
    hist_csv = Path(tmp_dir) / "historical_7col.csv"
    hist_data = pd.DataFrame([{
        "timestamp": "2026-11-20T10:00:00Z",
        "ticket_text": "Historical ticket without priority fields.",
        "predicted_category": "Billing",
        "predicted_confidence": 0.91,
        "corrected_category": "Billing",
        "corrected_by": "legacy_agent",
        "review_notes": "Pre-priority logging era.",
    }])
    hist_data.to_csv(hist_csv, index=False)
    
    loaded_hist = load_corrections_log(hist_csv)
    assert len(loaded_hist) == 1
    assert list(loaded_hist.columns) == CORRECTIONS_HEADER
    assert pd.isna(loaded_hist.iloc[0]["predicted_priority"])
    assert pd.isna(loaded_hist.iloc[0]["corrected_priority"])
    print("   [PASS] Historical 7-column CSV loaded with graceful None/NaN for Priority fields without fabricating values.")

    # Test 5: Reloading correction log after new entries appended to historical CSV
    print("\n5. Testing Reloading after appending to historical CSV...")
    log_manual_correction(
        ticket_text="New ticket appended to historical file.",
        predicted_category="Delivery",
        predicted_confidence=0.85,
        corrected_category="Delivery",
        corrected_by="new_agent",
        review_notes="Post-migration entry.",
        file_path=hist_csv,
        predicted_priority="Low",
        predicted_priority_confidence=0.75,
        corrected_priority="Low",
    )
    reloaded = load_corrections_log(hist_csv)
    assert len(reloaded) == 2
    assert reloaded.iloc[0]["corrected_by"] == "legacy_agent"
    assert pd.isna(reloaded.iloc[0]["corrected_priority"])
    assert reloaded.iloc[1]["corrected_by"] == "new_agent"
    assert reloaded.iloc[1]["corrected_priority"] == "Low"
    print("   [PASS] Reloading and appending to historical log maintains full integrity.")

    # Test 6: Verify model predictions remain unchanged
    print("\n6. Verifying model predictions remain immutable after corrections...")
    engine = InferenceEngine()
    test_ticket = "Disputed credit card fee on recent statement."
    pred_before = engine.predict_ticket(test_ticket)
    # Log an override
    log_manual_correction(
        ticket_text=test_ticket,
        predicted_category=pred_before["predicted_category"],
        predicted_confidence=pred_before["confidence"],
        corrected_category="Billing",
        corrected_by="test_reviewer",
        file_path=test_csv,
    )
    pred_after = engine.predict_ticket(test_ticket)
    assert pred_before["predicted_category"] == pred_after["predicted_category"]
    assert pred_before["confidence"] == pred_after["confidence"]
    assert pred_before["priority"] == pred_after["priority"]
    assert pred_before["priority_confidence"] == pred_after["priority_confidence"]
    print("   [PASS] Model predictions remain 100% immutable and independent of logged overrides.")

print("=" * 80)
print("ALL CORRECTION LOGGING WORKFLOWS 100% VERIFIED.")
print("=" * 80)
