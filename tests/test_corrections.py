"""Unit tests for manual correction logging and dataset immutability."""

import pytest
import pandas as pd
from pathlib import Path
from ticketsense.data.corrections import log_manual_correction, load_corrections_log
from ticketsense.config import APPROVED_CATEGORIES, APPROVED_PRIORITIES


def test_log_manual_correction(tmp_path: Path):
    temp_csv = tmp_path / "runtime_corrections.csv"

    success = log_manual_correction(
        ticket_text="Charged twice on statement.",
        predicted_category="Technical Issue",
        predicted_confidence=0.58,
        corrected_category="Billing",
        corrected_by="test_agent",
        review_notes="Explicit credit card dispute.",
        file_path=temp_csv,
    )

    assert success is True
    assert temp_csv.exists()

    df = load_corrections_log(temp_csv)
    assert len(df) == 1
    assert df.iloc[0]["corrected_category"] == "Billing"
    assert df.iloc[0]["predicted_category"] == "Technical Issue"
    assert df.iloc[0]["corrected_by"] == "test_agent"


def test_log_manual_correction_dual_category_and_priority(tmp_path: Path):
    """Verify logging overrides for both Category and Priority simultaneously."""
    temp_csv = tmp_path / "runtime_corrections.csv"

    success = log_manual_correction(
        ticket_text="Emergency: database crashed and customer checkout is completely down nationwide.",
        predicted_category="Technical Issue",
        predicted_confidence=0.665,
        corrected_category="Technical Issue",
        corrected_by="supervisor_user",
        review_notes="Escalated to High Priority due to critical enterprise impact.",
        file_path=temp_csv,
        predicted_priority="Low",
        predicted_priority_confidence=0.6395,
        corrected_priority="High",
    )

    assert success is True
    df = load_corrections_log(temp_csv)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["predicted_category"] == "Technical Issue"
    assert row["corrected_category"] == "Technical Issue"
    assert row["predicted_priority"] == "Low"
    assert row["corrected_priority"] == "High"
    assert row["corrected_by"] == "supervisor_user"


def test_log_manual_correction_invalid_category(tmp_path: Path):
    temp_csv = tmp_path / "runtime_corrections.csv"
    with pytest.raises(ValueError, match="Invalid corrected category"):
        log_manual_correction(
            ticket_text="Random text",
            predicted_category="Billing",
            predicted_confidence=0.7,
            corrected_category="NonExistentCategory",
            file_path=temp_csv,
        )


def test_log_manual_correction_invalid_priority(tmp_path: Path):
    """Verify ValueError is raised if corrected priority is not in APPROVED_PRIORITIES."""
    temp_csv = tmp_path / "runtime_corrections.csv"
    with pytest.raises(ValueError, match="Invalid corrected priority"):
        log_manual_correction(
            ticket_text="Random text",
            predicted_category="Billing",
            predicted_confidence=0.7,
            corrected_category="Billing",
            corrected_priority="UrgentCritical",
            file_path=temp_csv,
        )


def test_load_corrections_empty(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.csv"
    df = load_corrections_log(non_existent)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
