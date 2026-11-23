"""Integration tests for batch inference workflow."""

import pytest
import pandas as pd
from ticketsense.config import APPROVED_CATEGORIES


def test_batch_prediction_workflow(inference_engine, sample_batch_df):
    success, msg, enriched_df, stats = inference_engine.predict_batch(sample_batch_df)

    assert success is True
    assert "Processed 5 tickets" in msg
    assert len(enriched_df) == 5

    expected_cols = [
        "ticket_text",
        "predicted_category",
        "confidence",
        "is_uncertain",
        "uncertainty_reason",
        "provisional_priority_heuristic",
        "priority_status",
        "issue_summary",
    ]
    for col in expected_cols:
        assert col in enriched_df.columns

    for cat in enriched_df["predicted_category"]:
        assert cat in APPROVED_CATEGORIES

    assert stats["total_processed"] == 5
    assert 0 <= stats["uncertain_count"] <= 5
    assert isinstance(stats["category_distribution"], dict)


def test_batch_prediction_with_dirty_dataframe(inference_engine):
    dirty_df = pd.DataFrame({
        "ticket_text": [
            None,
            "nan",
            "Short",  # 5 chars -> valid
            12345,
            99.99,
            "I need help recovering my lost account password.",
        ]
    })
    success, msg, enriched_df, stats = inference_engine.predict_batch(dirty_df)

    assert success is True
    assert len(enriched_df) >= 3
    assert "provisional_priority_heuristic" in enriched_df.columns
    for text in enriched_df["ticket_text"]:
        assert isinstance(text, str)
        assert len(text) >= 5


def test_batch_prediction_heuristic_column_naming(inference_engine, sample_batch_df):
    _, _, enriched_df, _ = inference_engine.predict_batch(sample_batch_df)
    # Ensure bare ambiguous 'priority' column is not used in batch results
    assert "provisional_priority_heuristic" in enriched_df.columns
    assert "priority" not in enriched_df.columns


def test_batch_prediction_rejected_rows_reporting(inference_engine):
    mixed_df = pd.DataFrame({
        "ticket_text": [
            None,
            "hi",  # too short (< 5 chars)
            "Valid customer ticket message here.",
            "Another valid message for classification.",
        ]
    })
    success, msg, enriched_df, stats = inference_engine.predict_batch(mixed_df)
    assert success is True
    assert stats["total_uploaded"] == 4
    assert stats["total_processed"] == 2
    assert stats["rejected_count"] == 2

    rejected_df = stats["rejected_df"]
    assert "original_row_number" in rejected_df.columns
    assert "rejection_reason" in rejected_df.columns
    assert list(rejected_df["original_row_number"]) == [1, 2]
    assert "Missing or null" in rejected_df.iloc[0]["rejection_reason"]
    assert "too short" in rejected_df.iloc[1]["rejection_reason"].lower()


def test_batch_prediction_all_rejected(inference_engine):
    invalid_df = pd.DataFrame({"ticket_text": [None, "nan", "  "]})
    success, msg, enriched_df, stats = inference_engine.predict_batch(invalid_df)
    assert success is False
    assert stats["rejected_count"] == 3
    assert len(enriched_df) == 0
