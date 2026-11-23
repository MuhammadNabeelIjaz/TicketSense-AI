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
