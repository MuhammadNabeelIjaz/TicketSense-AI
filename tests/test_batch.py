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
        "priority",
        "issue_summary",
    ]
    for col in expected_cols:
        assert col in enriched_df.columns

    for cat in enriched_df["predicted_category"]:
        assert cat in APPROVED_CATEGORIES

    assert stats["total_processed"] == 5
    assert 0 <= stats["uncertain_count"] <= 5
    assert isinstance(stats["category_distribution"], dict)
