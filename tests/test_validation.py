"""Unit tests for input and batch validation."""

import pytest
import pandas as pd
from ticketsense.data.validator import validate_ticket_text, validate_batch_dataframe


def test_validate_ticket_text_valid():
    is_valid, msg = validate_ticket_text("My order has not arrived yet.")
    assert is_valid is True
    assert msg == ""


def test_validate_ticket_text_none():
    is_valid, msg = validate_ticket_text(None)
    assert is_valid is False
    assert "missing or null" in msg.lower()


def test_validate_ticket_text_empty():
    is_valid, msg = validate_ticket_text("   \n\t  ")
    assert is_valid is False
    assert "empty" in msg.lower()


def test_validate_ticket_text_too_short():
    is_valid, msg = validate_ticket_text("Hi")
    assert is_valid is False
    assert "too short" in msg.lower()


def test_validate_ticket_text_too_long():
    long_text = "a" * 5005
    is_valid, msg = validate_ticket_text(long_text)
    assert is_valid is False
    assert "exceeds maximum" in msg.lower()


def test_validate_batch_dataframe_valid(sample_batch_df):
    is_valid, msg, std_df = validate_batch_dataframe(sample_batch_df)
    assert is_valid is True
    assert len(std_df) == 5
    assert "ticket_text" in std_df.columns


def test_validate_batch_dataframe_alternative_column():
    df = pd.DataFrame({"message": ["Valid ticket text for classification."]})
    is_valid, msg, std_df = validate_batch_dataframe(df)
    assert is_valid is True
    assert "ticket_text" in std_df.columns
    assert len(std_df) == 1


def test_validate_batch_dataframe_empty():
    is_valid, msg, _ = validate_batch_dataframe(pd.DataFrame())
    assert is_valid is False
    assert "no rows" in msg.lower()


def test_validate_batch_dataframe_missing_column():
    df = pd.DataFrame({"wrong_col": [1, 2, 3]})
    is_valid, msg, _ = validate_batch_dataframe(df)
    assert is_valid is False
    assert "must contain" in msg.lower()
