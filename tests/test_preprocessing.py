"""Unit tests for NLP text cleaning and normalization."""

import pytest
from ticketsense.nlp.preprocessor import clean_text, TextCleaner


def test_clean_text_contractions():
    raw = "I can't log in and won't be able to pay if it's not fixed."
    cleaned = clean_text(raw)
    assert "cannot" in cleaned
    assert "will not" in cleaned
    assert "it is" in cleaned
    assert "can't" not in cleaned


def test_clean_text_urls_and_emails():
    raw = "Visit https://help.example.com or email support@example.com immediately."
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "support@" not in cleaned


def test_clean_text_special_characters():
    raw = "Refund $$$ required! Error: #404-crash &*^%"
    cleaned = clean_text(raw)
    assert "$" not in cleaned
    assert "#" not in cleaned
    assert "&" not in cleaned
    assert "refund" in cleaned
    assert "error" in cleaned


def test_clean_text_whitespace():
    raw = "   Multiple    spaces    and \n newlines \t\t here.  "
    cleaned = clean_text(raw)
    assert cleaned == "multiple spaces and newlines here"


def test_text_cleaner_transformer():
    cleaner = TextCleaner()
    inputs = ["I can't log in", "Contact me@test.com"]
    transformed = cleaner.transform(inputs)
    assert len(transformed) == 2
    assert "cannot" in transformed[0]
    assert "me@" not in transformed[1]


def test_clean_text_non_string_types():
    assert clean_text(12345) == "12345"
    assert clean_text(99.99) == "99 99"
    assert clean_text(None) == ""
