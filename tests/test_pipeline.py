"""Unit tests for inference pipeline, outputs, and calibration."""

import pytest
from ticketsense.config import APPROVED_CATEGORIES


def test_inference_engine_ready(inference_engine):
    assert inference_engine.is_ready() is True


def test_predict_ticket_structure(inference_engine):
    text = "I received an unauthorized charge on my account."
    result = inference_engine.predict_ticket(text)

    assert result["success"] is True
    assert result["predicted_category"] in APPROVED_CATEGORIES
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["all_probabilities"], dict)
    assert len(result["all_probabilities"]) == len(APPROVED_CATEGORIES)
    assert isinstance(result["is_uncertain"], bool)
    assert "priority" in result
    assert result["priority_status"] == "PAUSED_PENDING_GATE_G1"


def test_predict_ticket_probabilities_sum(inference_engine):
    text = "Where is my package? The tracking has stopped updating."
    result = inference_engine.predict_ticket(text)
    prob_sum = sum(result["all_probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01


def test_predict_ticket_invalid_input(inference_engine):
    result = inference_engine.predict_ticket("   ")
    assert result["success"] is False
    assert "error" in result
