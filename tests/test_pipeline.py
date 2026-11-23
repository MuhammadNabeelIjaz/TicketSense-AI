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


def test_pipeline_named_steps(inference_engine):
    assert "cleaner" in inference_engine.pipeline.named_steps
    assert "tfidf" in inference_engine.pipeline.named_steps
    assert "clf" in inference_engine.pipeline.named_steps


def test_pipeline_raw_text_direct_inference(inference_engine):
    raw_texts = [
        "I can't access my portal! It's completely broken.",
        "Where's my order? It hasn't arrived.",
    ]
    preds = inference_engine.pipeline.predict(raw_texts)
    probas = inference_engine.pipeline.predict_proba(raw_texts)

    assert len(preds) == 2
    assert probas.shape == (2, len(APPROVED_CATEGORIES))
    for p in probas:
        assert abs(sum(p) - 1.0) < 0.01
