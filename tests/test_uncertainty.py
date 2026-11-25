"""Unit tests for uncertainty calculation and human review flagging."""

import pytest
import pandas as pd
from ticketsense.inference.uncertainty import (
    compute_uncertainty_score,
    compute_priority_uncertainty_score,
)


def test_uncertainty_low_confidence():
    labels = ["A", "B", "C"]
    probs = [0.40, 0.35, 0.25]
    diag = compute_uncertainty_score(probs, labels, threshold=0.65, margin=0.15)

    assert diag["is_uncertain"] is True
    assert diag["is_low_confidence"] is True
    assert "Low confidence" in diag["reason"]


def test_uncertainty_ambiguous_margin():
    labels = ["A", "B", "C"]
    probs = [0.68, 0.65, 0.10]
    # top prob is 0.68 > 0.65, but margin is 0.03 < 0.15
    diag = compute_uncertainty_score(probs, labels, threshold=0.65, margin=0.15)

    assert diag["is_uncertain"] is True
    assert diag["is_ambiguous"] is True
    assert "Ambiguous" in diag["reason"]


def test_uncertainty_confident_autonomous():
    labels = ["A", "B", "C"]
    probs = [0.90, 0.06, 0.04]
    diag = compute_uncertainty_score(probs, labels, threshold=0.65, margin=0.15)

    assert diag["is_uncertain"] is False
    assert diag["is_low_confidence"] is False
    assert diag["is_ambiguous"] is False


def test_priority_uncertainty_low_priority_with_uncertain_components():
    """Verify Low Priority prediction with uncertain impact/urgency component is flagged."""
    diag = compute_priority_uncertainty_score(
        priority="Low",
        priority_probabilities={"Low": 0.68, "Medium": 0.22, "High": 0.10},
        predicted_impact="I2",
        impact_confidence=0.45,  # below 0.50 threshold
        predicted_urgency="U1",
        urgency_confidence=0.62,
        ticket_text="My report page refreshed unexpectedly.",
    )
    assert diag["priority_review_required"] is True
    assert diag["is_component_uncertain"] is True
    assert "Uncertain Impact component" in diag["reason"]


def test_priority_uncertainty_high_priority_with_strong_confidence():
    """Verify High Priority prediction with strong confidence and margin is autonomous."""
    diag = compute_priority_uncertainty_score(
        priority="High",
        priority_probabilities={"High": 0.85, "Medium": 0.10, "Low": 0.05},
        predicted_impact="I3",
        impact_confidence=0.88,
        predicted_urgency="U3",
        urgency_confidence=0.82,
        ticket_text="Account suspended and credit card charged repeatedly.",
    )
    assert diag["priority_review_required"] is False
    assert diag["is_low_confidence"] is False
    assert diag["is_ambiguous_margin"] is False
    assert "autonomous" in diag["reason"].lower()


def test_priority_uncertainty_low_priority_with_strong_confidence():
    """Verify clear Low Priority query without risk factors is accepted autonomously."""
    diag = compute_priority_uncertainty_score(
        priority="Low",
        priority_probabilities={"Low": 0.82, "Medium": 0.14, "High": 0.04},
        predicted_impact="I1",
        impact_confidence=0.80,
        predicted_urgency="U1",
        urgency_confidence=0.85,
        ticket_text="How can I change my profile avatar image?",
    )
    assert diag["priority_review_required"] is False
    assert diag["is_low_confidence"] is False
    assert diag["is_ambiguous_margin"] is False
    assert diag["is_high_risk_low"] is False


def test_priority_uncertainty_small_probability_margin():
    """Verify ambiguous small probability margin between top two priority classes flags review."""
    diag = compute_priority_uncertainty_score(
        priority="Medium",
        priority_probabilities={"Medium": 0.48, "Low": 0.45, "High": 0.07},
        predicted_impact="I2",
        impact_confidence=0.60,
        predicted_urgency="U2",
        urgency_confidence=0.60,
        ticket_text="Shipping address update requested for pending dispatch.",
    )
    assert diag["priority_review_required"] is True
    assert diag["is_ambiguous_margin"] is True
    assert "Ambiguous Priority margin" in diag["reason"]


def test_priority_uncertainty_invalid_probability_values():
    """Verify invalid or non-numeric probability values flag review safely."""
    diag = compute_priority_uncertainty_score(
        priority="Low",
        priority_probabilities={"Low": -0.5, "Medium": 1.2, "High": 0.3},
        ticket_text="Test message.",
    )
    assert diag["priority_review_required"] is True
    assert "Invalid probability value" in diag["reason"]


def test_priority_uncertainty_missing_model_outputs():
    """Verify missing or empty model probabilities flag human review safely."""
    diag1 = compute_priority_uncertainty_score(priority="Low", priority_probabilities=None)
    assert diag1["priority_review_required"] is True
    assert "Missing or invalid Priority probability" in diag1["reason"]

    diag2 = compute_priority_uncertainty_score(priority="Low", priority_probabilities={})
    assert diag2["priority_review_required"] is True
    assert "Missing or invalid Priority probability" in diag2["reason"]


def test_priority_uncertainty_critical_outage_discrepancy():
    """Verify enterprise outage ticket predicted Low is flagged by critical incident guardrail."""
    outage_text = "Emergency: our primary database cluster crashed and customer checkout is completely down nationwide. Immediate escalation needed."
    diag = compute_priority_uncertainty_score(
        priority="Low",
        priority_probabilities={"Low": 0.6395, "Medium": 0.2351, "High": 0.1254},
        predicted_impact="I2",
        impact_confidence=0.6312,
        predicted_urgency="U1",
        urgency_confidence=0.6637,
        ticket_text=outage_text,
    )
    assert diag["priority_review_required"] is True
    assert diag["is_critical_keyword_discrepancy"] is True
    assert "Critical incident vocabulary detected" in diag["reason"]


def test_category_review_independent_of_priority_review(inference_engine):
    """Verify Category review and Priority review decisions operate independently."""
    # Scenario 1: Ambiguous category but standard low priority
    res1 = inference_engine.predict_ticket("My account was billed twice but I also cannot log in to download my invoice.")
    assert res1["category_review_required"] is True
    # Overall review is required whenever either component flags review
    assert res1["overall_review_required"] is True

    # Scenario 2: Clear technical category but enterprise emergency text flagging priority review
    outage_text = "Emergency: our primary database cluster crashed and customer checkout is completely down nationwide. Immediate escalation needed."
    res2 = inference_engine.predict_ticket(outage_text)
    assert res2["predicted_category"] == "Technical Issue"
    assert res2["category_review_required"] is False
    assert res2["priority_review_required"] is True
    assert res2["overall_review_required"] is True


def test_batch_review_flags(inference_engine, sample_batch_df):
    """Verify batch processing produces independent category and priority review columns."""
    success, msg, enriched_df, stats = inference_engine.predict_batch(sample_batch_df)
    assert success is True
    assert "category_review_required" in enriched_df.columns
    assert "priority_review_required" in enriched_df.columns
    assert "overall_review_required" in enriched_df.columns
    assert "priority_uncertainty_reason" in enriched_df.columns

    assert "category_review_count" in stats
    assert "priority_review_count" in stats
    assert "overall_review_count" in stats
    assert "autonomous_acceptance_count" in stats
    assert stats["overall_review_count"] == stats["uncertain_count"]
