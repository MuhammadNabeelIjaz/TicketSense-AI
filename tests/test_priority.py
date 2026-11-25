"""Unit tests for priority assessment heuristic, Two-Stage ML model, and governance policy."""

import pytest
import pandas as pd
from ticketsense.inference.priority_stub import assess_priority
from ticketsense.config import PRIORITY_POLICY_STATUS_MESSAGE


def test_priority_governance_metadata():
    """Verify governance tags reflect PAUSED status and untrained model state."""
    result = assess_priority("My account is locked out.")
    assert result["priority_status"] == "PAUSED_PENDING_GATE_G1"
    assert result["is_model_trained"] is False
    assert result["policy_message"] == PRIORITY_POLICY_STATUS_MESSAGE
    assert "heuristic_priority" in result


@pytest.mark.parametrize("text,expected_pattern", [
    ("There is an unauthorized charge on my account.", "unauthorized"),
    ("I am locked out of the system.", "locked"),
    ("Someone committed fraud on my card.", "fraud"),
    ("A fraudulent transaction was posted today.", "fraud"),
    ("My account was hacked yesterday.", "hacked"),
    ("My laptop was stolen with access keys.", "stolen"),
    ("We detected a data breach in staging.", "breach"),
    ("Security team confirmed breached credentials.", "breach"),
    ("We are experiencing a major system outage.", "outage"),
    ("The primary database down alert just fired.", "down"),
])
def test_priority_high_detection(text, expected_pattern):
    """Verify critical security and operational incident patterns trigger High."""
    result = assess_priority(text)
    assert result["heuristic_priority"] == "High"
    assert expected_pattern in result["heuristic_notes"]


@pytest.mark.parametrize("text", [
    "How to reset my user profile picture?",
    "Could you send more information on pricing?",
    "General inquiry regarding product capabilities.",
    "I have a quick question about business hours.",
])
def test_priority_low_detection(text):
    """Verify routine informational query patterns trigger Low."""
    result = assess_priority(text)
    assert result["heuristic_priority"] == "Low"


def test_priority_default_medium():
    """Verify messages without specific High or Low patterns default to Medium."""
    texts = [
        "My tracking number TS-9821 has no scan update.",
        "The export button throws error 500.",
        "Please update my billing street address.",
        "",
        "   ",
    ]
    for text in texts:
        result = assess_priority(text)
        assert result["heuristic_priority"] == "Medium"


def test_priority_word_boundary_isolation():
    """Verify word boundaries prevent accidental substring triggers."""
    # 'misinformation' should NOT trigger 'information' -> Low
    res1 = assess_priority("We are studying misinformation on the platform.")
    assert res1["heuristic_priority"] == "Medium"

    # 'generalization' should NOT trigger 'general' -> Low
    res2 = assess_priority("This is a broad generalization.")
    assert res2["heuristic_priority"] == "Medium"


def test_priority_precedence_high_over_low():
    """Verify High priority pattern takes precedence over Low pattern."""
    text = "I have a general question about a fraud alert."
    result = assess_priority(text)
    assert result["heuristic_priority"] == "High"


def test_priority_documented_negation_limitation():
    """Verify documented behavior that heuristic does not parse natural language negation."""
    # Under deterministic keyword rules, presence of 'fraud' triggers High
    result = assess_priority("No fraud detected on my card statement.")
    assert result["heuristic_priority"] == "High"


def test_priority_predictor_loads_and_ready():
    """Verify trained ML PriorityPredictor loads successfully from disk."""
    from ticketsense.inference.priority_predictor import PriorityPredictor
    predictor = PriorityPredictor()
    assert predictor.is_ready() is True
    assert set(predictor.classes) == {"High", "Medium", "Low"}


def test_priority_predictor_ml_inference():
    """Verify ML inference outputs expected structure, probabilities, and status."""
    from ticketsense.inference.priority_predictor import PriorityPredictor
    predictor = PriorityPredictor()
    res = predictor.predict("Our database is down and users cannot log in.")

    assert res["is_model_trained"] is True
    assert res["priority_status"] == "ACTIVE_ML_MODEL"
    assert res["priority"] in ["High", "Medium", "Low"]
    assert 0.0 <= res["priority_confidence"] <= 1.0
    assert isinstance(res["priority_probabilities"], dict)
    assert len(res["priority_probabilities"]) == 3
    assert abs(sum(res["priority_probabilities"].values()) - 1.0) < 0.02
    assert res["predicted_impact"] in ["I1", "I2", "I3"]
    assert res["predicted_urgency"] in ["U1", "U2", "U3"]
    assert "priority_review_required" in res
    assert isinstance(res["priority_review_required"], bool)
    assert "priority_uncertainty_reason" in res
    assert "priority_margin" in res


def test_priority_predictor_critical_outage_handling():
    """Verify PriorityPredictor handles critical outage safely."""
    from ticketsense.inference.priority_predictor import PriorityPredictor
    predictor = PriorityPredictor()
    res = predictor.predict("Emergency: our primary database cluster crashed and customer checkout is completely down nationwide.")
    assert res["priority_review_required"] is True
    assert "priority_uncertainty_reason" in res
    assert len(res["priority_uncertainty_reason"]) > 0
    assert "priority_review_details" in res
    if res["priority"] == "Low":
        assert res["priority_review_details"]["is_critical_keyword_discrepancy"] is True
    else:
        assert res["priority"] == "High"


def test_priority_predictor_fallback_on_invalid():
    """Verify predictor handles empty and non-string inputs safely."""
    from ticketsense.inference.priority_predictor import PriorityPredictor
    predictor = PriorityPredictor()

    res_empty = predictor.predict("")
    assert res_empty["priority"] == "Medium"
    assert res_empty["priority_review_required"] is True
    assert res_empty["priority_status"] == "FALLBACK_HEURISTIC"

    res_none = predictor.predict(None)
    assert res_none["priority"] == "Medium"
    assert res_none["priority_review_required"] is True
    assert res_none["priority_status"] == "FALLBACK_HEURISTIC"


def test_priority_predictor_missing_model_fallback_safety(tmp_path):
    """Verify missing model file triggers safe fallback and human review routing."""
    from ticketsense.inference.priority_predictor import PriorityPredictor
    predictor = PriorityPredictor(model_path=tmp_path / "non_existent_model.joblib")
    assert predictor.is_ready() is False

    res = predictor.predict("Emergency: database crashed and checkout is failing.")
    assert res["priority_status"] == "FALLBACK_HEURISTIC"
    assert res["priority_review_required"] is True
    assert res["priority_confidence"] is None
    assert res["is_model_trained"] is False


def test_category_model_regression_protection(inference_engine):
    """Verify Category Classification predictions remain 100% stable."""
    res = inference_engine.predict_ticket("Where is my delayed delivery package?")
    assert res["predicted_category"] == "Delivery"
    assert res["confidence"] >= 0.70
    assert "priority" in res
    assert res["priority_status"] == "ACTIVE_ML_MODEL"


def test_rubric_mapping_matrix_consistency():
    """Verify all 9 Impact x Urgency combinations match the project rubric."""
    from ticketsense.models.priority_pipeline import AUTHORITATIVE_RUBRIC
    assert len(AUTHORITATIVE_RUBRIC) == 9
    assert AUTHORITATIVE_RUBRIC[("I3", "U3")] == "High"
    assert AUTHORITATIVE_RUBRIC[("I1", "U1")] == "Low"
    assert AUTHORITATIVE_RUBRIC[("I2", "U2")] == "Medium"
    assert AUTHORITATIVE_RUBRIC[("I3", "U2")] == "High"
    assert AUTHORITATIVE_RUBRIC[("I2", "U3")] == "High"
    assert AUTHORITATIVE_RUBRIC[("I1", "U3")] == "Medium"
    assert AUTHORITATIVE_RUBRIC[("I3", "U1")] == "Medium"
    assert AUTHORITATIVE_RUBRIC[("I2", "U1")] == "Low"
    assert AUTHORITATIVE_RUBRIC[("I1", "U2")] == "Low"


def test_two_stage_classifier_detailed_diagnostics():
    """Verify detailed diagnostics output impact, urgency, and confidence scores."""
    from ticketsense.inference.priority_predictor import PriorityPredictor
    predictor = PriorityPredictor()
    res = predictor.predict("Critical server failure affecting entire transaction processing pipeline.")
    assert "predicted_impact" in res
    assert "predicted_urgency" in res
    assert res["predicted_impact"] in ["I1", "I2", "I3"]
    assert res["predicted_urgency"] in ["U1", "U2", "U3"]
    assert "priority_review_required" in res
    assert isinstance(res["priority_review_required"], bool)
    assert "priority_uncertainty_reason" in res
    assert "priority_margin" in res


def test_priority_predictor_diagnostics_confidences():
    """Verify PriorityPredictor includes component confidences."""
    from ticketsense.inference.priority_predictor import PriorityPredictor
    predictor = PriorityPredictor()
    res = predictor.predict("Emergency: our primary database cluster crashed and customer checkout is completely down nationwide.")
    assert res["priority_review_required"] is True
    assert "priority_uncertainty_reason" in res
    assert len(res["priority_uncertainty_reason"]) > 0
    assert "priority_review_details" in res
    if res["priority"] == "Low":
        assert res["priority_review_details"]["is_critical_keyword_discrepancy"] is True
    else:
        assert res["priority"] == "High"
    assert res["impact_confidence"] is not None
    assert res["urgency_confidence"] is not None


def test_inference_engine_batch_two_stage_columns(inference_engine):
    """Verify batch inference adds impact, urgency, and priority columns."""
    sample_df = pd.DataFrame({
        "ticket_text": [
            "Please cancel my order and refund my credit card.",
            "Can I change my account email address?",
        ]
    })
    success, msg, enriched_df, stats = inference_engine.predict_batch(sample_df)
    assert success is True
    assert "predicted_category" in enriched_df.columns
    assert "predicted_priority" in enriched_df.columns
    assert "predicted_impact" in enriched_df.columns
    assert "predicted_urgency" in enriched_df.columns
    assert len(enriched_df) == 2
