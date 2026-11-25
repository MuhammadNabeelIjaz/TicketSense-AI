"""Uncertainty quantification and human-in-the-loop review flagging."""

import re
from typing import Dict, Any, List, Optional, Union
import numpy as np
from ticketsense.config import (
    DEFAULT_UNCERTAINTY_THRESHOLD,
    DEFAULT_UNCERTAINTY_MARGIN,
    DEFAULT_PRIORITY_UNCERTAINTY_THRESHOLD,
    DEFAULT_PRIORITY_UNCERTAINTY_MARGIN,
    DEFAULT_PRIORITY_COMPONENT_THRESHOLD,
    DEFAULT_PRIORITY_HIGH_RISK_THRESHOLD,
)

CRITICAL_INCIDENT_PATTERN = re.compile(
    r"\b(emergency|outage|crashed?|critical|nationwide|escalat\w+|immediate\w*)\b",
    re.IGNORECASE,
)


def compute_uncertainty_score(
    probabilities: List[float],
    labels: List[str],
    threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> Dict[str, Any]:
    """Evaluate prediction confidence distribution to flag uncertain cases for human review.

    Flagging criteria:
        1. Low Confidence: Top probability < threshold (default 0.65).
        2. Ambiguity: Difference between top-1 and top-2 probabilities < margin (default 0.15).

    Args:
        probabilities: Probability distribution across all classes.
        labels: Corresponding class labels.
        threshold: Minimum confidence required for autonomous acceptance.
        margin: Minimum margin between top two classes.

    Returns:
        Dict containing flags, top predictions, and decision reasoning.
    """
    probs = np.array(probabilities, dtype=float)
    if len(probs) != len(labels):
        raise ValueError("Probabilities and labels must have the same length.")

    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = sorted_indices[0]
    top1_prob = float(probs[top1_idx])
    top1_label = labels[top1_idx]

    top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
    top2_prob = float(probs[top2_idx]) if top2_idx is not None else 0.0
    top2_label = labels[top2_idx] if top2_idx is not None else None

    diff = float(top1_prob - top2_prob)

    is_low_confidence = top1_prob < threshold
    is_ambiguous = diff < margin

    is_uncertain = is_low_confidence or is_ambiguous

    reasons = []
    if is_low_confidence:
        reasons.append(
            f"Low confidence ({top1_prob:.1%} < {threshold:.1%})"
        )
    if is_ambiguous and top2_label:
        reasons.append(
            f"Ambiguous between '{top1_label}' ({top1_prob:.1%}) and '{top2_label}' ({top2_prob:.1%}), margin {diff:.1%} < {margin:.1%}"
        )

    reason_str = "; ".join(reasons) if reasons else "Prediction meets autonomous confidence thresholds."

    return {
        "is_uncertain": bool(is_uncertain),
        "is_low_confidence": bool(is_low_confidence),
        "is_ambiguous": bool(is_ambiguous),
        "top_category": top1_label,
        "confidence": round(top1_prob, 4),
        "second_category": top2_label,
        "second_confidence": round(top2_prob, 4),
        "margin": round(diff, 4),
        "reason": reason_str,
    }


def compute_priority_uncertainty_score(
    priority: str,
    priority_probabilities: Optional[Dict[str, float]] = None,
    predicted_impact: Optional[str] = None,
    impact_confidence: Optional[float] = None,
    predicted_urgency: Optional[str] = None,
    urgency_confidence: Optional[float] = None,
    ticket_text: Optional[str] = None,
    threshold: float = DEFAULT_PRIORITY_UNCERTAINTY_THRESHOLD,
    margin: float = DEFAULT_PRIORITY_UNCERTAINTY_MARGIN,
    component_threshold: float = DEFAULT_PRIORITY_COMPONENT_THRESHOLD,
    high_risk_threshold: float = DEFAULT_PRIORITY_HIGH_RISK_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate Priority prediction confidence and component distributions to flag uncertain cases.

    Flagging criteria:
        1. Low Confidence: Top Priority joint probability < threshold (default 0.60).
        2. Small Margin: Difference between top-1 and top-2 Priority probabilities < margin (default 0.15).
        3. Component Uncertainty: Impact confidence < component_threshold (0.50) OR
           Urgency confidence < component_threshold (0.50).
        4. High-Risk Discrepancy:
           a. Predicted Priority is 'Low' but P(High) >= high_risk_threshold (default 0.15).
           b. Predicted Priority is 'Low' but message contains critical incident vocabulary
              (e.g., 'emergency', 'outage', 'crashed', 'nationwide', 'escalation').

    Args:
        priority: Predicted priority label ('High', 'Medium', 'Low').
        priority_probabilities: Dict mapping class names to joint probabilities.
        predicted_impact: Predicted impact code ('I1', 'I2', 'I3').
        impact_confidence: Confidence of impact stage prediction.
        predicted_urgency: Predicted urgency code ('U1', 'U2', 'U3').
        urgency_confidence: Confidence of urgency stage prediction.
        ticket_text: Original customer support ticket message.
        threshold: Minimum joint priority probability required for autonomous acceptance.
        margin: Minimum margin between top-1 and top-2 priority classes.
        component_threshold: Minimum confidence required for Impact and Urgency stages.
        high_risk_threshold: Maximum allowable P(High) before flagging a 'Low' prediction.

    Returns:
        Dict containing review flag, individual criteria flags, and decision reasoning.
    """
    reasons = []

    # Safe validation of probability inputs
    if not priority_probabilities or not isinstance(priority_probabilities, dict):
        return {
            "priority_review_required": True,
            "is_low_confidence": True,
            "is_ambiguous_margin": False,
            "is_component_uncertain": False,
            "is_high_risk_low": False,
            "is_critical_keyword_discrepancy": False,
            "priority": str(priority),
            "priority_confidence": None,
            "second_priority": None,
            "second_priority_confidence": None,
            "priority_margin": None,
            "reason": "Missing or invalid Priority probability distribution; routed to human review.",
        }

    # Validate numeric values
    prob_items = []
    for k, v in priority_probabilities.items():
        try:
            val = float(v)
            if np.isnan(val) or val < 0.0 or val > 1.0:
                reasons.append(f"Invalid probability value {val} for '{k}'")
            prob_items.append((str(k), val))
        except (TypeError, ValueError):
            reasons.append(f"Non-numeric probability for '{k}': {v}")

    if reasons:
        return {
            "priority_review_required": True,
            "is_low_confidence": True,
            "is_ambiguous_margin": False,
            "is_component_uncertain": False,
            "is_high_risk_low": False,
            "is_critical_keyword_discrepancy": False,
            "priority": str(priority),
            "priority_confidence": None,
            "second_priority": None,
            "second_priority_confidence": None,
            "priority_margin": None,
            "reason": "; ".join(reasons),
        }

    # Sort probabilities descending
    sorted_probs = sorted(prob_items, key=lambda x: x[1], reverse=True)
    top1_label, top1_prob = sorted_probs[0]
    top2_label, top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0.0)
    diff = float(top1_prob - top2_prob)

    is_low_confidence = top1_prob < threshold
    is_ambiguous_margin = diff < margin

    # Component checks
    is_impact_uncertain = (impact_confidence is not None) and (float(impact_confidence) < component_threshold)
    is_urgency_uncertain = (urgency_confidence is not None) and (float(urgency_confidence) < component_threshold)
    is_component_uncertain = is_impact_uncertain or is_urgency_uncertain

    # High-risk checks
    p_high = float(priority_probabilities.get("High", 0.0))
    is_high_risk_low = (priority == "Low") and (p_high >= high_risk_threshold)

    # Critical incident vocabulary check when predicted as Low
    is_critical_keyword_discrepancy = False
    matched_incident_keywords = []
    if priority == "Low" and ticket_text and isinstance(ticket_text, str):
        matches = CRITICAL_INCIDENT_PATTERN.findall(ticket_text)
        if matches:
            is_critical_keyword_discrepancy = True
            matched_incident_keywords = sorted(list(set(m.lower() for m in matches)))

    # Compile explicit reasons
    if is_low_confidence:
        reasons.append(f"Low Priority confidence ({top1_prob:.1%} < {threshold:.1%})")
    if is_ambiguous_margin and top2_label:
        reasons.append(
            f"Ambiguous Priority margin between '{top1_label}' ({top1_prob:.1%}) and '{top2_label}' ({top2_prob:.1%}), margin {diff:.1%} < {margin:.1%}"
        )
    if is_impact_uncertain:
        reasons.append(f"Uncertain Impact component (Impact {predicted_impact} confidence {float(impact_confidence):.1%} < {component_threshold:.1%})")
    if is_urgency_uncertain:
        reasons.append(f"Uncertain Urgency component (Urgency {predicted_urgency} confidence {float(urgency_confidence):.1%} < {component_threshold:.1%})")
    if is_high_risk_low:
        reasons.append(f"Elevated High-priority probability ({p_high:.1%} >= {high_risk_threshold:.1%}) while predicted as Low Priority")
    if is_critical_keyword_discrepancy:
        reasons.append(f"Critical incident vocabulary detected ({', '.join(repr(w) for w in matched_incident_keywords)}) while predicted as Low Priority")

    priority_review_required = (
        is_low_confidence
        or is_ambiguous_margin
        or is_component_uncertain
        or is_high_risk_low
        or is_critical_keyword_discrepancy
    )

    reason_str = "; ".join(reasons) if reasons else "Priority meets autonomous confidence thresholds."

    return {
        "priority_review_required": bool(priority_review_required),
        "is_low_confidence": bool(is_low_confidence),
        "is_ambiguous_margin": bool(is_ambiguous_margin),
        "is_component_uncertain": bool(is_component_uncertain),
        "is_high_risk_low": bool(is_high_risk_low),
        "is_critical_keyword_discrepancy": bool(is_critical_keyword_discrepancy),
        "priority": str(priority),
        "priority_confidence": round(top1_prob, 4),
        "second_priority": top2_label,
        "second_priority_confidence": round(top2_prob, 4),
        "priority_margin": round(diff, 4),
        "reason": reason_str,
    }
