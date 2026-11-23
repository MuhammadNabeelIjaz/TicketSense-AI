"""Uncertainty quantification and human-in-the-loop review flagging."""

from typing import Dict, Any, List, Optional
import numpy as np
from ticketsense.config import DEFAULT_UNCERTAINTY_THRESHOLD, DEFAULT_UNCERTAINTY_MARGIN


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
