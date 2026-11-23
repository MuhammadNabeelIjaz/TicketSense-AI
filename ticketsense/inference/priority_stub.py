"""Priority assessment architecture and policy enforcement stub.

Per project policy guidelines, priority model training is strictly PAUSED
because provisional priority labels have severe class imbalance (6.5% High)
and are pending supervisor Gate G1 approval.
"""

from typing import Dict, Any
from ticketsense.config import PRIORITY_TRAINING_PAUSED, PRIORITY_POLICY_STATUS_MESSAGE


def assess_priority(ticket_text: str) -> Dict[str, Any]:
    """Provide priority assessment status and rule-based heuristic indicator.

    Strict guardrail: Returns explicit notice of paused model training.

    Args:
        ticket_text: Customer message.

    Returns:
        Dict detailing priority policy status and heuristic guidance.
    """
    if not isinstance(ticket_text, str):
        ticket_text = str(ticket_text) if ticket_text is not None else ""

    # Heuristic indicator based on approved high-impact keywords
    lower = ticket_text.lower()
    heuristic = "Medium"
    heuristic_notes = "Default provisional rating."

    if any(k in lower for k in ["unauthorized", "locked out", "fraud", "hacked", "stolen", "breach"]):
        heuristic = "High"
        heuristic_notes = "Contains security/unauthorized access keywords (heuristic only)."
    elif any(k in lower for k in ["how to", "information", "inquiry", "question", "general"]):
        heuristic = "Low"
        heuristic_notes = "Informational/general query phrasing (heuristic only)."

    return {
        "priority_status": "PAUSED_PENDING_GATE_G1",
        "heuristic_priority": heuristic,
        "is_model_trained": False,
        "policy_message": PRIORITY_POLICY_STATUS_MESSAGE,
        "heuristic_notes": heuristic_notes,
    }
