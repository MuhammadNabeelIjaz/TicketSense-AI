"""Priority assessment architecture and policy enforcement stub.

Per project policy guidelines, priority model training is strictly PAUSED
because provisional priority labels have severe class imbalance (6.5% High)
and are pending supervisor Gate G1 approval.

Operational triage is provided via a conservative, rule-based deterministic heuristic.
Limitations:
- Uses regex word-boundary matching to prevent substring collisions.
- Does NOT perform deep semantic understanding or natural language negation parsing.
  (Phrases like "no fraud detected" will trigger High due to the presence of "fraud".)
- Full context understanding requires supervised machine learning with verified operational labels,
  currently paused under Gate G1.
"""

import re
from typing import Dict, Any
from ticketsense.config import PRIORITY_TRAINING_PAUSED, PRIORITY_POLICY_STATUS_MESSAGE

# High-priority patterns: critical security, credential compromise, unauthorized activity, or infrastructure failure
HIGH_PRIORITY_PATTERNS = [
    r"\bunauthorized\b",
    r"\blocked\s+out\b",
    r"\bfraud(?:ulent)?\b",
    r"\bhacked\b",
    r"\bstolen\b",
    r"\bbreach(?:ed)?\b",
    r"\b(?:system|service|server|database)\s+outage\b",
    r"\b(?:server|database)\s+down\b",
]

# Low-priority patterns: routine informational queries without time-sensitive operational impact
LOW_PRIORITY_PATTERNS = [
    r"\bhow\s+to\b",
    r"\binformation\b",
    r"\binquiry\b",
    r"\bquestion\b",
    r"\bgeneral\b",
]

COMPILED_HIGH_RE = [re.compile(p, re.IGNORECASE) for p in HIGH_PRIORITY_PATTERNS]
COMPILED_LOW_RE = [re.compile(p, re.IGNORECASE) for p in LOW_PRIORITY_PATTERNS]


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

    text = ticket_text.strip()
    heuristic = "Medium"
    heuristic_notes = "Default provisional rating."

    # High priority check (takes precedence)
    matched_high = [p.pattern for p in COMPILED_HIGH_RE if p.search(text)]
    if matched_high:
        heuristic = "High"
        heuristic_notes = f"Contains high-impact indicator (heuristic pattern: {matched_high[0]})."
    else:
        # Low priority check (only if not high)
        matched_low = [p.pattern for p in COMPILED_LOW_RE if p.search(text)]
        if matched_low:
            heuristic = "Low"
            heuristic_notes = f"Informational/general query phrasing (heuristic pattern: {matched_low[0]})."

    return {
        "priority_status": "PAUSED_PENDING_GATE_G1",
        "heuristic_priority": heuristic,
        "is_model_trained": False,
        "policy_message": PRIORITY_POLICY_STATUS_MESSAGE,
        "heuristic_notes": heuristic_notes,
    }
