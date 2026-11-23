"""Auditable manual correction logger.

Ensures customer support feedback is preserved in an append-only CSV file
without altering historical training datasets or raw data.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
from ticketsense.config import RUNTIME_CORRECTIONS_PATH, APPROVED_CATEGORIES


CORRECTIONS_HEADER = [
    "timestamp",
    "ticket_text",
    "predicted_category",
    "predicted_confidence",
    "corrected_category",
    "corrected_by",
    "review_notes",
]


def log_manual_correction(
    ticket_text: str,
    predicted_category: str,
    predicted_confidence: float,
    corrected_category: str,
    corrected_by: str = "support_agent",
    review_notes: str = "",
    file_path: Path = RUNTIME_CORRECTIONS_PATH,
) -> bool:
    """Log an agent correction to an append-only CSV file.

    Args:
        ticket_text: The ticket message text.
        predicted_category: Category predicted by the model.
        predicted_confidence: Probability score of the model prediction.
        corrected_category: The ground truth category selected by the agent.
        corrected_by: Identifier of the reviewer.
        review_notes: Optional notes explaining the correction.
        file_path: Target CSV file path.

    Returns:
        bool: True if logged successfully.
    """
    if corrected_category not in APPROVED_CATEGORIES:
        raise ValueError(
            f"Invalid corrected category '{corrected_category}'. "
            f"Must be one of {APPROVED_CATEGORIES}"
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticket_text": ticket_text.strip(),
        "predicted_category": predicted_category,
        "predicted_confidence": round(float(predicted_confidence), 4),
        "corrected_category": corrected_category,
        "corrected_by": corrected_by.strip() or "support_agent",
        "review_notes": review_notes.strip(),
    }

    df_record = pd.DataFrame([record])

    if not file_path.exists():
        df_record.to_csv(file_path, index=False, columns=CORRECTIONS_HEADER, encoding="utf-8")
    else:
        df_record.to_csv(file_path, mode="a", header=False, index=False, columns=CORRECTIONS_HEADER, encoding="utf-8")

    return True


def load_corrections_log(file_path: Path = RUNTIME_CORRECTIONS_PATH) -> pd.DataFrame:
    """Load the manual corrections log.

    Returns:
        pd.DataFrame of logged corrections, or empty DataFrame if none exist.
    """
    if not file_path.exists():
        return pd.DataFrame(columns=CORRECTIONS_HEADER)
    return pd.read_csv(file_path, encoding="utf-8")
