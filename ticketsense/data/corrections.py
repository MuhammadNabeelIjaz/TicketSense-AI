"""Auditable manual correction logger.

Ensures customer support feedback is preserved in an append-only CSV file
without altering historical training datasets or raw data.
Supports both Category and Priority operational corrections.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
from ticketsense.config import RUNTIME_CORRECTIONS_PATH, APPROVED_CATEGORIES, APPROVED_PRIORITIES

CORRECTIONS_HEADER = [
    "timestamp",
    "ticket_text",
    "predicted_category",
    "predicted_confidence",
    "corrected_category",
    "corrected_by",
    "review_notes",
    "predicted_priority",
    "predicted_priority_confidence",
    "corrected_priority",
]


def log_manual_correction(
    ticket_text: str,
    predicted_category: str,
    predicted_confidence: float,
    corrected_category: Optional[str] = None,
    corrected_by: str = "support_agent",
    review_notes: str = "",
    file_path: Path = RUNTIME_CORRECTIONS_PATH,
    predicted_priority: Optional[str] = None,
    predicted_priority_confidence: Optional[float] = None,
    corrected_priority: Optional[str] = None,
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
        predicted_priority: Priority predicted by the model (e.g. 'Low', 'High').
        predicted_priority_confidence: Priority prediction confidence score.
        corrected_priority: Ground truth priority selected by the agent.

    Returns:
        bool: True if logged successfully.
    """
    if corrected_category is None and corrected_priority is None:
        raise ValueError("Must provide either corrected_category or corrected_priority.")

    if corrected_category is not None and corrected_category not in APPROVED_CATEGORIES:
        raise ValueError(
            f"Invalid corrected category '{corrected_category}'. "
            f"Must be one of {APPROVED_CATEGORIES}"
        )

    if corrected_priority is not None and corrected_priority not in APPROVED_PRIORITIES:
        raise ValueError(
            f"Invalid corrected priority '{corrected_priority}'. "
            f"Must be one of {APPROVED_PRIORITIES}"
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticket_text": ticket_text.strip(),
        "predicted_category": predicted_category,
        "predicted_confidence": round(float(predicted_confidence), 4) if predicted_confidence is not None else None,
        "corrected_category": corrected_category if corrected_category is not None else predicted_category,
        "corrected_by": corrected_by.strip() or "support_agent",
        "review_notes": review_notes.strip(),
        "predicted_priority": predicted_priority,
        "predicted_priority_confidence": round(float(predicted_priority_confidence), 4) if predicted_priority_confidence is not None else None,
        "corrected_priority": corrected_priority,
    }

    df_new = pd.DataFrame([record], columns=CORRECTIONS_HEADER)

    if not file_path.exists():
        df_new.to_csv(file_path, index=False, columns=CORRECTIONS_HEADER, encoding="utf-8")
    else:
        existing_df = pd.read_csv(file_path, encoding="utf-8")
        # Ensure all columns from CORRECTIONS_HEADER exist
        for col in CORRECTIONS_HEADER:
            if col not in existing_df.columns:
                existing_df[col] = None
        combined_df = pd.concat([existing_df, df_new], ignore_index=True)
        combined_df.to_csv(file_path, index=False, columns=CORRECTIONS_HEADER, encoding="utf-8")

    return True


def load_corrections_log(file_path: Path = RUNTIME_CORRECTIONS_PATH) -> pd.DataFrame:
    """Load the manual corrections log.

    Returns:
        pd.DataFrame of logged corrections, or empty DataFrame if none exist.
    """
    if not file_path.exists():
        return pd.DataFrame(columns=CORRECTIONS_HEADER)
    df = pd.read_csv(file_path, encoding="utf-8")
    # Ensure all standard columns are present
    for col in CORRECTIONS_HEADER:
        if col not in df.columns:
            df[col] = None
    return df[CORRECTIONS_HEADER]
