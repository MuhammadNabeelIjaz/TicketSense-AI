"""Input validation module for single tickets and batch CSV uploads."""

from typing import Tuple, List, Optional
import pandas as pd
from ticketsense.config import (
    MIN_TICKET_CHAR_LENGTH,
    MAX_TICKET_CHAR_LENGTH,
    APPROVED_CATEGORIES,
)


class ValidationError(Exception):
    """Custom exception raised for invalid input data."""
    pass


def validate_ticket_text(text: Optional[str]) -> Tuple[bool, str]:
    """Validate a single incoming support ticket message.

    Args:
        text: The raw customer support message.

    Returns:
        Tuple of (is_valid: bool, error_message: str).
    """
    if text is None:
        return False, "Ticket text is missing or null."

    if not isinstance(text, str):
        return False, f"Ticket text must be a string, got {type(text).__name__}."

    cleaned = text.strip()
    if not cleaned:
        return False, "Ticket text cannot be empty or solely whitespace."

    if len(cleaned) < MIN_TICKET_CHAR_LENGTH:
        return False, (
            f"Ticket text is too short ({len(cleaned)} chars). "
            f"Minimum required is {MIN_TICKET_CHAR_LENGTH} characters."
        )

    if len(cleaned) > MAX_TICKET_CHAR_LENGTH:
        return False, (
            f"Ticket text exceeds maximum allowed length "
            f"({len(cleaned)} > {MAX_TICKET_CHAR_LENGTH} chars)."
        )

    return True, ""


def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
    """Validate a batch dataframe uploaded via CSV.

    Accepts dataframes with 'ticket_text' or 'text' column.

    Args:
        df: Pandas DataFrame parsed from CSV upload.

    Returns:
        Tuple of (is_valid: bool, message: str, standardized_df: pd.DataFrame).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, "Input must be a valid pandas DataFrame.", pd.DataFrame()

    if df.empty:
        return False, "Uploaded CSV contains no rows.", pd.DataFrame()

    # Identify text column
    target_col = None
    if "ticket_text" in df.columns:
        target_col = "ticket_text"
    elif "text" in df.columns:
        target_col = "text"
    elif "message" in df.columns:
        target_col = "message"

    if not target_col:
        available_cols = ", ".join(df.columns.tolist())
        return False, (
            f"CSV must contain a 'ticket_text', 'text', or 'message' column. "
            f"Found columns: [{available_cols}]"
        ), pd.DataFrame()

    # Standardize column name
    standardized = df.copy()
    if target_col != "ticket_text":
        standardized["ticket_text"] = standardized[target_col]

    initial_count = len(standardized)

    # Filter out null / NaN entries
    standardized = standardized.dropna(subset=["ticket_text"]).copy()

    # Convert to stripped string
    standardized["ticket_text"] = standardized["ticket_text"].astype(str).str.strip()

    # Filter out empty strings and stringified representations of nulls
    null_string_values = {"", "nan", "none", "null", "undefined", "n/a"}
    is_not_null = ~standardized["ticket_text"].str.lower().isin(null_string_values)
    standardized = standardized[is_not_null].copy()

    # Filter out tickets below minimum character length
    is_valid_len = standardized["ticket_text"].str.len() >= MIN_TICKET_CHAR_LENGTH
    standardized = standardized[is_valid_len].copy()

    standardized = standardized.reset_index(drop=True)
    valid_count = len(standardized)
    dropped = initial_count - valid_count

    if valid_count == 0:
        return False, "All rows in the uploaded CSV contained empty, invalid, or too short text.", pd.DataFrame()

    msg = f"Successfully validated {valid_count} rows."
    if dropped > 0:
        msg += f" ({dropped} empty/invalid/short rows were filtered out)."

    return True, msg, standardized
