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


def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame, pd.DataFrame]:
    """Validate a batch dataframe uploaded via CSV.

    Identifies valid rows and tracks rejected rows with original row numbers and reasons.
    Accepts dataframes with 'ticket_text', 'text', or 'message' column.

    Args:
        df: Pandas DataFrame parsed from CSV upload.

    Returns:
        Tuple of (is_valid: bool, message: str, valid_df: pd.DataFrame, rejected_df: pd.DataFrame).
    """
    empty_rejected = pd.DataFrame(columns=["original_row_number", "raw_input", "rejection_reason"])

    if df is None or not isinstance(df, pd.DataFrame):
        return False, "Input must be a valid pandas DataFrame.", pd.DataFrame(), empty_rejected

    if df.empty:
        return False, "Uploaded CSV contains no rows.", pd.DataFrame(), empty_rejected

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
        ), pd.DataFrame(), empty_rejected

    null_string_values = {"", "nan", "none", "null", "undefined", "n/a"}

    valid_rows = []
    rejected_rows = []

    for idx, row in df.iterrows():
        row_num = idx + 1 if isinstance(idx, int) else len(valid_rows) + len(rejected_rows) + 1
        raw_val = row.get(target_col)

        # Check for null / NaN
        if raw_val is None or pd.isna(raw_val):
            rejected_rows.append({
                "original_row_number": row_num,
                "raw_input": str(raw_val),
                "rejection_reason": "Missing or null text entry.",
            })
            continue

        text_str = str(raw_val).strip()

        # Check for stringified nulls or empty
        if text_str.lower() in null_string_values:
            rejected_rows.append({
                "original_row_number": row_num,
                "raw_input": text_str,
                "rejection_reason": "Empty or stringified null value.",
            })
            continue

        # Check minimum length
        if len(text_str) < MIN_TICKET_CHAR_LENGTH:
            rejected_rows.append({
                "original_row_number": row_num,
                "raw_input": text_str,
                "rejection_reason": f"Text too short ({len(text_str)} chars; minimum required is {MIN_TICKET_CHAR_LENGTH}).",
            })
            continue

        # Valid row
        row_dict = row.to_dict()
        row_dict["original_row_number"] = row_num
        row_dict["ticket_text"] = text_str
        valid_rows.append(row_dict)

    valid_df = pd.DataFrame(valid_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    if rejected_df.empty:
        rejected_df = empty_rejected

    valid_count = len(valid_df)
    rejected_count = len(rejected_df)

    if valid_count == 0:
        return (
            False,
            f"All {rejected_count} rows in the uploaded CSV contained empty, invalid, or too short text.",
            pd.DataFrame(),
            rejected_df,
        )

    msg = f"Successfully validated {valid_count} rows."
    if rejected_count > 0:
        msg += f" ({rejected_count} invalid rows were flagged and isolated)."

    return True, msg, valid_df, rejected_df
