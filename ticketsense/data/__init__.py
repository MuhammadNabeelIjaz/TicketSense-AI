"""Data processing, validation, and corrections management for TicketSense-AI."""

from ticketsense.data.validator import validate_ticket_text, validate_batch_dataframe
from ticketsense.data.loader import load_dataset_splits
from ticketsense.data.corrections import log_manual_correction, load_corrections_log

__all__ = [
    "validate_ticket_text",
    "validate_batch_dataframe",
    "load_dataset_splits",
    "log_manual_correction",
    "load_corrections_log",
]
