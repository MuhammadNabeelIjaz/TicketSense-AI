"""Shared pytest fixtures for TicketSense-AI test suite."""

import pytest
import pandas as pd
from pathlib import Path
from ticketsense.inference.engine import InferenceEngine


@pytest.fixture(scope="session")
def inference_engine():
    """Session-scoped loaded inference engine."""
    engine = InferenceEngine()
    return engine


@pytest.fixture
def sample_valid_tickets():
    """Sample realistic tickets across categories."""
    return [
        "I was charged twice for my subscription this month. Please refund.",
        "I forgot my password and cannot access my account anymore.",
        "My parcel tracking has not updated in five days. Where is it?",
        "Getting error code 500 when saving settings on the web dashboard.",
        "Can you send me information about your holiday operating hours?",
    ]


@pytest.fixture
def sample_batch_df():
    """Sample batch DataFrame for testing."""
    return pd.DataFrame({
        "ticket_text": [
            "Unexpected credit card charge on my statement.",
            "Cannot log in to the portal.",
            "Package delivery was delayed by a week.",
            "App crashes on launch.",
            "What are your business support hours?",
        ]
    })
