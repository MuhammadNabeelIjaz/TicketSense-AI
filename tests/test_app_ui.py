"""Automated UI testing for Streamlit app using AppTest."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_streamlit_app_loads_and_renders_tabs():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    assert len(at.tabs) == 4
    tab_labels = [t.label for t in at.tabs]
    assert "Single Ticket" in tab_labels
    assert "Batch Processing" in tab_labels
    assert "Performance & Governance" in tab_labels
    assert "Corrections Log" in tab_labels


def test_streamlit_app_single_ticket_prediction_workflow():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Enter ticket message and click predict
    at.text_area[0].input("I need help with an unexpected charge on my credit card bill.").run()
    at.button[0].click().run()

    assert not at.exception
    metric_labels = [m.label for m in at.metric]
    assert "Predicted Category" in metric_labels
    assert "Model Confidence" in metric_labels
    assert "Triage Status" in metric_labels
    assert "Priority (ML)" in metric_labels
    assert "Predicted Impact" in metric_labels
    assert "Predicted Urgency" in metric_labels


def test_streamlit_app_renders_all_performance_visualizations():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.image) == 6
