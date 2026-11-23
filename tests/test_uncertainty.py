"""Unit tests for uncertainty calculation and human review flagging."""

import pytest
from ticketsense.inference.uncertainty import compute_uncertainty_score


def test_uncertainty_low_confidence():
    labels = ["A", "B", "C"]
    probs = [0.40, 0.35, 0.25]
    diag = compute_uncertainty_score(probs, labels, threshold=0.65, margin=0.15)

    assert diag["is_uncertain"] is True
    assert diag["is_low_confidence"] is True
    assert "Low confidence" in diag["reason"]


def test_uncertainty_ambiguous_margin():
    labels = ["A", "B", "C"]
    probs = [0.68, 0.65, 0.10]
    # top prob is 0.68 > 0.65, but margin is 0.03 < 0.15
    diag = compute_uncertainty_score(probs, labels, threshold=0.65, margin=0.15)

    assert diag["is_uncertain"] is True
    assert diag["is_ambiguous"] is True
    assert "Ambiguous" in diag["reason"]


def test_uncertainty_confident_autonomous():
    labels = ["A", "B", "C"]
    probs = [0.90, 0.06, 0.04]
    diag = compute_uncertainty_score(probs, labels, threshold=0.65, margin=0.15)

    assert diag["is_uncertain"] is False
    assert diag["is_low_confidence"] is False
    assert diag["is_ambiguous"] is False
