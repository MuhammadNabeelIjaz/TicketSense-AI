"""Tests for the production data visualization module."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ticketsense.visualization import (
    plot_class_distribution,
    plot_confusion_matrix_heatmap,
    plot_category_performance,
    plot_experiment_comparison,
    plot_confidence_distribution,
    plot_automatic_vs_manual_review,
    save_all_visualizations,
)


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Ensure all matplotlib figures are closed after each test."""
    yield
    plt.close("all")


def test_plot_class_distribution():
    fig = plot_class_distribution()
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert "Dataset Class Distribution" in ax.get_title()
    assert len(ax.patches) == 5  # 5 category bars


def test_plot_confusion_matrix_heatmap():
    fig = plot_confusion_matrix_heatmap()
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert "Confusion Matrix Heatmap" in ax.get_title()
    assert len(ax.get_xticklabels()) == 5
    assert len(ax.get_yticklabels()) == 5


def test_plot_category_performance():
    fig = plot_category_performance()
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert "Category-wise Performance" in ax.get_title()
    assert len(ax.get_xticklabels()) == 5
    # 3 bars per category = 15 bars
    assert len(ax.patches) == 15


def test_plot_experiment_comparison():
    fig = plot_experiment_comparison()
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert "Model Experiment Comparison" in ax.get_title()
    assert len(ax.patches) >= 8  # At least 4 experiments * 2 bars


def test_plot_confidence_distribution():
    sample_conf = [0.85, 0.90, 0.92, 0.70, 0.40, 0.88, 0.95]
    fig = plot_confidence_distribution(confidences=sample_conf, threshold=0.65)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert "Confidence" in ax.get_title()
    # Check threshold line exists
    lines = ax.get_lines()
    assert len(lines) >= 2  # KDE curve, threshold line, and/or mean line


def test_plot_automatic_vs_manual_review():
    fig = plot_automatic_vs_manual_review(auto_count=90, review_count=10)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert "Operational Routing" in ax.get_title()
    # Donut pie chart has 2 wedge patches
    assert len(ax.patches) == 2


def test_save_all_visualizations(tmp_path):
    saved = save_all_visualizations(output_dir=tmp_path, dpi=72)
    assert len(saved) == 6
    for name, path in saved.items():
        assert path.exists()
        assert path.stat().st_size > 0

