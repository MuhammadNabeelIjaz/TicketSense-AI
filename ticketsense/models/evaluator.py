"""Model evaluation and error analysis module."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_classification(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        labels: Optional ordered list of class labels.

    Returns:
        Dict of accuracy, macro/weighted metrics, and per-class metrics.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "labels": labels,
    }


def save_confusion_matrix_plot(
    cm: List[List[int]],
    labels: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Render and save confusion matrix figure.

    Args:
        cm: Confusion matrix grid.
        labels: Class labels.
        output_path: File destination.
        title: Plot title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(cm)

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=str(matrix[i, j]),
                va="center",
                ha="center",
                size="large",
                weight="bold" if matrix[i, j] > 0 else "normal",
                color="white" if matrix[i, j] > (matrix.max() / 2) else "black",
            )

    fig.colorbar(cax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10, weight="bold")
    ax.set_ylabel("True Label", fontsize=10, weight="bold")
    ax.set_title(title, fontsize=12, weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def analyze_errors(
    texts: List[str],
    y_true: List[str],
    y_pred: List[str],
    confidences: List[float],
) -> pd.DataFrame:
    """Identify and tabulate misclassifications for qualitative inspection.

    Args:
        texts: Input ticket texts.
        y_true: True categories.
        y_pred: Predicted categories.
        confidences: Model confidence scores.

    Returns:
        DataFrame containing error cases.
    """
    records = []
    for text, true_lbl, pred_lbl, conf in zip(texts, y_true, y_pred, confidences):
        if true_lbl != pred_lbl:
            records.append({
                "ticket_text": text,
                "true_category": true_lbl,
                "predicted_category": pred_lbl,
                "confidence": round(float(conf), 4),
            })
    return pd.DataFrame(records)
