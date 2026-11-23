"""Reproducible training pipeline for category classification."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ticketsense.config import (
    APPROVED_CATEGORIES,
    RANDOM_STATE,
    CATEGORY_MODEL_PATH,
    CATEGORY_METADATA_PATH,
    REPORTS_DIR,
)
from ticketsense.data.loader import load_dataset_splits
from ticketsense.nlp.preprocessor import clean_text, TextCleaner
from ticketsense.models.persistence import save_model_bundle
from ticketsense.models.evaluator import (
    evaluate_classification,
    save_confusion_matrix_plot,
    analyze_errors,
)


def build_category_pipeline(
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
    C: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Construct the TF-IDF + Logistic Regression classification pipeline.

    Args:
        ngram_range: n-gram bounds for TF-IDF.
        min_df: minimum document frequency.
        max_df: maximum document frequency proportion.
        C: Inverse regularization parameter for Logistic Regression.
        random_state: Seed for reproducibility.

    Returns:
        scikit-learn Pipeline instance.
    """
    pipeline = Pipeline([
        ("cleaner", TextCleaner()),
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
                solver="lbfgs",
            ),
        ),
    ])
    return pipeline


def train_category_model(
    save_artifacts: bool = True,
    model_path: Path = CATEGORY_MODEL_PATH,
    metadata_path: Path = CATEGORY_METADATA_PATH,
) -> Dict[str, Any]:
    """Train category classification pipeline on train split and evaluate on validation split.

    Strict guardrails:
        - Trains only on train split (400 samples).
        - Tunes and evaluates only on validation split (100 samples).
        - Test set is strictly isolated.

    Returns:
        Dictionary of training metadata and validation evaluation metrics.
    """
    train_df, val_df = load_dataset_splits()

    # Preprocess text
    X_train_raw = train_df["ticket_text"].tolist()
    y_train = train_df["category"].tolist()

    X_val_raw = val_df["ticket_text"].tolist()
    y_val = val_df["category"].tolist()

    # Build and train pipeline directly on raw text strings (TextCleaner is step 1)
    pipeline = build_category_pipeline()
    pipeline.fit(X_train_raw, y_train)

    # Validation predictions and probabilities on raw text
    y_val_pred = pipeline.predict(X_val_raw)
    val_probas = pipeline.predict_proba(X_val_raw)
    val_confidences = np.max(val_probas, axis=1).tolist()

    # Evaluation
    metrics = evaluate_classification(
        y_true=y_val,
        y_pred=y_val_pred.tolist(),
        labels=APPROVED_CATEGORIES,
    )

    # Error analysis
    errors_df = analyze_errors(
        texts=X_val_raw,
        y_true=y_val,
        y_pred=y_val_pred.tolist(),
        confidences=val_confidences,
    )

    metadata = {
        "model_name": "TF-IDF + LogisticRegression Category Classifier",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "classes": APPROVED_CATEGORIES,
        "hyperparameters": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.9,
            "sublinear_tf": True,
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": RANDOM_STATE,
        },
        "validation_metrics": {
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        },
        "errors_count": len(errors_df),
    }

    if save_artifacts:
        save_model_bundle(pipeline, metadata, model_path, metadata_path)

        # Save confusion matrix plot and evaluation report
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        cm_path = REPORTS_DIR / "category_confusion_matrix.png"
        save_confusion_matrix_plot(
            cm=metrics["confusion_matrix"],
            labels=APPROVED_CATEGORIES,
            output_path=cm_path,
            title="Category Validation Confusion Matrix",
        )

        # Save errors table
        errors_path = REPORTS_DIR / "category_validation_errors.csv"
        errors_df.to_csv(errors_path, index=False, encoding="utf-8")

        # Save markdown summary
        report_md_path = REPORTS_DIR / "CATEGORY_MODEL_EVALUATION.md"
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(f"# Category Model Evaluation Report\n\n")
            f.write(f"- **Trained At:** {metadata['timestamp_utc']}\n")
            f.write(f"- **Train Samples:** {metadata['train_samples']}\n")
            f.write(f"- **Validation Samples:** {metadata['val_samples']}\n")
            f.write(f"- **Validation Accuracy:** {metrics['accuracy']:.4f}\n")
            f.write(f"- **Macro F1-Score:** {metrics['macro_f1']:.4f}\n")
            f.write(f"- **Weighted F1-Score:** {metrics['weighted_f1']:.4f}\n\n")
            f.write(f"## Per-Class Performance\n\n")
            f.write("| Category | Precision | Recall | F1-Score | Support |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for cat in APPROVED_CATEGORIES:
                rep = metrics["classification_report"].get(cat, {})
                f.write(
                    f"| {cat} | {rep.get('precision', 0.0):.3f} | "
                    f"{rep.get('recall', 0.0):.3f} | {rep.get('f1-score', 0.0):.3f} | "
                    f"{int(rep.get('support', 0))} |\n"
                )
            f.write(f"\n## Misclassifications\n\n")
            f.write(f"Total validation errors: {len(errors_df)} / {len(val_df)} ({len(errors_df)/len(val_df)*100:.1f}%)\n")

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "metrics": metrics,
        "errors_df": errors_df,
    }
