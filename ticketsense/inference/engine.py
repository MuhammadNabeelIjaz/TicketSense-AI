"""End-to-end inference engine for single messages and CSV batch prediction."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

from ticketsense.config import (
    CATEGORY_MODEL_PATH,
    CATEGORY_METADATA_PATH,
    APPROVED_CATEGORIES,
    DEFAULT_UNCERTAINTY_THRESHOLD,
    DEFAULT_UNCERTAINTY_MARGIN,
)
from ticketsense.models.persistence import load_model_bundle
from ticketsense.nlp.preprocessor import clean_text
from ticketsense.data.validator import validate_ticket_text, validate_batch_dataframe
from ticketsense.inference.uncertainty import compute_uncertainty_score
from ticketsense.inference.priority_stub import assess_priority


def extract_ticket_summary(text: str, max_chars: int = 120) -> str:
    """Extract a concise summary of the customer's issue from the ticket text.

    Args:
        text: Customer message.
        max_chars: Character limit for summary.

    Returns:
        Extracted summary sentence.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    clean = text.strip().replace("\n", " ")
    sentences = [s.strip() for s in clean.split(".") if s.strip()]
    if not sentences:
        return clean[:max_chars]
    first = sentences[0]
    if len(first) > max_chars:
        return first[:max_chars] + "..."
    return first + "."


class InferenceEngine:
    """High-level inference engine for ticket classification."""

    def __init__(
        self,
        model_path: Path = CATEGORY_MODEL_PATH,
        metadata_path: Path = CATEGORY_METADATA_PATH,
        uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
        uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_margin = uncertainty_margin
        self.pipeline = None
        self.metadata = None
        self.classes = APPROVED_CATEGORIES
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load serialized pipeline from disk."""
        if self.model_path.exists():
            self.pipeline, self.metadata = load_model_bundle(
                self.model_path, self.metadata_path
            )
            if hasattr(self.pipeline, "classes_"):
                self.classes = list(self.pipeline.classes_)

    def is_ready(self) -> bool:
        """Check if model pipeline is loaded and ready."""
        return self.pipeline is not None

    def predict_ticket(self, text: str) -> Dict[str, Any]:
        """Predict category, confidence, uncertainty, and priority status for a single ticket.

        Args:
            text: Customer support message.

        Returns:
            Dict containing validation status, prediction, uncertainty, and priority metadata.
        """
        is_valid, error_msg = validate_ticket_text(text)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "ticket_text": text,
            }

        if not self.is_ready():
            return {
                "success": False,
                "error": "Model is not loaded. Train or deploy the model pipeline first.",
                "ticket_text": text,
            }

        cleaned = clean_text(text)
        probas = self.pipeline.predict_proba([cleaned])[0].tolist()
        class_probas = {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes, probas)}

        uncertainty_diag = compute_uncertainty_score(
            probabilities=probas,
            labels=self.classes,
            threshold=self.uncertainty_threshold,
            margin=self.uncertainty_margin,
        )

        priority_info = assess_priority(text)
        summary = extract_ticket_summary(text)

        return {
            "success": True,
            "ticket_text": text,
            "cleaned_text": cleaned,
            "predicted_category": uncertainty_diag["top_category"],
            "confidence": uncertainty_diag["confidence"],
            "all_probabilities": class_probas,
            "is_uncertain": uncertainty_diag["is_uncertain"],
            "uncertainty_reason": uncertainty_diag["reason"],
            "second_category": uncertainty_diag["second_category"],
            "margin": uncertainty_diag["margin"],
            "summary": summary,
            "priority": priority_info["heuristic_priority"],
            "priority_status": priority_info["priority_status"],
            "priority_policy_message": priority_info["policy_message"],
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
    ) -> Tuple[bool, str, pd.DataFrame, Dict[str, Any]]:
        """Run batch inference on a DataFrame of tickets.

        Args:
            df: Uploaded batch DataFrame.

        Returns:
            Tuple of (success: bool, message: str, enriched_df: pd.DataFrame, stats: Dict[str, Any]).
        """
        is_valid, val_msg, valid_df = validate_batch_dataframe(df)
        if not is_valid:
            return False, val_msg, pd.DataFrame(), {}

        if not self.is_ready():
            return False, "Model is not loaded.", pd.DataFrame(), {}

        cleaned_texts = [clean_text(t) for t in valid_df["ticket_text"]]
        # Pipeline accepts raw text list directly via TextCleaner
        all_probas = self.pipeline.predict_proba(valid_df["ticket_text"].tolist())

        results = []
        uncertain_count = 0

        for idx, (orig_text, probas) in enumerate(zip(valid_df["ticket_text"], all_probas)):
            u_diag = compute_uncertainty_score(
                probabilities=probas.tolist(),
                labels=self.classes,
                threshold=self.uncertainty_threshold,
                margin=self.uncertainty_margin,
            )
            p_info = assess_priority(orig_text)
            summary = extract_ticket_summary(orig_text)

            if u_diag["is_uncertain"]:
                uncertain_count += 1

            results.append({
                "predicted_category": u_diag["top_category"],
                "confidence": u_diag["confidence"],
                "is_uncertain": u_diag["is_uncertain"],
                "uncertainty_reason": u_diag["reason"],
                "provisional_priority_heuristic": p_info["heuristic_priority"],
                "priority_status": p_info["priority_status"],
                "issue_summary": summary,
            })

        results_df = pd.DataFrame(results)
        enriched_df = pd.concat([valid_df.reset_index(drop=True), results_df], axis=1)

        category_counts = enriched_df["predicted_category"].value_counts().to_dict()

        stats = {
            "total_processed": len(enriched_df),
            "uncertain_count": uncertain_count,
            "uncertain_percentage": round(uncertain_count / len(enriched_df) * 100, 1),
            "category_distribution": category_counts,
        }

        return True, f"Processed {len(enriched_df)} tickets successfully.", enriched_df, stats
