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
    DEFAULT_PRIORITY_UNCERTAINTY_THRESHOLD,
    DEFAULT_PRIORITY_UNCERTAINTY_MARGIN,
    DEFAULT_PRIORITY_COMPONENT_THRESHOLD,
    DEFAULT_PRIORITY_HIGH_RISK_THRESHOLD,
)
from ticketsense.models.persistence import load_model_bundle
from ticketsense.nlp.preprocessor import clean_text
from ticketsense.data.validator import validate_ticket_text, validate_batch_dataframe
from ticketsense.inference.uncertainty import (
    compute_uncertainty_score,
    compute_priority_uncertainty_score,
)
from ticketsense.inference.priority_stub import assess_priority
from ticketsense.inference.priority_predictor import PriorityPredictor


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
        priority_threshold: float = DEFAULT_PRIORITY_UNCERTAINTY_THRESHOLD,
        priority_margin: float = DEFAULT_PRIORITY_UNCERTAINTY_MARGIN,
        priority_component_threshold: float = DEFAULT_PRIORITY_COMPONENT_THRESHOLD,
        priority_high_risk_threshold: float = DEFAULT_PRIORITY_HIGH_RISK_THRESHOLD,
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_margin = uncertainty_margin
        self.pipeline = None
        self.metadata = None
        self.classes = APPROVED_CATEGORIES
        self.priority_predictor = PriorityPredictor(
            uncertainty_threshold=priority_threshold,
            uncertainty_margin=priority_margin,
            component_threshold=priority_component_threshold,
            high_risk_threshold=priority_high_risk_threshold,
        )
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

        priority_info = self.priority_predictor.predict(text)
        summary = extract_ticket_summary(text)

        cat_review = uncertainty_diag["is_uncertain"]
        prio_review = priority_info.get("priority_review_required", False)
        overall_review = cat_review or prio_review

        return {
            "success": True,
            "ticket_text": text,
            "cleaned_text": cleaned,
            "predicted_category": uncertainty_diag["top_category"],
            "confidence": uncertainty_diag["confidence"],
            "all_probabilities": class_probas,
            "is_uncertain": cat_review,
            "category_review_required": cat_review,
            "uncertainty_reason": uncertainty_diag["reason"],
            "second_category": uncertainty_diag["second_category"],
            "margin": uncertainty_diag["margin"],
            "summary": summary,
            "priority": priority_info["priority"],
            "priority_confidence": priority_info.get("priority_confidence"),
            "priority_probabilities": priority_info.get("priority_probabilities", {}),
            "priority_margin": priority_info.get("priority_margin"),
            "priority_review_required": prio_review,
            "priority_uncertainty_reason": priority_info.get("priority_uncertainty_reason", ""),
            "overall_review_required": overall_review,
            "predicted_impact": priority_info.get("predicted_impact"),
            "impact_confidence": priority_info.get("impact_confidence"),
            "predicted_urgency": priority_info.get("predicted_urgency"),
            "urgency_confidence": priority_info.get("urgency_confidence"),
            "priority_status": priority_info["priority_status"],
            "priority_policy_message": priority_info["policy_message"],
            "provisional_priority_heuristic": priority_info.get("heuristic_fallback"),
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
        is_valid, val_msg, valid_df, rejected_df = validate_batch_dataframe(df)
        if not is_valid:
            stats = {
                "total_uploaded": len(df) if isinstance(df, pd.DataFrame) else 0,
                "rejected_count": len(rejected_df),
                "rejected_df": rejected_df,
            }
            return False, val_msg, pd.DataFrame(), stats

        if not self.is_ready():
            return False, "Model is not loaded.", pd.DataFrame(), {}

        all_probas = self.pipeline.predict_proba(valid_df["ticket_text"].tolist())

        results = []
        overall_uncertain_count = 0
        cat_review_count = 0
        prio_review_count = 0

        for idx, (orig_text, probas) in enumerate(zip(valid_df["ticket_text"], all_probas)):
            u_diag = compute_uncertainty_score(
                probabilities=probas.tolist(),
                labels=self.classes,
                threshold=self.uncertainty_threshold,
                margin=self.uncertainty_margin,
            )
            p_info = self.priority_predictor.predict(orig_text)
            summary = extract_ticket_summary(orig_text)

            cat_rev = u_diag["is_uncertain"]
            prio_rev = p_info.get("priority_review_required", False)
            overall_rev = cat_rev or prio_rev

            if cat_rev:
                cat_review_count += 1
            if prio_rev:
                prio_review_count += 1
            if overall_rev:
                overall_uncertain_count += 1

            results.append({
                "predicted_category": u_diag["top_category"],
                "confidence": u_diag["confidence"],
                "is_uncertain": cat_rev,
                "category_review_required": cat_rev,
                "uncertainty_reason": u_diag["reason"],
                "predicted_impact": p_info.get("predicted_impact"),
                "predicted_urgency": p_info.get("predicted_urgency"),
                "predicted_priority": p_info["priority"],
                "priority_confidence": p_info.get("priority_confidence"),
                "priority_margin": p_info.get("priority_margin"),
                "priority_review_required": prio_rev,
                "priority_uncertainty_reason": p_info.get("priority_uncertainty_reason", ""),
                "overall_review_required": overall_rev,
                "provisional_priority_heuristic": p_info.get("heuristic_fallback"),
                "priority_status": p_info["priority_status"],
                "issue_summary": summary,
            })

        results_df = pd.DataFrame(results)
        enriched_df = pd.concat([valid_df.reset_index(drop=True), results_df], axis=1)

        category_counts = enriched_df["predicted_category"].value_counts().to_dict()
        priority_counts = enriched_df["predicted_priority"].value_counts().to_dict()

        stats = {
            "total_uploaded": len(df),
            "total_processed": len(enriched_df),
            "rejected_count": len(rejected_df),
            "rejected_df": rejected_df,
            "uncertain_count": overall_uncertain_count,
            "uncertain_percentage": round(overall_uncertain_count / len(enriched_df) * 100, 1) if len(enriched_df) > 0 else 0.0,
            "category_review_count": cat_review_count,
            "priority_review_count": prio_review_count,
            "overall_review_count": overall_uncertain_count,
            "autonomous_acceptance_count": len(enriched_df) - overall_uncertain_count,
            "category_distribution": category_counts,
            "priority_distribution": priority_counts,
        }

        return True, f"Processed {len(enriched_df)} tickets successfully.", enriched_df, stats
